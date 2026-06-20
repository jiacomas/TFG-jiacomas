# ruff: noqa: E402

import os

# Must be set before any ML library imports.
# Forcing single-thread mode here prevents the fork entirely.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_ML_DIR = PROJECT_ROOT / "data" / "processed" / "ml"

ODS_LABELS = [f"ODS {i}" for i in range(1, 18)]
ODS_SHORT = [
    "Fi de la pobresa",
    "Fam zero",
    "Salut i benestar",
    "Educació de qualitat",
    "Igualtat de gènere",
    "Aigua neta i sanejament",
    "Energia neta i assequible",
    "Treball digne i creixement econòmic",
    "Indústria, innovació, infraestructura",
    "Reducció de desigualtats",
    "Ciutats i comunitats sostenibles",
    "Consum i producció responsables",
    "Acció climàtica",
    "Vida submarina",
    "Vida d'ecosistemes terrestres",
    "Pau, justícia i institucions sòlides",
    "Aliances pels objectius",
]

# --- APP -------------------------------------------------------------------

app = FastAPI(title="BOPB-ODS Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Lazy-loaded model cache: key → loaded object
_cache: dict[str, Any] = {}


# --- Text preprocessing (lightweight, no spaCy dependency) -----------------
def preprocess_for_tfidf(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- BertMultilabel architecture (mirrors src/models/deep_learning.py) -----
class BertMultilabel(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = (
            out.pooler_output
            if getattr(out, "pooler_output", None) is not None
            else out.last_hidden_state[:, 0, :]
        )
        return self.classifier(self.dropout(pooled))


# --- Model loaders ---------------------------------------------------------
def _force_single_thread(model) -> None:
    """Patch n_jobs=1 on model and all sub-estimators.

    RandomForestClassifier is saved with n_jobs=-1. On macOS, joblib's loky
    backend forks worker processes — which segfaults after PyTorch is imported
    because fork() doesn't play well with PyTorch's internal CUDA/MPS state.
    Setting n_jobs=1 forces sequential execution and avoids the fork entirely.
    """
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    for est in getattr(model, "estimators_", []):
        if hasattr(est, "n_jobs"):
            est.n_jobs = 1


def _load_ml_models():
    if "tfidf" not in _cache:
        with open(PROCESSED_ML_DIR / "tfidf_model.pkl", "rb") as f:
            _cache["tfidf"] = pickle.load(f)
    if "rf" not in _cache:
        rf = joblib.load(MODELS_DIR / "random_forest_balanced.pkl")
        _force_single_thread(rf)
        _cache["rf"] = rf
    if "xgb" not in _cache:
        xgb_model = joblib.load(MODELS_DIR / "xgboost_balanced.pkl")
        _force_single_thread(xgb_model)
        _cache["xgb"] = xgb_model


def _load_dl_model(model_key: str):
    if model_key in _cache:
        return

    ckpt_path = MODELS_DIR / (
        "bert_berta_v2.pt" if model_key == "berta" else "mmbert_base.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    base_model_name = config.get("base_model") or config.get("base_model_name")
    if not base_model_name:
        raise ValueError(f"base_model not found in checkpoint config for {model_key}")

    thresholds = np.array(ckpt.get("thresholds", [0.5] * 17), dtype=np.float32)
    max_len = int(config.get("max_len", 256))
    dropout = float(config.get("dropout", 0.3))

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = BertMultilabel(base_model_name, num_labels=17, dropout=dropout)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)

    _cache[model_key] = {
        "model": model,
        "tokenizer": tokenizer,
        "thresholds": thresholds,
        "max_len": max_len,
        "device": device,
        "base_model": base_model_name,
    }


# --- Inference helpers ------------------------------------------------------
def _predict_ml(text: str, model_key: str) -> tuple[list[int], dict[int, float]]:
    _load_ml_models()
    X = _cache["tfidf"].transform([preprocess_for_tfidf(text)])
    model = _cache[model_key]

    # predict_proba returns list of (n_samples, n_classes) arrays — one per label
    probas_list = model.predict_proba(X)
    confidences: dict[int, float] = {}
    for i, arr in enumerate(probas_list):
        # arr shape: (1, 2) for binary; take P(class=1)
        p = float(arr[0, 1]) if arr.shape[1] == 2 else float(arr[0, 0])
        confidences[i + 1] = round(p, 4)

    preds_raw = model.predict(X)[0]  # shape (17,)
    predicted = [i + 1 for i, v in enumerate(preds_raw) if v == 1]
    return predicted, confidences


def _predict_dl(text: str, model_key: str) -> tuple[list[int], dict[int, float]]:
    _load_dl_model(model_key)
    c = _cache[model_key]
    model = c["model"]
    tokenizer = c["tokenizer"]
    thresholds = c["thresholds"]
    max_len = c["max_len"]
    device = c["device"]

    enc = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    confidences = {i + 1: round(float(probs[i]), 4) for i in range(17)}
    predicted = [i + 1 for i in range(17) if probs[i] >= thresholds[i]]
    return predicted, confidences


# --- Reasoning generator (no LLM needed) -----------------------------------
def _make_reasoning(
    model_key: str, predicted: list[int], confidences: dict[int, float]
) -> str:
    model_names = {
        "rf": "Random Forest (TF-IDF)",
        "xgb": "XGBoost (TF-IDF)",
        "berta": "BERTa (RoBERTa català)",
        "mmbert": "mmBERT (ModernBERT multilingüe)",
    }
    name = model_names.get(model_key, model_key)

    if not predicted:
        if model_key == "rf":
            return f"{name}: cap ODS supera el llindar de decisió. El model és molt conservador (F1-macro 0.315) i tendeix a no predir classes minoritàries."
        return f"{name}: cap ODS supera el llindar ajustat per validació. El text pot ser massa curt o ambigu per al model."

    top = sorted(predicted, key=lambda n: confidences.get(n, 0), reverse=True)
    ods_names = ", ".join(f"ODS {n} ({ODS_SHORT[n - 1]})" for n in top[:3])
    top_conf = confidences.get(top[0], 0)

    if model_key in ("rf", "xgb"):
        return (
            f"{name}: ha detectat paraules clau associades a {ods_names}. "
            f"Predicció basada en la representació TF-IDF (freqüència de termes, sense context semàntic). "
            f"Confiança màxima: {top_conf * 100:.0f}%."
        )
    return (
        f"{name}: ha analitzat el context semàntic del text i ha identificat {ods_names} com a objectius rellevants. "
        f"Llindars ajustats per ODS sobre el conjunt de validació. "
        f"Confiança màxima: {top_conf * 100:.0f}%."
    )


# --- Endpoints ------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str
    model: str  # rf | xgb | berta | mmbert


class PredictResponse(BaseModel):
    predicted: list[int]
    confidences: dict[str, float]
    reasoning: str


@app.get("/health")
def health():
    loaded = list(_cache.keys())
    return {"status": "ok", "loaded_models": loaded}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.model not in ("rf", "xgb", "berta", "mmbert"):
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")

    text = req.text[:8000].strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")

    try:
        if req.model in ("rf", "xgb"):
            predicted, confidences = _predict_ml(text, req.model)
        else:
            predicted, confidences = _predict_dl(text, req.model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    reasoning = _make_reasoning(req.model, predicted, confidences)
    str_confidences = {str(k): v for k, v in confidences.items()}

    return PredictResponse(
        predicted=predicted,
        confidences=str_confidences,
        reasoning=reasoning,
    )
