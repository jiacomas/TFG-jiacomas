import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

import wandb
from src.metrics import (
    HardwareMonitor,
    compute_all_metrics,
    log_dataset_stats,
    log_environment,
    log_model_artifact,
)
from src.schema import ProcessedData
from src.utils import MODELS_DIR, ODS_ALL, PROCESSED_DL_DIR, RANDOM_SEED

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _resolve_device()


class BopbDataset(Dataset):
    """Tokenises BOPB announcements on-the-fly for a HuggingFace encoder.

    Expects a parquet split with `ProcessedData.TEXT_DL` plus the 17 ODS columns.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int,
        target_cols: list = ODS_ALL,
        text_col: str = ProcessedData.TEXT_DL,
    ):
        self.texts = df[text_col].astype(str).tolist()
        self.targets = df[target_cols].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = " ".join(self.texts[idx].split())
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "targets": torch.as_tensor(self.targets[idx]),
        }


class BertMultilabel(nn.Module):
    """BERT/RoBERTa backbone + dropout + linear head with sigmoid via BCEWithLogitsLoss."""

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


def _train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device) -> float:
    model.train()
    losses = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["targets"].to(device)

        logits = model(ids, mask)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def _evaluate(model, loader, loss_fn, device, threshold: float):
    model.eval()
    losses, logits_all, targets_all = [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["targets"].to(device)
        logits = model(ids, mask)
        losses.append(loss_fn(logits, y).item())
        logits_all.append(logits.detach().cpu())
        targets_all.append(y.detach().cpu())

    logits_cat = torch.cat(logits_all)
    probs = torch.sigmoid(logits_cat).numpy()
    preds = (probs >= threshold).astype(int)
    targets = torch.cat(targets_all).numpy().astype(int)
    return float(np.mean(losses)), preds, targets, probs


def train_bert(
    variant: str = "berta",
    base_model_name: str = "projecte-aina/roberta-base-ca-v2",
    max_len: int = 256,
    batch_size: int = 16,
    epochs: int = 4,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    threshold: float = 0.5,
    dropout: float = 0.3,
    num_workers: int = 2,
):
    """Fine-tune a BERT-style encoder on the BOPB-ODS multilabel task.

    Mirrors `train_xgboost` / `train_random_forest` in ml_classic.py: one W&B run
    per call, the same `compute_all_metrics` summary on the test split, hardware
    profiling via `HardwareMonitor`, and the trained model dumped to MODELS_DIR.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    run = wandb.init(
        entity="TFG-66910",
        project="comparation-multilabel",
        job_type="train",
        name=f"bert-{variant}",
        config={
            "architecture": "BERT-multilabel",
            "variant": variant,
            "base_model": base_model_name,
            "max_len": max_len,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "threshold": threshold,
            "dropout": dropout,
            "device": str(DEVICE),
        },
    )
    log_environment()

    train_df = pd.read_parquet(f"{PROCESSED_DL_DIR}/split_train.parquet")
    test_df = pd.read_parquet(f"{PROCESSED_DL_DIR}/split_test.parquet")

    log_dataset_stats(
        train_df[ODS_ALL].values,
        test_df[ODS_ALL].values,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(
        BopbDataset(train_df, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        BopbDataset(test_df, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    model = BertMultilabel(
        base_model_name,
        num_labels=len(ODS_ALL),
        dropout=dropout,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log(
        {
            "model/bert/n_params": n_params,
            "model/bert/n_trainable_params": n_trainable,
        }
    )
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    train_mon = HardwareMonitor().start()

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            DEVICE,
        )
        wandb.log({"train/loss": train_loss, "epoch": epoch})
        print(f"[bert-{variant}] epoch={epoch} train_loss={train_loss:.4f}")

    train_mon.stop("bert", phase="train")

    infer_mon = HardwareMonitor().start()
    test_loss, test_preds, test_targets, test_probs = _evaluate(
        model,
        test_loader,
        loss_fn,
        DEVICE,
        threshold,
    )
    infer_mon.stop("bert", phase="inference")

    wandb.log({"test/loss": test_loss})
    compute_all_metrics(test_targets, test_preds, y_proba=test_probs, step_name="test")

    save_path = f"{MODELS_DIR}/bert_{variant}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": dict(run.config),
            "labels": ODS_ALL,
        },
        save_path,
    )
    log_model_artifact(
        save_path, "bert", n_params=n_params, n_trainable_params=n_trainable
    )
    run.finish()


if __name__ == "__main__":
    train_bert(variant="berta")
