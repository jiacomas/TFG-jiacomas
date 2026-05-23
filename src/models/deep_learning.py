import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
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
    """BOPB announcements for a HuggingFace encoder.

    Two construction modes:
      * `__init__(df, tokenizer, max_len, ...)` — tokenise on-the-fly from a parquet split
        (`ProcessedData.TEXT_DL` plus the 17 ODS columns).
      * `BopbDataset.from_pretokenized(path)` — load a pre-tokenised `.pt` produced by
        notebook 03 (keys: `input_ids`, `attention_mask`, `targets`). Skips per-epoch
        tokenisation entirely — faster, but the file must have been generated with the
        same checkpoint as the model being trained.
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        tokenizer=None,
        max_len: int | None = None,
        target_cols: list = ODS_ALL,
        text_col: str = ProcessedData.TEXT_DL,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ):
        if input_ids is not None:
            self.mode = "pretokenized"
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.targets = targets.to(torch.float32)
        else:
            self.mode = "on_the_fly"
            self.texts = df[text_col].astype(str).tolist()
            self.targets = df[target_cols].values.astype(np.float32)
            self.tokenizer = tokenizer
            self.max_len = max_len

    @classmethod
    def from_pretokenized(cls, path: str) -> "BopbDataset":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            input_ids=payload["input_ids"],
            attention_mask=payload["attention_mask"],
            targets=payload["targets"],
        )

    def __len__(self) -> int:
        if self.mode == "pretokenized":
            return self.input_ids.shape[0]
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        if self.mode == "pretokenized":
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "targets": self.targets[idx],
            }
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


def _compute_pos_weight(y: np.ndarray, cap: float = 20.0) -> torch.Tensor:
    """`(neg / pos)` per label, capped to avoid runaway weights on ultra-rare classes."""
    pos = y.sum(axis=0).astype(np.float64)
    neg = y.shape[0] - pos
    pos_safe = np.maximum(pos, 1.0)
    pw = np.minimum(neg / pos_safe, cap)
    return torch.tensor(pw, dtype=torch.float32)


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
def _forward_split(model, loader, loss_fn, device):
    """Return (mean_loss, probs, targets) for a whole loader."""
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

    probs = torch.sigmoid(torch.cat(logits_all)).numpy()
    targets = torch.cat(targets_all).numpy().astype(int)
    return float(np.mean(losses)), probs, targets


def _tune_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    grid: np.ndarray | None = None,
    default: float = 0.5,
) -> np.ndarray:
    """Per-class threshold that maximises F1 on a held-out split.

    Falls back to `default` for labels with zero positives (grid search degenerate).
    """
    if grid is None:
        grid = np.arange(0.05, 0.95 + 1e-9, 0.05)
    n_labels = probs.shape[1]
    best = np.full(n_labels, default, dtype=np.float32)
    for j in range(n_labels):
        y_j = targets[:, j]
        if y_j.sum() == 0:
            continue
        p_j = probs[:, j]
        best_f1, best_t = -1.0, default
        for t in grid:
            pred = (p_j >= t).astype(int)
            f1 = f1_score(y_j, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        best[j] = best_t
    return best


def train_bert(
    variant: str = "berta_v2",
    base_model_name: str = "projecte-aina/roberta-base-ca-v2",
    max_len: int = 256,
    batch_size: int = 16,
    epochs: int = 6,
    patience: int = 2,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    threshold: float = 0.5,
    dropout: float = 0.3,
    num_workers: int = 2,
    pos_weight_cap: float = 20.0,
    monitor_metric: str = "val/f1_macro",
    family: str = "bert",
    architecture: str = "BERT-multilabel",
    use_pretokenized: bool = False,
):
    """Fine-tune a BERT-style encoder on the BOPB-ODS multilabel task.

    `family` namespaces filenames, W&B run name, hardware monitor labels and
    metric prefixes so multiple encoder families (BERTa, mmBERT, …) can share
    this trainer while keeping their results clearly separated in the dashboard.

    Improvements over v1 (`bert_berta`):
      * uses `split_val.parquet` for early stopping + threshold tuning
      * `pos_weight` BCE loss (capped) so rare ODS labels are not ignored
      * keeps the best-by-val checkpoint in memory and restores it before test
      * tunes per-class decision thresholds on val before scoring test
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    run = wandb.init(
        entity="TFG-66910",
        project="comparation-multilabel",
        job_type="train",
        name=f"{family}-{variant}",
        config={
            "architecture": architecture,
            "family": family,
            "variant": variant,
            "base_model": base_model_name,
            "max_len": max_len,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "threshold": threshold,
            "dropout": dropout,
            "pos_weight_cap": pos_weight_cap,
            "monitor_metric": monitor_metric,
            "device": str(DEVICE),
            "use_pretokenized": use_pretokenized,
        },
    )
    log_environment()

    pin = DEVICE.type == "cuda"

    if use_pretokenized:
        train_ds = BopbDataset.from_pretokenized(
            f"{PROCESSED_DL_DIR}/tokens_{family}_train.pt"
        )
        val_ds = BopbDataset.from_pretokenized(
            f"{PROCESSED_DL_DIR}/tokens_{family}_val.pt"
        )
        test_ds = BopbDataset.from_pretokenized(
            f"{PROCESSED_DL_DIR}/tokens_{family}_test.pt"
        )
        Y_train_np = train_ds.targets.numpy().astype(np.int64)
        Y_test_np = test_ds.targets.numpy().astype(np.int64)
    else:
        train_df = pd.read_parquet(f"{PROCESSED_DL_DIR}/split_train.parquet")
        val_df = pd.read_parquet(f"{PROCESSED_DL_DIR}/split_val.parquet")
        test_df = pd.read_parquet(f"{PROCESSED_DL_DIR}/split_test.parquet")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        train_ds = BopbDataset(train_df, tokenizer, max_len)
        val_ds = BopbDataset(val_df, tokenizer, max_len)
        test_ds = BopbDataset(test_df, tokenizer, max_len)
        Y_train_np = train_df[ODS_ALL].values
        Y_test_np = test_df[ODS_ALL].values

    log_dataset_stats(Y_train_np, Y_test_np)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
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
            f"model/{family}/n_params": n_params,
            f"model/{family}/n_trainable_params": n_trainable,
        }
    )

    pos_weight = _compute_pos_weight(
        Y_train_np.astype(np.float32),
        cap=pos_weight_cap,
    ).to(DEVICE)
    wandb.log(
        {
            f"train/pos_weight/{lbl}": float(w)
            for lbl, w in zip(ODS_ALL, pos_weight.cpu())
        }
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

    best_score = -float("inf")
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, DEVICE
        )
        val_loss, val_probs, val_targets = _forward_split(
            model, val_loader, loss_fn, DEVICE
        )
        val_preds = (val_probs >= threshold).astype(int)
        val_f1_macro = f1_score(
            val_targets, val_preds, average="macro", zero_division=0
        )
        val_f1_micro = f1_score(
            val_targets, val_preds, average="micro", zero_division=0
        )

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/f1_macro": val_f1_macro,
                "val/f1_micro": val_f1_micro,
            }
        )
        print(
            f"[{family}-{variant}] epoch={epoch} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1_macro={val_f1_macro:.4f} val_f1_micro={val_f1_micro:.4f}"
        )

        score = val_f1_macro if monitor_metric == "val/f1_macro" else val_f1_micro
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(
                    f"[{family}-{variant}] early stop at epoch {epoch} "
                    f"(best epoch={best_epoch}, best {monitor_metric}={best_score:.4f})"
                )
                break

    train_mon.stop(family, phase="train")

    if best_state is not None:
        model.load_state_dict(best_state)
    wandb.log(
        {
            "best_val/epoch": best_epoch,
            f"best_{monitor_metric}": best_score,
        }
    )

    _, val_probs, val_targets = _forward_split(model, val_loader, loss_fn, DEVICE)
    tuned_thresholds = _tune_thresholds(val_probs, val_targets, default=threshold)
    wandb.log(
        {f"threshold/{lbl}": float(t) for lbl, t in zip(ODS_ALL, tuned_thresholds)}
    )

    infer_mon = HardwareMonitor().start()
    test_loss, test_probs, test_targets = _forward_split(
        model, test_loader, loss_fn, DEVICE
    )
    infer_mon.stop(family, phase="inference")

    test_preds_default = (test_probs >= threshold).astype(int)
    test_preds_tuned = (test_probs >= tuned_thresholds).astype(int)

    wandb.log({"test/loss": test_loss})
    compute_all_metrics(
        test_targets, test_preds_default, y_proba=test_probs, step_name="test"
    )
    compute_all_metrics(
        test_targets, test_preds_tuned, y_proba=test_probs, step_name="test_tuned"
    )

    save_path = f"{MODELS_DIR}/{family}_{variant}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": dict(run.config),
            "labels": ODS_ALL,
            "thresholds": tuned_thresholds.tolist(),
            "best_epoch": best_epoch,
            f"best_{monitor_metric}": best_score,
        },
        save_path,
    )
    log_model_artifact(
        save_path, family, n_params=n_params, n_trainable_params=n_trainable
    )
    run.finish()


def train_mmbert(
    variant: str = "base",
    base_model_name: str = "jhu-clsp/mmBERT-base",
    **kwargs,
):
    """Fine-tune mmBERT (multilingual ModernBERT, jhu-clsp) on BOPB-ODS.

    Delegates to `train_bert` with the same training recipe (max_len, batch,
    epochs, pos_weight BCE, per-class threshold tuning, early stopping) so the
    resulting metrics share scale and methodology with the BERTa baseline and
    the classical ML models. Any keyword overrides flow through `**kwargs`.
    """
    return train_bert(
        variant=variant,
        base_model_name=base_model_name,
        family="mmbert",
        architecture="mmBERT-multilabel",
        **kwargs,
    )


if __name__ == "__main__":
    # train_bert(variant="berta_v2")
    train_mmbert(variant="base")
