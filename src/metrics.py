import os
import platform
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

import wandb
from src.utils import ODS_ALL


def log_environment() -> dict:
    """Capture the machine running the experiment."""
    env = {
        "env/os": platform.platform(),
        "env/python": platform.python_version(),
        "env/cpu_model": platform.processor() or platform.machine(),
        "env/cpu_logical_cores": psutil.cpu_count(logical=True),
        "env/cpu_physical_cores": psutil.cpu_count(logical=False),
        "env/total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    try:
        import torch

        env["env/torch_version"] = torch.__version__
        if torch.cuda.is_available():
            env["env/accelerator"] = "cuda"
            env["env/gpu_name"] = torch.cuda.get_device_name(0)
            env["env/gpu_total_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            env["env/accelerator"] = "mps"
        else:
            env["env/accelerator"] = "cpu"
    except ImportError:
        env["env/accelerator"] = "cpu"

    if wandb.run is not None:
        wandb.run.config.update(env, allow_val_change=True)
        wandb.log(env)
    return env


def log_dataset_stats(
    y_train,
    y_test,
    target_names: list = ODS_ALL,
    prefix: str = "data",
) -> dict:
    """Multilabel label distribution + imbalance descriptors for train/test.

    Records per-label positive counts so the dashboard can show *how* balanced
    each split is, plus single-number summaries (cardinality, density,
    imbalance ratio) that explain macro-vs-micro gaps in the metrics.
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    pos_train = y_train.sum(axis=0)
    pos_test = y_test.sum(axis=0)
    pos_train_safe = np.maximum(pos_train, 1)

    stats = {
        f"{prefix}/n_train": int(y_train.shape[0]),
        f"{prefix}/n_test": int(y_test.shape[0]),
        f"{prefix}/n_labels": int(y_train.shape[1]),
        f"{prefix}/train_label_cardinality": float(y_train.sum(axis=1).mean()),
        f"{prefix}/test_label_cardinality": float(y_test.sum(axis=1).mean()),
        f"{prefix}/train_label_density": float(
            y_train.sum(axis=1).mean() / y_train.shape[1]
        ),
        f"{prefix}/test_label_density": float(
            y_test.sum(axis=1).mean() / y_test.shape[1]
        ),
        f"{prefix}/imbalance_ratio_train": float(
            pos_train_safe.max() / pos_train_safe.min()
        ),
    }

    rows = []
    for i, name in enumerate(target_names):
        rows.append(
            {
                "label": name,
                "train_positives": int(pos_train[i]),
                "train_prevalence": float(pos_train[i] / max(1, y_train.shape[0])),
                "test_positives": int(pos_test[i]),
                "test_prevalence": float(pos_test[i] / max(1, y_test.shape[0])),
            }
        )
    table = wandb.Table(dataframe=pd.DataFrame(rows))
    wandb.log({f"{prefix}/label_distribution": table, **stats})
    return stats


def compute_all_metrics(
    y_true,
    y_pred,
    y_proba=None,
    step_name: str = "test",
    target_names: list = ODS_ALL,
) -> dict:
    """Multilabel classification metrics logged to W&B."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        f"{step_name}/subset_accuracy": accuracy_score(y_true, y_pred),
        f"{step_name}/hamming_loss": hamming_loss(y_true, y_pred),
        f"{step_name}/f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        f"{step_name}/f1_micro": f1_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        f"{step_name}/f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        f"{step_name}/f1_samples": f1_score(
            y_true, y_pred, average="samples", zero_division=0
        ),
        f"{step_name}/precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        f"{step_name}/precision_micro": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        f"{step_name}/recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        f"{step_name}/recall_micro": recall_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
    }

    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_label_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_label_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    for i, label in enumerate(target_names):
        slug = label.replace(" ", "_")
        metrics[f"{step_name}/f1_{slug}"] = float(per_label_f1[i])
        metrics[f"{step_name}/precision_{slug}"] = float(per_label_p[i])
        metrics[f"{step_name}/recall_{slug}"] = float(per_label_r[i])

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            metrics[f"{step_name}/ap_macro"] = average_precision_score(
                y_true, y_proba, average="macro"
            )
            metrics[f"{step_name}/ap_micro"] = average_precision_score(
                y_true, y_proba, average="micro"
            )
            metrics[f"{step_name}/ap_weighted"] = average_precision_score(
                y_true, y_proba, average="weighted"
            )
        except ValueError:
            pass
        try:
            metrics[f"{step_name}/roc_auc_macro"] = roc_auc_score(
                y_true, y_proba, average="macro"
            )
            metrics[f"{step_name}/roc_auc_micro"] = roc_auc_score(
                y_true, y_proba, average="micro"
            )
        except ValueError:
            # Raised when any label has a single class in y_true.
            pass

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    wandb.log(
        {f"{step_name}/detailed_report": wandb.Table(dataframe=report_df.reset_index())}
    )
    wandb.log(metrics)
    return metrics


class HardwareMonitor:
    """Samples CPU%, RSS and (if torch is available) GPU memory in a thread."""

    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._proc = psutil.Process(os.getpid())
        self._cpu_samples: list[float] = []
        self._rss_samples: list[float] = []
        self._gpu_samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0: float | None = None
        self._rss_t0: float = 0.0
        self._has_cuda = False
        self._has_mps = False
        try:
            import torch

            self._torch = torch
            self._has_cuda = torch.cuda.is_available()
            self._has_mps = bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            )
        except ImportError:
            self._torch = None

    def _sample_gpu_mb(self) -> float:
        if self._has_cuda:
            return self._torch.cuda.memory_allocated() / (1024**2)
        if self._has_mps and hasattr(self._torch.mps, "current_allocated_memory"):
            return self._torch.mps.current_allocated_memory() / (1024**2)
        return 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            self._cpu_samples.append(self._proc.cpu_percent())
            self._rss_samples.append(self._proc.memory_info().rss / (1024**2))
            self._gpu_samples.append(self._sample_gpu_mb())
            time.sleep(self._interval)

    def start(self) -> "HardwareMonitor":
        self._proc.cpu_percent()  # prime - first call always returns 0.0
        self._rss_t0 = self._proc.memory_info().rss / (1024**2)
        if self._has_cuda:
            self._torch.cuda.reset_peak_memory_stats()
        self._stop.clear()
        self._cpu_samples.clear()
        self._rss_samples.clear()
        self._gpu_samples.clear()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self, model_name: str, phase: str = "train") -> dict:
        """Stop monitoring, log to W&B, and return the summary dict."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        wall = time.perf_counter() - (self._t0 or time.perf_counter())

        cpu = self._cpu_samples or [0.0]
        rss = self._rss_samples or [self._rss_t0]
        gpu = self._gpu_samples or [0.0]
        peak_ram = max(rss)
        mean_ram = sum(rss) / len(rss)
        delta_ram = peak_ram - self._rss_t0
        mean_cpu = sum(cpu) / len(cpu)
        max_cpu = max(cpu)
        peak_gpu = max(gpu)

        prefix = f"efficiency/{model_name}/{phase}"
        summary = {
            f"{prefix}/wall_time_sec": wall,
            f"{prefix}/peak_ram_mb": peak_ram,
            f"{prefix}/mean_ram_mb": mean_ram,
            f"{prefix}/delta_ram_mb": delta_ram,
            f"{prefix}/mean_cpu_pct": mean_cpu,
            f"{prefix}/max_cpu_pct": max_cpu,
        }
        if self._has_cuda or self._has_mps:
            summary[f"{prefix}/peak_gpu_mem_mb"] = peak_gpu
            if self._has_cuda:
                summary[f"{prefix}/cuda_peak_alloc_mb"] = (
                    self._torch.cuda.max_memory_allocated() / (1024**2)
                )

        wandb.log(summary)
        gpu_msg = (
            f"  peak_gpu={peak_gpu:.0f} MB" if (self._has_cuda or self._has_mps) else ""
        )
        print(
            f"[{model_name}:{phase}] wall={wall:.1f}s  "
            f"peak_ram={peak_ram:.0f} MB  delta_ram={delta_ram:+.0f} MB  "
            f"mean_cpu={mean_cpu:.1f}%  max_cpu={max_cpu:.1f}%{gpu_msg}"
        )
        return summary


def log_model_artifact(
    path: str | Path,
    model_name: str,
    n_params: int | None = None,
    n_trainable_params: int | None = None,
) -> dict:
    """Record on-disk size (and, for DL, parameter counts) of a trained model."""
    p = Path(path)
    info = {f"model/{model_name}/size_mb": round(p.stat().st_size / (1024**2), 3)}
    if n_params is not None:
        info[f"model/{model_name}/n_params"] = int(n_params)
    if n_trainable_params is not None:
        info[f"model/{model_name}/n_trainable_params"] = int(n_trainable_params)
    wandb.log(info)
    return info
