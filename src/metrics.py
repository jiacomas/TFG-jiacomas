import os
import threading
import time

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

import wandb
from src.utils import ODS_ALL


def compute_all_metrics(
    y_true, y_pred, step_name: str = "test", target_names: list = ODS_ALL
):
    """Multilabel classification metrics logged to W&B.

    Accepts numpy arrays or DataFrames for y_true / y_pred.
    """
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
        f"{step_name}/precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        f"{step_name}/recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }

    # Per-label F1 as individual scalars so each ODS is a separate series in W&B
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for label, score in zip(target_names, per_label_f1):
        metrics[f"{step_name}/f1_{label.replace(' ', '_')}"] = float(score)

    # Per-label detailed report as a W&B Table
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
    """Samples CPU% and RSS every `interval` seconds in a background thread.

    Usage:
        monitor = HardwareMonitor().start()
        model.fit(...)
        monitor.stop("random_forest")
    """

    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._proc = psutil.Process(os.getpid())
        self._cpu_samples: list[float] = []
        self._rss_samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0: float | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._cpu_samples.append(self._proc.cpu_percent())
            self._rss_samples.append(self._proc.memory_info().rss / (1024**2))
            time.sleep(self._interval)

    def start(self) -> "HardwareMonitor":
        self._proc.cpu_percent()  # prime — first call always returns 0.0
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self, model_name: str) -> float:
        """Stop monitoring, log to W&B, and return wall time in seconds."""
        self._stop.set()
        self._thread.join()
        wall = time.perf_counter() - self._t0

        cpu = self._cpu_samples or [0.0]
        rss = self._rss_samples or [0.0]
        mean_cpu = sum(cpu) / len(cpu)
        max_cpu = max(cpu)
        peak_ram = max(rss)

        wandb.log(
            {
                f"efficiency/{model_name}/wall_time_sec": wall,
                f"efficiency/{model_name}/peak_ram_mb": peak_ram,
                f"efficiency/{model_name}/mean_cpu_pct": mean_cpu,
                f"efficiency/{model_name}/max_cpu_pct": max_cpu,
            }
        )
        print(
            f"[{model_name}] wall={wall:.1f}s  "
            f"peak_ram={peak_ram:.0f} MB  "
            f"mean_cpu={mean_cpu:.1f}%  "
            f"max_cpu={max_cpu:.1f}%"
        )
        return wall
