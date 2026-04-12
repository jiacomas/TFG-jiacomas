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


def compute_all_metrics(y_true, y_pred, y_prob=None, step_name="test"):
    """
    Calculates multilabel classification metrics and logs them to W&B.

    Args:
        y_true: Ground truth labels (binary matrix).
        y_pred: Predicted labels (binary matrix).
        y_prob: Predicted probabilities (optional).
        step_name: Label for the current step (e.g., 'train', 'val', 'test').
    """

    # --- Efficiency & Performance Metrics ---
    # Subset Accuracy: Strict match (all labels must be correct)
    # Hamming Loss: Fraction of labels that are incorrectly predicted (ideal for multilabel)
    metrics = {
        f"{step_name}/accuracy_subset": accuracy_score(y_true, y_pred),
        f"{step_name}/f1_macro": f1_score(y_true, y_pred, average="macro"),
        f"{step_name}/f1_micro": f1_score(y_true, y_pred, average="micro"),
        f"{step_name}/precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        f"{step_name}/recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        f"{step_name}/hamming_loss": hamming_loss(y_true, y_pred),
    }

    # --- Per-Class Detailed Report ---
    # Generate a dictionary and convert to DataFrame for W&B Table logging
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()

    # Log the detailed report table to W&B
    wandb.log(
        {f"{step_name}/detailed_report": wandb.Table(dataframe=report_df.reset_index())}
    )

    # Log scalar metrics
    wandb.log(metrics)

    return metrics


def log_efficiency(start_time, end_time, model_name):
    """
    Logs computational resources used during the process.
    """
    duration = end_time - start_time
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)

    wandb.log(
        {
            f"efficiency/{model_name}_duration_sec": duration,
            f"efficiency/{model_name}_memory_mb": memory_mb,
        }
    )
