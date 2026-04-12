import time

import joblib
import pandas as pd
from xgboost import XGBClassifier

import wandb
from src.metrics import compute_all_metrics, log_efficiency
from src.utils import MODELS_DIR, ODS_ALL, PROCESSED_OUTPUT_DIR


def train_xgboost():
    # Initialize W&B Run
    run = wandb.init(
        entity="TFG-66910",
        project="bopb-ods-multilabel",
        job_type="train",
        name="xgboost-multilabel",
        config={
            "learning_rate": 0.3,  # Hyperparameter for XGBoost
            "architecture": "XGBoost",
            "dataset": "train.parquet",
            "n_estimators": 100,  # Number of boosting rounds
            "tree_method": "gpu_hist",  # Crucial for using the 3090 GPU
        },
    )

    # Load preprocessed datasets
    train_df = pd.read_parquet(f"{PROCESSED_OUTPUT_DIR}/train.parquet")
    test_df = pd.read_parquet(f"{PROCESSED_OUTPUT_DIR}/test.parquet")

    X_train, y_train = train_df.drop(columns=ODS_ALL), train_df[ODS_ALL]
    X_test, y_test = test_df.drop(columns=ODS_ALL), test_df[ODS_ALL]

    # Model definition: XGBoost with GPU acceleration
    model = XGBClassifier(
        tree_method=run.config.tree_method,
        predictor="gpu_predictor",
        learning_rate=run.config.learning_rate,
        n_estimators=run.config.n_estimators,
        callbacks=[wandb.xgboost.WandbCallback()],
    )

    # Train and measure time
    start_t = time.time()
    model.fit(X_train, y_train)
    end_t = time.time()

    # Log computational efficiency
    log_efficiency(start_t, end_t, "xgboost")

    # Evaluation
    preds = model.predict(X_test)
    compute_all_metrics(y_test, preds, step_name="test")

    # Save artifact
    joblib.dump(model, f"{MODELS_DIR}/xgboost_multilabel.pkl")
    run.finish()


def train_random_forest():
    # Placeholder for Random Forest training
    pass


if __name__ == "__main__":
    train_xgboost()
