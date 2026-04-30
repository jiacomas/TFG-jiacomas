import os
import pickle
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

import wandb
from src.metrics import HardwareMonitor, compute_all_metrics
from src.schema import ProcessedData
from src.utils import MODELS_DIR, ODS_ALL, PROCESSED_ML_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def train_xgboost(variant: str = "raw"):
    run = wandb.init(
        entity="TFG-66910",
        project="bopb-ods-multilabel",
        job_type="train",
        name=f"xgboost-{variant}",
        config={
            "architecture": "XGBoost-OvR",
            "variant": variant,
            "learning_rate": 0.3,
            "n_estimators": 100,
            "tree_method": "hist",  # CPU (no CUDA on MacBook Air)
            "n_jobs": -1,
        },
    )

    train_df = pd.read_parquet(f"{PROCESSED_ML_DIR}/split_train.parquet")
    test_df = pd.read_parquet(f"{PROCESSED_ML_DIR}/split_test.parquet")

    with open(os.path.join(PROCESSED_ML_DIR, "tfidf_model.pkl"), "rb") as f:
        tfidf = pickle.load(f)

    X_train = tfidf.transform(train_df[ProcessedData.TEXT_ML])
    X_test = tfidf.transform(test_df[ProcessedData.TEXT_ML])
    y_train = train_df[ODS_ALL]
    y_test = test_df[ODS_ALL]

    # MultiOutputClassifier trains one binary XGBClassifier per ODS label
    base = XGBClassifier(
        tree_method=run.config.tree_method,
        learning_rate=run.config.learning_rate,
        n_estimators=run.config.n_estimators,
        n_jobs=run.config.n_jobs,
        eval_metric="logloss",
        verbosity=0,
    )
    model = MultiOutputClassifier(base, n_jobs=1)  # parallelism handled inside XGB

    monitor = HardwareMonitor().start()
    model.fit(X_train, y_train.values)
    monitor.stop("xgboost")

    preds = model.predict(X_test)
    compute_all_metrics(y_test.values, preds, step_name="test")

    joblib.dump(model, f"{MODELS_DIR}/xgboost_{variant}.pkl")
    run.finish()


def train_random_forest(variant: str = "balanced"):
    run = wandb.init(
        entity="TFG-66910",
        project="bopb-ods-multilabel",
        job_type="train",
        name=f"random-forest-{variant}",
        config={
            "architecture": "RandomForest",
            "variant": variant,
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
        },
    )

    train_df = pd.read_parquet(f"{PROCESSED_ML_DIR}/split_train.parquet")
    test_df = pd.read_parquet(f"{PROCESSED_ML_DIR}/split_test.parquet")

    with open(os.path.join(PROCESSED_ML_DIR, "tfidf_model.pkl"), "rb") as f:
        tfidf = pickle.load(f)

    X_train = tfidf.transform(train_df[ProcessedData.TEXT_ML])
    X_test = tfidf.transform(test_df[ProcessedData.TEXT_ML])
    y_train = train_df[ODS_ALL]
    y_test = test_df[ODS_ALL]

    # RandomForestClassifier handles multi-output natively;
    # class_weight='balanced' is applied per output column.
    model = RandomForestClassifier(
        n_estimators=run.config.n_estimators,
        max_depth=run.config.max_depth,
        min_samples_split=run.config.min_samples_split,
        min_samples_leaf=run.config.min_samples_leaf,
        class_weight=run.config.class_weight,
        n_jobs=-1,
        verbose=0,
    )

    monitor = HardwareMonitor().start()
    model.fit(X_train, y_train.values)
    monitor.stop("random_forest")

    preds = model.predict(X_test)
    compute_all_metrics(y_test.values, preds, step_name="test")

    joblib.dump(model, f"{MODELS_DIR}/random_forest_{variant}.pkl")
    run.finish()


if __name__ == "__main__":
    train_xgboost(variant="balanced")
    # train_random_forest(variant="balanced")
