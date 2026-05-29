import pickle

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

import wandb
from src.metrics import (
    HardwareMonitor,
    compute_all_metrics,
    log_dataset_stats,
    log_environment,
    log_model_artifact,
)
from src.schema import ProcessedData
from src.utils import MODELS_DIR, ODS_ALL, PROCESSED_ML_DIR, RANDOM_SEED


def _load_train_test(use_mlsmote: bool):
    """Return (X_train, y_train, X_test, y_test).

    When `use_mlsmote=True` the pre-augmented TF-IDF matrices saved by
    `03_modelling.ipynb` are loaded directly; otherwise the raw split parquets
    are re-transformed with the persisted TF-IDF vectoriser.
    """
    test_df = pd.read_parquet(f"{PROCESSED_ML_DIR}/split_test.parquet")
    with open(f"{PROCESSED_ML_DIR}/tfidf_model.pkl", "rb") as f:
        tfidf = pickle.load(f)
    X_test = tfidf.transform(test_df[ProcessedData.TEXT_ML])
    y_test = test_df[ODS_ALL].values

    if use_mlsmote:
        X_train = load_npz(f"{PROCESSED_ML_DIR}/X_train_smote.npz")
        y_train = np.load(f"{PROCESSED_ML_DIR}/Y_train_smote.npy")
    else:
        train_df = pd.read_parquet(f"{PROCESSED_ML_DIR}/split_train.parquet")
        X_train = tfidf.transform(train_df[ProcessedData.TEXT_ML])
        y_train = train_df[ODS_ALL].values

    return X_train, y_train, X_test, y_test


def train_xgboost(variant: str = "balanced", use_mlsmote: bool = False):
    run = wandb.init(
        entity="TFG-66910",
        project="bopb-ods-multilabel",
        job_type="train",
        name=f"xgboost-{variant}",
        config={
            "architecture": "XGBoost-OvR",
            "variant": variant,
            "use_mlsmote": use_mlsmote,
            "learning_rate": 0.3,
            "n_estimators": 100,
            "tree_method": "hist",  # CPU (no CUDA on MacBook Air)
            "n_jobs": -1,
            "random_state": RANDOM_SEED,
        },
    )
    log_environment()

    X_train, y_train, X_test, y_test = _load_train_test(use_mlsmote)
    log_dataset_stats(y_train, y_test)

    # MultiOutputClassifier trains one binary XGBClassifier per ODS label
    base = XGBClassifier(
        tree_method=run.config.tree_method,
        learning_rate=run.config.learning_rate,
        n_estimators=run.config.n_estimators,
        n_jobs=run.config.n_jobs,
        random_state=run.config.random_state,
        eval_metric="logloss",
        verbosity=0,
    )
    model = MultiOutputClassifier(base, n_jobs=1)  # parallelism handled inside XGB

    train_mon = HardwareMonitor().start()
    model.fit(X_train, y_train)
    train_mon.stop("xgboost", phase="train")

    infer_mon = HardwareMonitor().start()
    preds = model.predict(X_test)
    infer_mon.stop("xgboost", phase="inference")
    compute_all_metrics(y_test, preds, step_name="test")

    save_path = f"{MODELS_DIR}/xgboost_{variant}.pkl"
    joblib.dump(model, save_path)
    log_model_artifact(save_path, "xgboost")
    run.finish()


def train_random_forest(variant: str = "balanced", use_mlsmote: bool = False):
    run = wandb.init(
        entity="TFG-66910",
        project="bopb-ods-multilabel",
        job_type="train",
        name=f"random-forest-{variant}",
        config={
            "architecture": "RandomForest",
            "variant": variant,
            "use_mlsmote": use_mlsmote,
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
        },
    )
    log_environment()

    X_train, y_train, X_test, y_test = _load_train_test(use_mlsmote)
    log_dataset_stats(y_train, y_test)

    # RandomForestClassifier handles multi-output natively;
    # class_weight='balanced' is applied per output column.
    model = RandomForestClassifier(
        n_estimators=run.config.n_estimators,
        max_depth=run.config.max_depth,
        min_samples_split=run.config.min_samples_split,
        min_samples_leaf=run.config.min_samples_leaf,
        class_weight=run.config.class_weight,
        random_state=run.config.random_state,
        n_jobs=-1,
        verbose=0,
    )

    train_mon = HardwareMonitor().start()
    model.fit(X_train, y_train)
    train_mon.stop("random_forest", phase="train")

    infer_mon = HardwareMonitor().start()
    preds = model.predict(X_test)
    infer_mon.stop("random_forest", phase="inference")
    compute_all_metrics(y_test, preds, step_name="test")

    save_path = f"{MODELS_DIR}/random_forest_{variant}.pkl"
    joblib.dump(model, save_path)
    log_model_artifact(save_path, "random_forest")
    run.finish()


if __name__ == "__main__":
    train_xgboost(variant="balanced")
    # train_xgboost(variant="mlsmote", use_mlsmote=True)
    # train_random_forest(variant="balanced")
    # train_random_forest(variant="mlsmote", use_mlsmote=True)
