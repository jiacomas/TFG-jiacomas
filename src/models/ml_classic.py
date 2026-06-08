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
    tune_thresholds,
)
from src.schema import ProcessedData
from src.utils import MODELS_DIR, ODS_ALL, PROCESSED_ML_DIR, RANDOM_SEED


def _load_train_test(use_mlsmote: bool):
    """Return (X_train, y_train, X_val, y_val, X_test, y_test).

    When `use_mlsmote=True` the pre-augmented TF-IDF matrices saved by
    `03_modelling.ipynb` are loaded directly; otherwise the raw split parquets
    are re-transformed with the persisted TF-IDF vectoriser. The validation
    split is always taken from the real (non-augmented) distribution so the
    per-class decision thresholds are tuned on representative data.
    """
    test_df = pd.read_parquet(f"{PROCESSED_ML_DIR}/split_test.parquet")
    val_df = pd.read_parquet(f"{PROCESSED_ML_DIR}/split_val.parquet")
    with open(f"{PROCESSED_ML_DIR}/tfidf_model.pkl", "rb") as f:
        tfidf = pickle.load(f)
    X_test = tfidf.transform(test_df[ProcessedData.TEXT_ML])
    y_test = test_df[ODS_ALL].values
    X_val = tfidf.transform(val_df[ProcessedData.TEXT_ML])
    y_val = val_df[ODS_ALL].values

    if use_mlsmote:
        X_train = load_npz(f"{PROCESSED_ML_DIR}/X_train_smote.npz")
        y_train = np.load(f"{PROCESSED_ML_DIR}/Y_train_smote.npy")
    else:
        train_df = pd.read_parquet(f"{PROCESSED_ML_DIR}/split_train.parquet")
        X_train = tfidf.transform(train_df[ProcessedData.TEXT_ML])
        y_train = train_df[ODS_ALL].values

    return X_train, y_train, X_val, y_val, X_test, y_test


def _positive_proba(model, X) -> np.ndarray:
    """Stack per-label P(class=1) into an `(n_samples, n_labels)` matrix.

    Both `MultiOutputClassifier` and `RandomForestClassifier` expose
    `predict_proba` as a list of per-label arrays. A label that saw a single
    class during training yields a degenerate one-column array; we resolve its
    lone class so the probability is 0.0 (all-negative) or 1.0 (all-positive).
    """
    proba_list = model.predict_proba(X)
    # Both estimators expose a per-output `classes_` list (one array per label).
    classes_per_label = getattr(model, "classes_", None)
    cols = []
    for j, p in enumerate(proba_list):
        p = np.asarray(p)
        if p.shape[1] >= 2:
            cols.append(p[:, 1])
            continue
        lone_class = classes_per_label[j][0] if classes_per_label is not None else 0
        cols.append(np.full(p.shape[0], 1.0 if lone_class == 1 else 0.0))
    return np.column_stack(cols)


def train_xgboost(
    variant: str = "balanced",
    use_mlsmote: bool = False,
    threshold: float = 0.5,
):
    run = wandb.init(
        entity="TFG-66910",
        project="bopb-ods-multilabel",
        job_type="train",
        name=f"xgboost-{variant}",
        config={
            "architecture": "XGBoost-OvR",
            "variant": variant,
            "use_mlsmote": use_mlsmote,
            "threshold": threshold,
            "learning_rate": 0.3,
            "n_estimators": 100,
            "tree_method": "hist",  # CPU (no CUDA on MacBook Air)
            "n_jobs": -1,
            "random_state": RANDOM_SEED,
        },
    )
    log_environment()

    X_train, y_train, X_val, y_val, X_test, y_test = _load_train_test(use_mlsmote)
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

    # Tune per-class thresholds on the validation split (same procedure as DL).
    val_proba = _positive_proba(model, X_val)
    tuned_thresholds = tune_thresholds(val_proba, y_val, default=threshold)
    wandb.log(
        {f"threshold/{lbl}": float(t) for lbl, t in zip(ODS_ALL, tuned_thresholds)}
    )

    infer_mon = HardwareMonitor().start()
    test_proba = _positive_proba(model, X_test)
    infer_mon.stop("xgboost", phase="inference")

    preds_default = (test_proba >= threshold).astype(int)
    preds_tuned = (test_proba >= tuned_thresholds).astype(int)
    compute_all_metrics(y_test, preds_default, y_proba=test_proba, step_name="test")
    compute_all_metrics(y_test, preds_tuned, y_proba=test_proba, step_name="test_tuned")

    save_path = f"{MODELS_DIR}/xgboost_{variant}.pkl"
    joblib.dump(model, save_path)
    log_model_artifact(save_path, "xgboost")
    run.finish()


def train_random_forest(
    variant: str = "balanced",
    use_mlsmote: bool = False,
    threshold: float = 0.5,
):
    run = wandb.init(
        entity="TFG-66910",
        project="bopb-ods-multilabel",
        job_type="train",
        name=f"random-forest-{variant}",
        config={
            "architecture": "RandomForest",
            "variant": variant,
            "use_mlsmote": use_mlsmote,
            "threshold": threshold,
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
        },
    )
    log_environment()

    X_train, y_train, X_val, y_val, X_test, y_test = _load_train_test(use_mlsmote)
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

    # Tune per-class thresholds on the validation split (same procedure as DL).
    val_proba = _positive_proba(model, X_val)
    tuned_thresholds = tune_thresholds(val_proba, y_val, default=threshold)
    wandb.log(
        {f"threshold/{lbl}": float(t) for lbl, t in zip(ODS_ALL, tuned_thresholds)}
    )

    infer_mon = HardwareMonitor().start()
    test_proba = _positive_proba(model, X_test)
    infer_mon.stop("random_forest", phase="inference")

    preds_default = (test_proba >= threshold).astype(int)
    preds_tuned = (test_proba >= tuned_thresholds).astype(int)
    compute_all_metrics(y_test, preds_default, y_proba=test_proba, step_name="test")
    compute_all_metrics(y_test, preds_tuned, y_proba=test_proba, step_name="test_tuned")

    save_path = f"{MODELS_DIR}/random_forest_{variant}.pkl"
    joblib.dump(model, save_path)
    log_model_artifact(save_path, "random_forest")
    run.finish()


if __name__ == "__main__":
    train_xgboost(variant="balanced")
    # train_random_forest(variant="balanced")

    # train_xgboost(variant="mlsmote", use_mlsmote=True)
    # train_random_forest(variant="mlsmote", use_mlsmote=True)
