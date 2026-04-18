import os
import pickle
import time

import joblib
import pandas as pd
from xgboost import XGBClassifier

import wandb
from src.metrics import compute_all_metrics, log_efficiency
from src.utils import MODELS_DIR, ODS_ALL, PROCESSED_ML_DIR


def train_xgboost():
    run = wandb.init(
        entity="TFG-66910",
        project="bopb-ods-multilabel",
        job_type="train",
        name="xgboost-multilabel",
        config={
            "learning_rate": 0.3,
            "architecture": "XGBoost",
            "dataset": "train.parquet",
            "n_estimators": 100,
            "tree_method": "gpu_hist",
        },
    )

    # Load data
    train_df = pd.read_parquet(f"{PROCESSED_ML_DIR}/split_train.parquet")
    test_df = pd.read_parquet(f"{PROCESSED_ML_DIR}/split_test.parquet")

    X_train_text = train_df["text_ml"]
    X_test_text = test_df["text_ml"]

    y_train = train_df[ODS_ALL]
    y_test = test_df[ODS_ALL]

    # Load TF-IDF + MLB (ja entrenats)
    with open(os.path.join(PROCESSED_ML_DIR, "tfidf_model.pkl"), "rb") as f:
        tfidf = pickle.load(f)

    # with open(os.path.join(PROCESSED_ML_DIR, "mlb.pkl"), "rb") as f:
    #     mlb = pickle.load(f)

    # Vectorització (IMPORTANT: només transform)
    X_train = tfidf.transform(X_train_text)
    X_test = tfidf.transform(X_test_text)

    # (Opcional) assegurar format correcte de y
    # y_train = mlb.transform(y_train)  # només si NO està ja binaritzat
    # y_test = mlb.transform(y_test)

    # Load Model
    model = XGBClassifier(
        tree_method="hist",
        device="cuda",
        predictor="gpu_predictor",
        learning_rate=run.config.learning_rate,
        n_estimators=run.config.n_estimators,
        callbacks=[wandb.xgboost.WandbCallback()],
    )

    # Train
    start_t = time.time()
    model.fit(X_train, y_train)
    end_t = time.time()

    log_efficiency(start_t, end_t, "xgboost")

    # Evaluate + log metrics
    preds = model.predict(X_test)
    compute_all_metrics(y_test, preds, step_name="test")

    # Save model
    joblib.dump(model, f"{MODELS_DIR}/xgboost_multilabel.pkl")

    run.finish()


def train_random_forest():
    # Placeholder for Random Forest training
    pass


if __name__ == "__main__":
    train_xgboost()
