import os
import pickle
import re
import sys
import unicodedata

import pandas as pd
import spacy
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from schema import Description, Metadata, ProcessedData
from utils import (
    OUTPUT_DIR,
    PROCESSED_DL_DIR,
    PROCESSED_ML_DIR,
    RANDOM_SEED,
    RAW_METADATA_PATH,
    RAW_TEXT_CSV_DIR,
    TEST_SIZE,
    VAL_SIZE,
)

sys.path.append("../src")

os.makedirs(PROCESSED_ML_DIR, exist_ok=True)
os.makedirs(PROCESSED_DL_DIR, exist_ok=True)

nlp = spacy.load("ca_core_news_lg")


def clean_base(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    return text


def process_ml(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zàáèéíïòóúüçñ·\s]", " ", text)
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc if not token.is_stop and len(token.text) > 2
    ]
    return " ".join(tokens)


def export_data(dataframe, y_matrix, indices, folder, text_col, ods_cols):
    t_idx, v_idx, ts_idx = indices
    for name, idx in zip(["train", "val", "test"], [t_idx, v_idx, ts_idx]):
        subset = dataframe.iloc[idx][[text_col, Metadata.ODS_LIST]]
        labels_df = pd.DataFrame(y_matrix[idx], columns=ods_cols, index=subset.index)
        pd.concat([subset, labels_df], axis=1).to_parquet(
            os.path.join(folder, f"split_{name}.parquet"), index=False
        )


def main():
    # 1.1 Load Metadata
    df_meta = pd.read_excel(RAW_METADATA_PATH, dtype=str)
    df_meta[Metadata.ODS] = df_meta[Metadata.ODS].str[:6].str.strip()

    df_grouped_ods = (
        df_meta.groupby(Metadata.ID)[Metadata.ODS]
        .apply(lambda x: list(set(v for v in x if pd.notna(v))))
        .reset_index()
    )
    df_grouped_ods.rename(columns={Metadata.ODS: ProcessedData.ODS_LIST}, inplace=True)
    df_meta_unique = df_meta.drop(columns=[Metadata.ODS]).drop_duplicates(
        subset=[Metadata.ID]
    )
    df_meta = df_meta_unique.merge(df_grouped_ods, on=Metadata.ID, how="left")

    # 1.2 Load Texts from multiple CSVs
    csv_files = [
        os.path.join(RAW_TEXT_CSV_DIR, f)
        for f in os.listdir(RAW_TEXT_CSV_DIR)
        if f.endswith(".csv")
    ]
    df_texts = pd.concat(
        [pd.read_csv(f, dtype=str) for f in csv_files], ignore_index=True
    )

    # 1.3 Merge Metadata and Texts on ID
    df = df_meta.merge(
        df_texts[[Description.ID_REGISTRE, Description.TEXT]].rename(
            columns={Description.ID_REGISTRE: Metadata.ID_REGISTRE}
        ),
        on=Metadata.ID_REGISTRE,
        how="left",
    )

    # Create full_text field (Title + Body)
    df[ProcessedData.FULL_TEXT] = (
        df[Metadata.TITLE].fillna("") + " " + df[Description.TEXT].fillna("")
    )

    # Export df for EDA
    df.to_parquet(os.path.join(OUTPUT_DIR, "full_dataset.parquet"), index=False)

    print(f"Initial dataset size: {len(df)}")

    # Filter: Keep only announcements with at least one ODS
    df_filtered = df[df[ProcessedData.ODS_LIST].apply(len) > 0].copy()
    print(f"Filtered dataset size (announcements with ODSs): {len(df_filtered)}")

    print("Processing texts for DL and ML paths...")
    df_filtered[ProcessedData.FULL_TEXT] = df_filtered[ProcessedData.FULL_TEXT].apply(
        clean_base
    )
    df_filtered[ProcessedData.TEXT_DL] = df_filtered[
        ProcessedData.FULL_TEXT
    ].str.strip()
    df_filtered[ProcessedData.TEXT_ML] = df_filtered[ProcessedData.FULL_TEXT].apply(
        process_ml
    )

    # Label Binarization
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df_filtered[Metadata.ODS_LIST])
    ods_cols = mlb.classes_.tolist()

    # Multi-label Stratified Split
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    train_val_idx, test_idx = next(msss.split(df_filtered["text_base"].values, Y))

    val_size_adj = VAL_SIZE / (1 - TEST_SIZE)
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_size_adj, random_state=RANDOM_SEED
    )
    train_sub_idx, val_sub_idx = next(msss2.split(train_val_idx, Y[train_val_idx]))

    train_idx, val_idx = train_val_idx[train_sub_idx], train_val_idx[val_sub_idx]

    # Export DL splits
    export_data(
        df_filtered,
        Y,
        (train_idx, val_idx, test_idx),
        PROCESSED_DL_DIR,
        "text_dl",
        ods_cols,
    )

    # Export ML splits + Vectorization
    export_data(
        df_filtered,
        Y,
        (train_idx, val_idx, test_idx),
        PROCESSED_ML_DIR,
        "text_ml",
        ods_cols,
    )

    tfidf = TfidfVectorizer(max_features=5000)
    tfidf.fit(df_filtered.iloc[train_idx]["text_ml"])

    with open(os.path.join(PROCESSED_ML_DIR, "tfidf_model.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(PROCESSED_ML_DIR, "mlb.pkl"), "wb") as f:
        pickle.dump(mlb, f)

    print("Success: Data integrated, processed, and exported for both ML and DL.")


if __name__ == "__main__":
    main()
