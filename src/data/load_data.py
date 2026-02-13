import csv
import sys
import pandas as pd
from pathlib import Path
import ast

from src.utils.routes import CSV_2022, CSV_2023, CSV_2024, META_FILE, DATA_CLEAN
from src.data.schema import Description, Metadata
from src.data.clean_data import clean
from src.data.merge_sources import merge_data

csv.field_size_limit(
    sys.maxsize
)  # Increase CSV field size limit to handle large text fields


def validate_columns(df, expected_cols):
    """
    Validate that the DataFrame has all the expected columns.
    """
    if not set(expected_cols).issubset(df.columns):
        missing = set(expected_cols) - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")


def load_csv(filepath: Path, expected_cols: list[str]) -> pd.DataFrame:
    """
    Load a CSV file into a list of dictionaries.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)

    validate_columns(df, expected_cols)

    return df[expected_cols]


def load_excel(filepath: Path, expected_cols: list[str]) -> pd.DataFrame:
    """
    Load all sheets from an Excel file and merge them into a single DataFrame.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    sheets = pd.read_excel(filepath, sheet_name=None)

    dfs = []
    for sheet_name, df in sheets.items():
        validate_columns(df, expected_cols)
        dfs.append(df[expected_cols])

    return pd.concat(dfs, ignore_index=True)


def load_files(
    csv_loads: list[Path], meta_file: Path, csv_cols: list[str], meta_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all files and merge them into a single DataFrame.
    """
    load = []
    for csv_path in csv_loads:
        load.append(load_csv(csv_path, csv_cols))

    load = pd.concat(load, ignore_index=True)

    meta = load_excel(meta_file, meta_cols)

    return load, meta


def create_clean_data(
    csv_loads: list[Path] = None,
    meta_file: Path = None,
    csv_cols: list[str] = None,
    meta_cols: list[str] = None,
) -> pd.DataFrame:
    csv_loads = csv_loads or [CSV_2022, CSV_2023, CSV_2024]
    meta_file = meta_file or META_FILE

    csv_cols = csv_cols or [Description.ID_REGISTRE, Description.TEXT_RAW]
    meta_cols = meta_cols or [
        Metadata.ID,
        Metadata.DATE,
        Metadata.ID_REGISTRE,
        Metadata.ORGANIZATION,
        Metadata.TITLE,
        Metadata.TYPE,
        Metadata.ODS,
        Metadata.PDF_URL,
    ]

    desc_df, meta_df = load_files(csv_loads, meta_file, csv_cols, meta_cols)

    descriptions, metadata = clean(desc_df, meta_df)

    merged = merge_data(descriptions, metadata)

    return merged


def parse_ods(value):
    if isinstance(value, str):
        if "nan" in value:
            return []
        return ast.literal_eval(value)
    return []


def get_clean_data(
    filepath: Path = DATA_CLEAN, expected_columns: list[str] = None
) -> pd.DataFrame:
    expected_columns = expected_columns or [
        Description.ID_REGISTRE,
        Description.TEXT_RAW,
        Description.TEXT_CLEAN,
        Metadata.ID,
        Metadata.DATE,
        Metadata.ORGANIZATION,
        Metadata.TITLE,
        Metadata.TYPE,
        Metadata.ODS,
        Metadata.PDF_URL,
    ]

    data = load_csv(filepath, expected_columns)

    data[Metadata.ODS] = data[Metadata.ODS].apply(parse_ods)
    data[Metadata.DATE] = pd.to_datetime(data[Metadata.DATE], format="%Y-%m-%d")

    return data
