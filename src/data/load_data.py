import csv
import sys
import pandas as pd
from pathlib import Path

from src.utils.routes import CSV_2022, CSV_2023, CSV_2024, META_FILE, DATA_CLEAN
from src.data.schema import Description, Metadata
from src.data.clean_data import clean
from src.data.merge_sources import merge_data

# Increase CSV field size limit to handle large text fields
csv.field_size_limit(sys.maxsize)

def validate_columns(df, expected_cols):
    """
    Validate that the DataFrame has all the expected columns.
    """
    if not expected_cols.issubset(df.columns):
        missing = expected_cols - set(df.columns)
        raise ValueError(
            f"Missing columns: {missing}"
        )

def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a CSV file into a list of dictionaries.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)

    excepted_col = [
        Description.ID_REGISTRE,
        Description.TEXT_RAW
    ]
    validate_columns(df, excepted_col)

    return df[excepted_col]

def load_excel(filepath: Path) -> pd.DataFrame:
    """
    Load all sheets from an Excel file and merge them into a single DataFrame.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    sheets = pd.read_excel(filepath, sheet_name=None)

    dfs = []
    expected_cols = [
        Metadata.ID,
        Metadata.DATE,
        Metadata.ID_REGISTRE,
        Metadata.ORGANIZATION,
        Metadata.TITLE,
        Metadata.TYPE,
        Metadata.ODS,
        Metadata.PDF_URL,
    ]

    for sheet_name, df in sheets.items():
        validate_columns(df, expected_cols)
        dfs.append(df[expected_cols])

    return pd.concat(dfs, ignore_index=True)


def load_files(csv_loads: list[Path], meta_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all files and merge them into a single DataFrame.
    """
    load = []
    for csv_path in csv_loads:
        load.append(load_csv(csv_path))
    
    load = pd.concat(load, ignore_index=True)
    
    meta = load_excel(meta_file)
    
    return load, meta

def load_clean_data(csv_loads: list[Path] = None, meta_file: Path = None) -> pd.DataFrame:
    csv_loads = csv_loads or [CSV_2022, CSV_2023, CSV_2024]
    meta_file = meta_file or META_FILE

    desc_df, meta_df = load_files(csv_loads, meta_file)
    
    descriptions, metadata = clean(desc_df, meta_df)

    merged = merge_data(descriptions, metadata)

    return merged