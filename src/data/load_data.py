import csv
import sys
import pandas as pd
from pathlib import Path

from src.utils.routes import CSV_2022, CSV_2023, CSV_2024, META_FILE, DATA_CLEAN
from src.data.schema import Description, Metadata
from src.data.clean_data import clean_text_from_data
from src.data.merge_sources import merge_data

# Increase CSV field size limit to handle large text fields
csv.field_size_limit(sys.maxsize)

"""
Load data from CSV and Excel files. Clean them.
"""

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
    if not set(excepted_col).issubset(df.columns):
        raise ValueError(f"Missing columns in {filepath}")

    return df[excepted_col]

def load_excel(filepath: Path) -> pd.DataFrame:
    """
    Load an Excel file into a list of dictionaries.
    It has different pages, so we need to load them all.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_excel(filepath, sheet_name=None)

    excepted_col = [
        Metadata.ID,
        Metadata.DATE,
        Metadata.ID_REGISTRE,
        Metadata.ORGANIZATION,
        Metadata.TITLE,
        Metadata.TYPE,
        Metadata.ODS,
        Metadata.PDF_URL
    ]
    if not set(excepted_col).issubset(df.columns):
        raise ValueError(f"Missing columns in {filepath}")

    return df[excepted_col]

def load_excel(filepath: Path) -> pd.DataFrame:
    """
    Load all sheets from an Excel file and merge them into a single DataFrame.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    sheets = pd.read_excel(filepath, sheet_name=None)

    dfs = []
    expected_cols = {
        Metadata.ID,
        Metadata.DATE,
        Metadata.ID_REGISTRE,
        Metadata.ORGANIZATION,
        Metadata.TITLE,
        Metadata.TYPE,
        Metadata.ODS,
        Metadata.PDF_URL,
    }

    for sheet_name, df in sheets.items():
        if not expected_cols.issubset(df.columns):
            missing = expected_cols - set(df.columns)
            raise ValueError(
                f"Missing columns in sheet '{sheet_name}': {missing}"
            )

        dfs.append(df[list(expected_cols)])

    return pd.concat(dfs, ignore_index=True)

def save_data(data: pd.DataFrame, filepath: Path):
    """
    Save data to a CSV file.
    """
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(filepath, index=False)

def load_clean_data():
    # Load data
    csv_loads = [CSV_2022, CSV_2023, CSV_2024]
    csv_22 = load_csv(CSV_2022)
    csv_23 = load_csv(CSV_2023)
    csv_24 = load_csv(CSV_2024)

    csv_22_24 = pd.concat([csv_22, csv_23, csv_24], ignore_index=True)

    meta = load_excel(META_FILE)
    
    # Clean data
    csv_22_24 = clean_text_from_data(
        csv_22_24,
        Description.TEXT_RAW,
        Description.TEXT_CLEAN
    )

    meta = clean_text_from_data(
        meta,
        Metadata.TITLE,
        Metadata.TITLE_CLEAN
    )

    # Merge data
    merged = merge_data(csv_22_24, meta)

    save_data(merged, DATA_CLEAN)

    return merged
