import csv
import sys
import pandas as pd

from src.utils.routes import CSV_2022, CSV_2023, CSV_2024, META_FILE
from src.data.schema import Description, Metadata

# Increase CSV field size limit to handle large text fields
csv.field_size_limit(sys.maxsize)

"""
Load data from CSV and Excel files. Clean them.
"""

def load_csv(filepath: str) -> list[dict]:
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

def load_excel(filepath: str) -> list[dict]:
    """
    Load an Excel file into a list of dictionaries.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_excel(filepath)

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

def main():
    csv_22 = load_csv(CSV_2022)
    csv_23 = load_csv(CSV_2023)
    csv_24 = load_csv(CSV_2024)

    meta = load_excel(META_FILE)

    return csv_22, csv_23, csv_24, meta

if __name__ == "__main__":
    main()