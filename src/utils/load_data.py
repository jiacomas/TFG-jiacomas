"""
Data loading utilities for TFG project.
Handles loading and combining CSV data from multiple years.
"""

import pandas as pd
from pathlib import Path
import sys

from src.config import CSV_2022, CSV_2023, CSV_2024

def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a single CSV file into a DataFrame.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    return pd.read_csv(filepath)

def main():
    dataframe = load_csv(CSV_2022)
    
    if dataframe.empty:
        print("Error: No data files could be loaded", file=sys.stderr)
        sys.exit(1)
    
    print(dataframe.head())
    
    return dataframe


if __name__ == "__main__":
    df = main()