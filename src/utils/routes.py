from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/config.py -> src -> tfg

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

CSV_2022 = RAW_DIR / "anuncis_BOPB_2022_contingut.csv"
CSV_2023 = RAW_DIR / "anuncis_BOPB_2023_contingut.csv"
CSV_2024 = RAW_DIR / "anuncis_BOPB_2024_contingut.csv"
META_FILE = RAW_DIR / "Anuncis_2022_2024.xlsx"

DATA_CLEAN = PROCESSED_DIR / "data_clean.csv"