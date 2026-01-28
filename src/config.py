from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # src/config.py -> src -> tfg

DATA_DIR = ROOT / "data"
PDFS = DATA_DIR / "pdfs"

CSV_2022 = DATA_DIR / "anuncis_BOPB_2022_contingut.csv"
CSV_2023 = DATA_DIR / "anuncis_BOPB_2023_contingut.csv"
CSV_2024 = DATA_DIR / "anuncis_BOPB_2024_contingut.csv"