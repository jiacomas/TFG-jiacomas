from pathlib import Path

# --- Dynamic Path Resolution ---
# This finds the absolute path to the TFG-jiacomas folder,
# ensuring scripts work regardless of where they are launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Paths ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_METADATA_PATH = DATA_DIR / "raw" / "Anuncis_2022_2024.xlsx"
RAW_TEXT_CSV_DIR = DATA_DIR / "raw" / "description"

OUTPUT_DIR = DATA_DIR / "processed"
PROCESSED_ML_DIR = OUTPUT_DIR / "ml"
PROCESSED_DL_DIR = OUTPUT_DIR / "dl"

OUTPUT_DIR_EDA = PROJECT_ROOT / "figures" / "eda"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist so the script doesn't fail when saving
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_EDA.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Configuració general ---
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# --- ODS ---
ODS_ALL = [f"ODS {i}" for i in range(1, 18)]

ODS_COLORS = {
    f"ODS {i}": color
    for i, color in enumerate(
        [
            "#E5243B",
            "#DDA63A",
            "#4C9F38",
            "#C5192D",
            "#FF3A21",
            "#26BDE2",
            "#FCC30B",
            "#A21942",
            "#FD6925",
            "#DD1367",
            "#FD9D24",
            "#BF8B2E",
            "#3F7E44",
            "#0A97D9",
            "#56C02B",
            "#00689D",
            "#19486A",
        ],
        1,
    )
}
