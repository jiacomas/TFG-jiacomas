# --- Paths ---
DATA_DIR = "../data"
RAW_METADATA_PATH = f"{DATA_DIR}/raw/Anuncis_2022_2024.xlsx"
RAW_TEXT_CSV_DIR = f"{DATA_DIR}/raw/description"  # 2022, 2023, 2024

OUTPUT_DIR = f"{DATA_DIR}/processed"
PROCESSED_ML_DIR = f"{OUTPUT_DIR}/ml"
PROCESSED_DL_DIR = f"{OUTPUT_DIR}/dl"

OUTPUT_DIR_EDA = "../figures/eda"

# --- Configuració general ---
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# --- ODS ---
ODS_ALL = [f"ODS {i}" for i in range(1, 18)]

ODS_COLORS = {
    "ODS 1": "#E5243B",
    "ODS 2": "#DDA63A",
    "ODS 3": "#4C9F38",
    "ODS 4": "#C5192D",
    "ODS 5": "#FF3A21",
    "ODS 6": "#26BDE2",
    "ODS 7": "#FCC30B",
    "ODS 8": "#A21942",
    "ODS 9": "#FD6925",
    "ODS 10": "#DD1367",
    "ODS 11": "#FD9D24",
    "ODS 12": "#BF8B2E",
    "ODS 13": "#3F7E44",
    "ODS 14": "#0A97D9",
    "ODS 15": "#56C02B",
    "ODS 16": "#00689D",
    "ODS 17": "#19486A",
}
