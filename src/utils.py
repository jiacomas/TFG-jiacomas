# --- Paths ---
RAW_METADATA_PATH = "../data/raw/Anuncis_2022_2024.xlsx"
RAW_TEXT_CSV_DIR = "../data/raw/description/"  # 2022.csv, 2023.csv, 2024.csv
OUTPUT_DIR = "../data/processed/"

# --- Configuració general ---
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# --- ODS ---
ODS_ALL = [f"ODS {i}" for i in range(1, 18)]
