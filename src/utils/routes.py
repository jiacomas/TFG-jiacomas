from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/config.py -> src -> tfg

# Data
DATA_DIR = ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

CSV_2022 = RAW_DIR / "anuncis_BOPB_2022_contingut.csv"
CSV_2023 = RAW_DIR / "anuncis_BOPB_2023_contingut.csv"
CSV_2024 = RAW_DIR / "anuncis_BOPB_2024_contingut.csv"
META_FILE = RAW_DIR / "Anuncis_2022_2024.xlsx"

DATA_CLEAN = PROCESSED_DIR / "data_clean.csv"

# Results
RESULTS_DIR = ROOT / "results"

## EDA
EDA_DIR = RESULTS_DIR / "eda"

ODS_VISUALIZATION = EDA_DIR / "ods_visualization.png"
ODS_PERCENTAGE_VISUALIZATION = EDA_DIR / "ods_percentage_visualization.png"
ODS_DISTRIBUTION_REGISTER_VISUALIZATION = EDA_DIR / "ods_distribution_register_visualization.png"
ODS_CORRELATION_VISUALIZATION = EDA_DIR / "ods_correlation_visualization.png"
LENGTH_DESCRIPTION_HISTOGRAM_VISUALIZATION = EDA_DIR / "length_description_histogram_visualization.png"
LENGTH_DESCRIPTION_BOXPLOT_VISUALIZATION = EDA_DIR / "length_description_boxplot_visualization.png"
ORGANIZATION_VISUALIZATION = EDA_DIR / "organization_visualization.png"
HEATMAP_CONCURRENCY_ODS = EDA_DIR / "heatmap_concurrency_ods.png"
ORGANIZATION_CSV = EDA_DIR / "organization.csv"
