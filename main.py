import pandas as pd
from pathlib import Path
from src.utils.routes import DATA_CLEAN

def load_raw_data():
    from src.data.load_data import create_clean_data

    merged = create_clean_data()
    return merged

def save_data(data: pd.DataFrame = None, filepath: Path = DATA_CLEAN):
    if data is None:
        data = load_raw_data()

    from src.data.save_data import save_data
    save_data(data, filepath)

def visualize_ods(data: pd.DataFrame = None):
    from src.data.load_data import get_clean_data
    from src.data.analysis import count_ods
    from src.data.visualize import visualize_ods

    data = data or get_clean_data()
    ods_count = count_ods(data)
    visualize_ods(ods_count)

if __name__ == "__main__":
    visualize_ods()
    