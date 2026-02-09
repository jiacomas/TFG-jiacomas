import pandas as pd
from pathlib import Path
from src.utils.routes import DATA_CLEAN, ODS_VISUALIZATION, ODS_PERCENTAGE_VISUALIZATION

def load_raw_data(save = True, filepath: Path = DATA_CLEAN):
    from src.data.load_data import create_clean_data
    from src.data.save_data import save_data

    merged = create_clean_data()
    if save:
        save_data(merged, filepath)
    return merged

def load_clean_data(filepath: Path = DATA_CLEAN):
    from src.data.load_data import get_clean_data
    return get_clean_data(filepath)

def visualize_ods(data: pd.DataFrame = None):
    from src.data.analysis import count_ods
    from src.data.visualize import visualize_ods, save_visualization, visualize_ods_percentage

    data = data or load_clean_data()
    ods_count = count_ods(data)
    ods = visualize_ods(ods_count)
    percentage = visualize_ods_percentage(ods_count)
    save_visualization(ods, ODS_VISUALIZATION)
    save_visualization(percentage, ODS_PERCENTAGE_VISUALIZATION)

if __name__ == "__main__":
    visualize_ods()
    