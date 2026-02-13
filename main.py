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
    from src.data.analysis.analysis import count_ods, clean_nan
    from src.data.analysis.visualize import bar_chart_ods, save_visualization, bar_chart_ods_percentage

    data = data or load_clean_data()
    data = clean_nan(data)
    ods_count = count_ods(data)
    ods = bar_chart_ods(ods_count)
    percentage = bar_chart_ods_percentage(ods_count)
    save_visualization(ods, ODS_VISUALIZATION)
    save_visualization(percentage, ODS_PERCENTAGE_VISUALIZATION)

if __name__ == "__main__":
    from src.data.analysis.analysis import main
    main()

    # visualize_ods()
    