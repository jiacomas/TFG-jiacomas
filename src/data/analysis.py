import pandas as pd
from src.data.schema import Metadata
from src.utils.routes import ODS_VISUALIZATION, ODS_PERCENTAGE_VISUALIZATION

def clean_nan(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna()
    return data

def count_ods(data: pd.DataFrame) -> dict[str, int]:
    ods_count = {f"ODS {i}": 0 for i in range(1, 18)}

    for _, row in data.iterrows():
        analyse_ods = row[Metadata.ODS]

        for ods_name in analyse_ods:
            key = ods_name[:6].strip()  # "ODS 1", "ODS 10", etc.
            if key in ods_count:
                ods_count[key] += 1

    return ods_count

if __name__ == "__main__":
    from src.data.load_data import get_clean_data

    data = get_clean_data()

    print("Anuncis carregats:", len(data))

    data = clean_nan(data)
    print("Anuncis sense NaN:", len(data))

    count = count_ods(data)

    print("\nODS:", count)
    from src.data.visualize import visualize_ods, save_visualization, visualize_ods_percentage
    save_visualization(visualize_ods(count), ODS_VISUALIZATION)
    save_visualization(visualize_ods_percentage(count), ODS_PERCENTAGE_VISUALIZATION)

    print("\nFINAL")