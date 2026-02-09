from src.data.load_data import get_clean_data
import pandas as pd
from src.data.schema import Metadata

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

def main():
    data = get_clean_data()

    print("Anuncis carregats:", len(data))

    data = clean_nan(data)
    print("Anuncis sense NaN:", len(data))

    print("\nODS:", count_ods(data))

    print("\nFINAL")


if __name__ == "__main__":
    main()