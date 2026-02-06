# Load data
from src.data.load_data import main as load_data


if __name__ == "__main__":
    csv_22, csv_23, csv_24, meta = load_data()
    print(csv_22.head())
    print(csv_23.head())
    print(csv_24.head())
    print(meta.head())