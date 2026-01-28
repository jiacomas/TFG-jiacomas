# Load data
from src.utils.load_data import main as load_data


if __name__ == "__main__":
    df = load_data()
    print(df.head())