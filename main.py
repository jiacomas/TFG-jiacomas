from src.data.load_data import load_clean_data

if __name__ == "__main__":
    merged = load_clean_data()
    print(merged.head())
    