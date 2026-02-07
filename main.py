from src.data.load_data import load_clean_data


if __name__ == "__main__":
    merged = load_clean_data()

    # Save data
    # from src.data.save_data import save_data
    # from src.utils.routes import DATA_CLEAN
    # save_data(merged, DATA_CLEAN)

    print(merged.head())
    