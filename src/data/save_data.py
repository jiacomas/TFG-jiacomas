from pathlib import Path
import pandas as pd

def save_data(data: pd.DataFrame, filepath: Path):
    """
    Save data to a CSV file.
    """
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(filepath, index=False)