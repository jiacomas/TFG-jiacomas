import re
import pandas as pd

def clean_text(text: str) -> str:
    """
    Clean text data.
    """
    if not isinstance(text, str):
        return ""

    # lower case
    text = text.lower()

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)    
           
    # remove special characters
    text = re.sub(r"[^\w\sàèíòóúç]", "", text) 
    return text.strip()

def clean_text_from_data(data: pd.DataFrame, col: str, new_col: str) -> pd.DataFrame:
    data[new_col] = data[col].apply(clean_text)
    return data

    