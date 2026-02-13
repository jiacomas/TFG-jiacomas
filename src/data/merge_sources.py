import pandas as pd
from src.data.schema import Metadata, Description


def multiple_ods(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Treat multiple ODS for the same announcement.
    """
    cols = [
        col for col in meta.columns if col not in [Metadata.ID_REGISTRE, Metadata.ODS]
    ]
    return (
        meta.groupby(Metadata.ID_REGISTRE)
        .agg({Metadata.ODS: list, **{col: "first" for col in cols}})
        .reset_index()
    )


def merge_data(description: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Merge clean data from CSV and Excel files.
    """
    metadata = multiple_ods(metadata)
    return pd.merge(
        metadata,
        description,
        left_on=Metadata.ID_REGISTRE,  # column from metadata
        right_on=Description.ID_REGISTRE,  # column from description
        how="outer",  # full outer join
    )
