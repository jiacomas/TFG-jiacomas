import csv
from pathlib import Path

import pandas as pd

from src.data.schema import Description, Metadata


def clean_nan(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna()
    return data


def count_ods(data: pd.DataFrame) -> dict[str, int]:
    count_aparicions_ods = {f"ODS {i}": 0 for i in range(1, 18)}
    ods_anunci = []
    num_ods_anunci = {i: 0 for i in range(18)}

    for _, row in data.iterrows():
        analyse_ods = row[Metadata.ODS]
        temp_ods = []

        for ods_name in analyse_ods:
            key = ods_name[:6].strip()  # "ODS 1", "ODS 10", etc.
            if key in temp_ods:
                continue
            temp_ods.append(key)
            if key in count_aparicions_ods:
                count_aparicions_ods[key] += 1

        num_ods_anunci[len(temp_ods)] += 1
        ods_anunci.append(temp_ods)

    return count_aparicions_ods, ods_anunci, num_ods_anunci


def get_correlation(ods_anunci: list[list[str]]) -> pd.DataFrame:
    ods_labels = [f"ODS {i}" for i in range(1, 18)]
    concurrency = pd.DataFrame(0, index=ods_labels, columns=ods_labels)

    for ods_list in ods_anunci:
        for ods1 in ods_list:
            for ods2 in ods_list:
                concurrency.at[ods1, ods2] += 1
                concurrency.at[ods2, ods1] += 1
    return concurrency


def get_average(data: list[int]) -> float:
    return sum(data) / len(data)


def get_max(data: list[int]) -> int:
    return max(data)


def get_min(data: list[int]) -> int:
    return min(data)


def get_length_description(data: pd.DataFrame) -> list[int]:
    return [len(row[Description.TEXT_CLEAN]) for _, row in data.iterrows()]


def get_organization(data: pd.DataFrame) -> dict[str, int]:
    org_count = {}
    for _, row in data.iterrows():
        org = row[Metadata.ORGANIZATION].lower()
        if org in org_count:
            org_count[org] += 1
        else:
            org_count[org] = 1
    return dict(sorted(org_count.items()))


def save_dict(data: dict, filepath: Path):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ORG", "APARICIONS"])
        for key, value in data.items():
            writer.writerow([key, value])
