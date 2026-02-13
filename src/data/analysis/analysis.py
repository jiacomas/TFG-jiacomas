import pandas as pd
from src.data.schema import Metadata, Description

def clean_nan(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna()
    return data

def count_ods(data: pd.DataFrame) -> dict[str, int]:
    ods_count = {f"ODS {i}": 0 for i in range(1, 18)}
    ods_anunci = []
    ods_concurrency = []

    for _, row in data.iterrows():
        analyse_ods = row[Metadata.ODS]
        temp_ods = []

        for ods_name in analyse_ods:
            key = ods_name[:6].strip()  # "ODS 1", "ODS 10", etc.
            if key in temp_ods:
                continue
            temp_ods.append(key)
            if key in ods_count:
                ods_count[key] += 1
        
        # if len(analyse_ods) != len(temp_ods):
        #     print(row[Metadata.ID], row[Metadata.PDF_URL])
        ods_anunci.append(temp_ods)

    return ods_count, ods_anunci

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

from pathlib import Path
import csv

def save_dict(data: dict, filepath: Path):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ORG", "APARICIONS"])
        for key, value in data.items():
            writer.writerow([key, value])


def main():
    from src.data.load_data import get_clean_data
    from src.data.analysis.visualize import save_visualization
    from src.utils.routes import (ODS_VISUALIZATION, ODS_PERCENTAGE_VISUALIZATION, 
                                LENGTH_DESCRIPTION_HISTOGRAM_VISUALIZATION, LENGTH_DESCRIPTION_BOXPLOT_VISUALIZATION,
                                ORGANIZATION_VISUALIZATION, HEATMAP_CONCURRENCY_ODS, ORGANIZATION_CSV, 
                                ODS_DISTRIBUTION_REGISTER_VISUALIZATION, ODS_CORRELATION_VISUALIZATION)

    data = get_clean_data()

    print("Anuncis carregats:", len(data))

    data = clean_nan(data)
    print("Anuncis sense NaN:", len(data))

    count, ods_anunci = count_ods(data)

    repeat_ods = [len(ods) for ods in ods_anunci]

    print("\nMitjana ODS per anunci:", get_average(repeat_ods))
    print("Màxim ODS per anunci:", get_max(repeat_ods))
    print("Distribució dels ODS:", count)

    concurrency = get_correlation(ods_anunci)
    print("Concurrencia dels ODS:", concurrency)

    if input("visualitzar? Y/N: ").upper() == "Y":
        from src.data.analysis.visualize import bar_chart_ods, bar_chart_ods_percentage, bar_chart_distribution_ods_register, heatmap_correlation_ods
        save_visualization(bar_chart_ods(count), ODS_VISUALIZATION)
        save_visualization(bar_chart_ods_percentage(count), ODS_PERCENTAGE_VISUALIZATION)
        save_visualization(bar_chart_distribution_ods_register(ods_anunci), ODS_DISTRIBUTION_REGISTER_VISUALIZATION)
        save_visualization(heatmap_correlation_ods(concurrency), ODS_CORRELATION_VISUALIZATION)

    length_description = get_length_description(data)
    print("\nMitjana longitud descripció:", get_average(length_description))
    print("Màxim longitud descripció:", get_max(length_description))
    print("Mínim longitud descripció:", get_min(length_description))
    if input("visualitzar? Y/N: ").upper() == "Y":
        from src.data.analysis.visualize import box_plot_length_description, histogram_length_description
        save_visualization(histogram_length_description(length_description), LENGTH_DESCRIPTION_HISTOGRAM_VISUALIZATION)
        save_visualization(box_plot_length_description(length_description), LENGTH_DESCRIPTION_BOXPLOT_VISUALIZATION)

    organization = get_organization(data)
    # print("\nOrganització:", organization)
    if input("visualitzar? Y/N: ").upper() == "Y":
        from src.data.analysis.visualize import bar_chart_organization
        save_visualization(bar_chart_organization(organization), ORGANIZATION_VISUALIZATION)
    save_dict(organization, ORGANIZATION_CSV)

    print("\nFINAL")