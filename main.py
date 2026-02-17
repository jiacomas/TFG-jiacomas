from pathlib import Path

import pandas as pd

from src.data.analysis.analysis import (
    clean_nan,
    count_ods,
    get_average,
    get_correlation,
    get_length_description,
    get_max,
    get_min,
    get_organization,
    save_dict,
)
from src.data.analysis.visualize import (
    bar_chart_distribution_ods_register,
    bar_chart_ods,
    bar_chart_ods_percentage,
    bar_chart_organization,
    box_plot_length_description,
    heatmap_correlation_ods,
    histogram_length_description,
    save_visualization,
)
from src.data.load_data import create_clean_data, get_clean_data
from src.data.save_data import save_data
from src.utils.routes import (
    DATA_CLEAN,
    LENGTH_DESCRIPTION_BOXPLOT_VISUALIZATION,
    LENGTH_DESCRIPTION_HISTOGRAM_VISUALIZATION,
    ODS_CORRELATION_VISUALIZATION,
    ODS_DISTRIBUTION_REGISTER_VISUALIZATION,
    ODS_PERCENTAGE_VISUALIZATION,
    ODS_VISUALIZATION,
    ORGANIZATION_CSV,
    ORGANIZATION_VISUALIZATION,
)

Y_value = "Y"


def load_raw_data(save=True, filepath: Path = DATA_CLEAN):
    merged = create_clean_data()
    if save:
        save_data(merged, filepath)
    return merged


def visualize_data(data: pd.DataFrame = None):
    data = data or get_clean_data()

    print("Anuncis carregats:", len(data))

    data = clean_nan(data)
    print("Anuncis sense NaN:", len(data))

    count, ods_anunci, num_ods_anunci = count_ods(data)

    repeat_ods = [len(ods) for ods in ods_anunci]

    print("\nMitjana ODS per anunci:", get_average(repeat_ods))
    print("Màxim ODS per anunci:", get_max(repeat_ods))
    print("Distribució dels ODS:", count)

    concurrency = get_correlation(ods_anunci)
    print("Concurrencia dels ODS:", concurrency)

    response = input("visualitzar ODS? Y/N: ").strip().upper()
    if response == Y_value:
        save_visualization(bar_chart_ods(count), ODS_VISUALIZATION)
        save_visualization(
            bar_chart_ods_percentage(count), ODS_PERCENTAGE_VISUALIZATION
        )
        save_visualization(
            bar_chart_distribution_ods_register(num_ods_anunci),
            ODS_DISTRIBUTION_REGISTER_VISUALIZATION,
        )
        save_visualization(
            heatmap_correlation_ods(concurrency), ODS_CORRELATION_VISUALIZATION
        )

    length_description = get_length_description(data)
    print("\nMitjana longitud descripció:", get_average(length_description))
    print("Màxim longitud descripció:", get_max(length_description))
    print("Mínim longitud descripció:", get_min(length_description))

    response = input("visualitzar Longitud descripció? Y/N: ").strip().upper()
    if response == Y_value:
        save_visualization(
            histogram_length_description(length_description),
            LENGTH_DESCRIPTION_HISTOGRAM_VISUALIZATION,
        )
        save_visualization(
            box_plot_length_description(length_description),
            LENGTH_DESCRIPTION_BOXPLOT_VISUALIZATION,
        )

    organization = get_organization(data)
    response = input("visualitzar Organització? Y/N: ").strip().upper()
    if response == Y_value:
        save_visualization(
            bar_chart_organization(organization), ORGANIZATION_VISUALIZATION
        )
    save_dict(organization, ORGANIZATION_CSV)

    print("\nFINAL")


if __name__ == "__main__":
    visualize_data()
