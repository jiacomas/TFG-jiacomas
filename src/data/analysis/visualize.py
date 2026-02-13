import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

ODS_COLORS = {
    "ODS 1": "#E5243B",
    "ODS 2": "#DDA63A",
    "ODS 3": "#4C9F38",
    "ODS 4": "#C5192D",
    "ODS 5": "#FF3A21",
    "ODS 6": "#26BDE2",
    "ODS 7": "#FCC30B",
    "ODS 8": "#A21942",
    "ODS 9": "#FD6925",
    "ODS 10": "#DD1367",
    "ODS 11": "#FD9D24",
    "ODS 12": "#BF8B2E",
    "ODS 13": "#3F7E44",
    "ODS 14": "#0A97D9",
    "ODS 15": "#56C02B",
    "ODS 16": "#00689D",
    "ODS 17": "#19486A",
}


def bar_chart_ods(ods_count: dict[str, int]):
    fig, ax = plt.subplots()

    ods = list(ods_count.keys())
    values = list(ods_count.values())
    total = sum(values)

    # Bars color
    colors = [ODS_COLORS.get(o, "#CCCCCC") for o in ods]
    bars = ax.bar(ods, values, color=colors)

    # Labels
    ax.set_xlabel("ODS")
    ax.set_ylabel("Anuncis")
    ax.set_title("Nombre d'anuncis per ODS")
    ax.tick_params(axis="x", rotation=45)

    # Grid
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6, color="gray")
    ax.set_axisbelow(True)

    # Values on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Legend with total
    ax.legend([f"Total d'ODS: {total}"])

    fig.tight_layout()
    return fig


def bar_chart_ods_percentage(ods_count: dict[str, int]):
    total = sum(ods_count.values())
    ods_percentage = {k: v / total * 100 for k, v in ods_count.items()}

    fig, ax = plt.subplots()

    # Bars color
    colors = [ODS_COLORS.get(o, "#CCCCCC") for o in ods_percentage.keys()]
    bars = ax.bar(ods_percentage.keys(), ods_percentage.values(), color=colors)

    # Labels
    ax.set_xlabel("ODS")
    ax.set_ylabel("Percentatge d'anuncis (%)")
    ax.set_title("Percentatge d'anuncis per ODS")
    ax.tick_params(axis="x", rotation=45)

    # Grid
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6, color="gray")
    ax.set_axisbelow(True)

    # Values on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def bar_chart_distribution_ods_register(ods_anunci_count: list[int]):
    fig, ax = plt.subplots()

    # Bars color
    bars = ax.bar(ods_anunci_count.keys(), ods_anunci_count.values())

    # Fixar exactament les etiquetes de l'eix X
    ax.set_xticks(list(ods_anunci_count.keys()))
    ax.set_xticklabels(list(ods_anunci_count.keys()))

    ax.set_xlabel("ODS")
    ax.set_ylabel("Anuncis")
    ax.set_title("Nombre d'ODS per anunci")
    ax.tick_params(axis="x")

    # Grid
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6, color="gray")
    ax.set_axisbelow(True)

    # Values on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Legend with total
    ax.legend([f"Total anuncis: {sum(ods_anunci_count.values())}"])

    fig.tight_layout()
    return fig


def heatmap_correlation_ods(ods_correlation: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.imshow(ods_correlation, cmap="coolwarm")
    ax.set_xticks(range(len(ods_correlation.columns)))
    ax.set_xticklabels(ods_correlation.columns, rotation=90)
    ax.set_yticks(range(len(ods_correlation.index)))
    ax.set_yticklabels(ods_correlation.index)
    ax.set_title("Correlació d'ODS")
    fig.tight_layout()
    return fig


def histogram_length_description(length_description: list[int]):
    fig, ax = plt.subplots()
    ax.hist(length_description, bins=20)
    ax.set_xlabel("Longitud de la descripció")
    ax.set_ylabel("Freqüència")
    ax.set_title("Freqüència de la longitud de la descripció")
    fig.tight_layout()
    return fig


def box_plot_length_description(length_description: list[int]):
    fig, ax = plt.subplots()
    ax.boxplot(length_description)
    ax.set_xlabel("Longitud de la descripció")
    ax.set_ylabel("Longitud de la descripció")
    ax.set_title("Box plot de la longitud de la descripció")
    fig.tight_layout()
    return fig


def bar_chart_organization(organization: dict[str, int]):
    fig, ax = plt.subplots()
    ax.bar(organization.keys(), organization.values())
    ax.set_xlabel("Organització")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Freqüència")
    ax.set_title("Freqüència d'organització")
    fig.tight_layout()
    return fig


def save_visualization(fig, filepath: Path):
    fig.savefig(filepath)
    plt.close(fig)
