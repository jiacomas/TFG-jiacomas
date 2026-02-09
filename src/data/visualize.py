import matplotlib.pyplot as plt

def visualize_ods(ods_count: dict[str, int]):
    plt.bar(ods_count.keys(), ods_count.values())
    plt.xlabel("ODS")
    plt.ylabel("Nombre d'anuncis")
    plt.title("Nombre d'anuncis per ODS")
    plt.xticks(rotation=45)
    plt.show()