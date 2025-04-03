# Importar librerías necesarias
import os
import pandas as pd
import matplotlib.pyplot as plt
import squarify
import networkx as nx
import numpy as np
import matplotlib.cm as cm


def generar_treemap():
    # Cargar los archivos CSV
    treemap_df = pd.read_csv("data/API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_13431.csv", skiprows=4)
    metadata_country_df = pd.read_csv("data/Metadata_Country_API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_13431.csv")

    # Filtrar países válidos
    valid_countries = metadata_country_df[metadata_country_df["Region"].notna()]["Country Code"].tolist()
    year = "2022"
    treemap_filtered = treemap_df[treemap_df["Country Code"].isin(valid_countries)][["Country Name", "Country Code", year]].dropna()
    treemap_filtered[year] = pd.to_numeric(treemap_filtered[year], errors='coerce')
    treemap_filtered = treemap_filtered.nlargest(20, year)

    # Preparar datos para el treemap
    labels = [f"{name}\n{value:.2f}%" for name, value in zip(treemap_filtered["Country Name"], treemap_filtered[year])]
    values = treemap_filtered[year].tolist()
    colors = plt.cm.viridis(plt.Normalize(min(values), max(values))(values))

    # Generar el Treemap
    fig, ax = plt.subplots(figsize=(12, 8))
    squarify.plot(sizes=values, label=labels, alpha=0.8, color=colors, text_kwargs={'fontsize': 10})
    plt.title(f"Treemap del PIB en Agricultura por País (Año {year})")
    plt.axis("off")
    plt.savefig("img/treemap_plot.png")
    plt.close()


def generar_arc_diagram():
    # Cargar los archivos CSV
    characters_df = pd.read_csv("data/starwars-characters.csv")
    links_df = pd.read_csv("data/starwars-links.csv")

    # Filtrar personajes principales
    top_characters = characters_df.nlargest(20, 'scenes')
    top_ids = set(top_characters['number'])
    top_links = links_df[(links_df['scenes'] >= 20) & (links_df['character1'].isin(top_ids)) & (links_df['character2'].isin(top_ids))]

    # Crear el grafo
    G = nx.Graph()
    for _, row in top_characters.iterrows():
        G.add_node(row['number'], label=row['name'], scenes=row['scenes'])
    for _, row in top_links.iterrows():
        G.add_edge(row['character1'], row['character2'], weight=row['scenes'])

    # Ordenar nodos por el número de escenas
    sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[1]['scenes'], reverse=True)
    labels = {node: G.nodes[node]['label'] for node, _ in sorted_nodes}
    colors = cm.get_cmap('tab20', len(sorted_nodes))

    plt.figure(figsize=(16, 8))
    positions = {node: (i * 2, 0) for i, (node, _) in enumerate(sorted_nodes)}

    for i, (node, _) in enumerate(sorted_nodes):
        nx.draw_networkx_nodes(G, positions, nodelist=[node], node_size=300, node_color=[colors(i)], alpha=0.8)

    nx.draw_networkx_labels(G, positions, labels, font_size=10, verticalalignment='top')

    for (u, v, data) in G.edges(data=True):
        x = np.linspace(positions[u][0], positions[v][0], 100)
        y = np.sin(np.pi * (x - positions[u][0]) / (positions[v][0] - positions[u][0]))
        color = colors(list(G.nodes).index(u))
        plt.plot(x, y, linewidth=data['weight'] / 3, color=color, alpha=0.7)

    plt.title('Arc Diagram - Star Wars Character Interactions (Colores Diferenciados)')
    plt.axis('off')
    plt.savefig("img/arcdiagram_plot.png")
    plt.close()


def generar_sparklines():
    # Cargar datos
    gdp_df = pd.read_csv('data/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_26433.csv', skiprows=4)
    years = [str(year) for year in range(2010, 2024)]
    gdp_filtered = gdp_df.dropna(subset=years)
    gdp_filtered.loc[:, '2023'] = pd.to_numeric(gdp_filtered['2023'], errors='coerce')
    top_countries = gdp_filtered.nlargest(10, '2023')

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, row in top_countries.iterrows():
        sparkline = row[years].values
        country_name = row['Country Name']
        plt.subplot(5, 2, i + 1)
        plt.plot(years, sparkline, marker='o', color='black', linewidth=1)
        plt.fill_between(years, sparkline, alpha=0.3)
        plt.title(country_name, fontsize=8)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.box(False)
    plt.tight_layout()
    plt.savefig("img/Sparklines_plot.png")
    plt.close()


generar_treemap()
generar_arc_diagram()
generar_sparklines()
print("Gráficos generados y guardados en la carpeta 'img/'.")
