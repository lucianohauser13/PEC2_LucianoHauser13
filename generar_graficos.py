# Importar librerías necesarias
import os
import pandas as pd
import matplotlib.pyplot as plt
import squarify
import networkx as nx
import numpy as np
import matplotlib.cm as cm


# Verificar y crear la carpeta img si no existe
def crear_carpeta_img():
    if not os.path.exists("img"):
        os.makedirs("img")


def generar_treemap():
    # Cargar los archivos CSV
    treemap_df = pd.read_csv("API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_13431.csv", skiprows=4)
    metadata_country_df = pd.read_csv("Metadata_Country_API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_13431.csv")

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
    # Cargar datos
    characters_df = pd.read_csv('starwars-characters.csv')
    links_df = pd.read_csv('starwars-links.csv')

    # Filtrar personajes principales
    top_characters = characters_df.nlargest(20, 'scenes')
    top_ids = set(top_characters['number'])
    top_links = links_df[(links_df['scenes'] >= 20) & (links_df['character1'].isin(top_ids)) & (links_df['character2'].isin(top_ids))]

    # Crear grafo
    G = nx.Graph()
    for _, row in top_characters.iterrows():
        G.add_node(row['number'], label=row['name'], scenes=row['scenes'])
    for _, row in top_links.iterrows():
        G.add_edge(row['character1'], row['character2'], weight=row['scenes'])

    # Dibujar Arc Diagram
    plt.figure(figsize=(16, 8))
    pos = {node: (i * 2, 0) for i, (node, _) in enumerate(sorted(G.nodes(data=True), key=lambda x: x[1]['scenes'], reverse=True))}
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=12)
    plt.title('Arc Diagram - Star Wars Character Interactions')
    plt.savefig("img/arcdiagram_plot.png")
    plt.close()


def generar_sparklines():
    # Cargar datos
    gdp_df = pd.read_csv('API_NY.GDP.PCAP.CD_DS2_en_csv_v2_26433.csv', skiprows=4)
    years = [str(year) for year in range(2010, 2024)]
    gdp_filtered = gdp_df.dropna(subset=years)
    gdp_filtered['2023'] = pd.to_numeric(gdp_filtered['2023'], errors='coerce')
    top_countries = gdp_filtered.nlargest(10, '2023')

    # Crear gráfico de Sparklines
    fig, ax = plt.subplots(figsize=(10, 8))
    for _, row in top_countries.iterrows():
        ax.plot(years, row[years].values, marker='o', label=row['Country Name'])
    ax.set_title('Sparklines: PIB per cápita (2010-2023) - Top 10 Países')
    plt.xticks(rotation=45)
    plt.xlabel('Año')
    plt.ylabel('PIB per cápita (USD)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("img/Sparklines_plot.png")
    plt.close()


crear_carpeta_img()
generar_treemap()
generar_arc_diagram()
generar_sparklines()
print("Gráficos generados y guardados en la carpeta 'img/'.")
