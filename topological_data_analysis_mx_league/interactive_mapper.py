from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm
from ripser import ripser
from scipy.spatial import distance_matrix

from topological_data_analysis_mx_league.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "normalized_features.csv",
    output_path: Path = FIGURES_DIR / "mapper/persistent_homology_graph.html",
):
    try:
        logger.info("Loading dataset...")
        df = pd.read_csv(input_path)

        # Seleccionar solo las columnas numéricas para el análisis topológico
        numeric_columns = df.select_dtypes(include=[float, int]).columns.tolist()
        df_numeric = df[numeric_columns]

        # Calcular la matriz de distancias
        logger.info("Calculating distance matrix...")
        dist_matrix = distance_matrix(df_numeric, df_numeric)

        # Realizar el cálculo de la homología persistente
        logger.info("Computing persistent homology...")
        result = ripser(dist_matrix, maxdim=2, distance_matrix=True)

        # Extraer los diagramas de persistencia
        diagrams = result['dgms']

        # Crear el grafo interactivo de persistencia
        logger.info("Creating persistence diagram graph...")

        # Crear un grafo dirigido (para visualizar las conexiones entre los puntos de nacimiento y muerte)
        G = nx.Graph()

        # Añadir los nodos y aristas al grafo
        for dim, diagram in enumerate(diagrams):
            for birth, death in diagram:
                # Usar coordenadas de nacimiento y muerte como nodos
                G.add_node(f"{birth:.2f}-{death:.2f}", pos=(birth, death), dimension=dim)

                # Añadir una arista de nacimiento a muerte
                G.add_edge(f"{birth:.2f}-{death:.2f}", f"{birth:.2f}-{death:.2f}", weight=death - birth)

        # Crear un layout para la visualización con Plotly
        pos = nx.get_node_attributes(G, 'pos')
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        edge_x = []
        edge_y = []

        # Construir las coordenadas de las aristas
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_y.append(y0)
            edge_y.append(y1)

        # Crear el gráfico de Plotly
        fig = go.Figure()

        # Añadir las aristas al gráfico
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='gray'), hoverinfo='none'))

        # Añadir los nodos con diferentes estilos según la dimensión
        for node in G.nodes():
            x, y = pos[node]
            dimension = G.nodes[node]['dimension']
            if dimension == 0:
                fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(size=8, color='blue', symbol='circle'),
                                         name=f"Dim 0 (birth={x:.2f}, death={y:.2f})"))
            elif dimension == 1:
                fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(size=8, color='green', symbol='square'),
                                         name=f"Dim 1 (birth={x:.2f}, death={y:.2f})"))
            elif dimension == 2:
                fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(size=8, color='red', symbol='diamond'),
                                         name=f"Dim 2 (birth={x:.2f}, death={y:.2f})"))

        # Actualizar el layout
        fig.update_layout(title="Persistence Homology Graph", title_x=0.5,
                          xaxis_title="Birth", yaxis_title="Death",
                          showlegend=True, hovermode="closest")

        # Guardar el gráfico interactivo como archivo HTML
        fig.write_html(output_path)
        logger.info(f"Interactive persistence homology graph saved to {output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    app()
