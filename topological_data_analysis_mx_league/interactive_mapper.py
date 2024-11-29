from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import typer
from loguru import logger
from ripser import ripser
from scipy.spatial import distance_matrix
import kmapper as km
from sklearn.cluster import DBSCAN
from topological_data_analysis_mx_league.config import FIGURES_DIR, INTERIM_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "pca_components.csv",  # Cambiar por el archivo PCA
    output_path: Path = FIGURES_DIR / "mapper/persistent_homology_graph_two.html",
):
    try:
        logger.info("Loading dataset...")
        df = pd.read_csv(input_path)

        # Seleccionar solo las columnas numéricas (si ya son las componentes PCA, pueden ser las únicas)
        numeric_columns = df.select_dtypes(include=[float, int]).columns.tolist()
        df_numeric = df[numeric_columns]

        # Crear el Mapper
        mapper = km.KeplerMapper(verbose=1)

        # Crear la proyección usando las componentes PCA ya existentes
        lens = df_numeric.values  # Si ya tienes las componentes PCA, usa directamente los valores

        # Crear grafo con Mapper
        logger.info("Mapping with KeplerMapper...")
        graph = mapper.map(
            lens, 
            df_numeric,
            clusterer=DBSCAN(eps=0.5, min_samples=5), 
            cover=km.Cover(n_cubes=20, perc_overlap=0.5)
        )

        # Visualizar el gráfico con Plotly y guardarlo directamente
        logger.info("Visualizing Mapper graph...")
        mapper.visualize(graph, path_html=str(output_path))  # Visualiza y guarda el gráfico en el archivo

        logger.info(f"Interactive persistence homology graph saved to {output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    app()
