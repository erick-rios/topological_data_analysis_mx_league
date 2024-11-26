from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import typer
from loguru import logger
from tqdm import tqdm
from ripser import ripser
from scipy.spatial import distance_matrix
import numpy as np

from topological_data_analysis_mx_league.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "normalized_features.csv",
    output_path: Path = FIGURES_DIR / "persistent_homology",
):
    try:
        logger.info("Loading dataset...")
        df = pd.read_csv(input_path)

        # Inspección inicial de las columnas
        logger.info(f"Loaded dataset with columns: {df.columns.tolist()}")

        # Seleccionar solo las columnas numéricas para el análisis topológico
        numeric_columns = df.select_dtypes(include=[float, int]).columns.tolist()

        # Filtrar los datos numéricos
        df_numeric = df[numeric_columns]

        # Calcular la matriz de distancias
        logger.info("Calculating distance matrix...")
        dist_matrix = distance_matrix(df_numeric, df_numeric)

        # Realizar el cálculo de la homología persistente
        logger.info("Computing persistent homology...")
        result = ripser(dist_matrix, maxdim=2, distance_matrix=True)

        # Extraer los diagramas de persistencia
        diagrams = result['dgms']

        # Visualizar los diagramas de persistencia usando matplotlib
        logger.info("Plotting persistence diagrams...")
        plt.figure(figsize=(10, 6))
        for dim, diagram in enumerate(diagrams):
            # Graficar puntos (diagrama de persistencia)
            plt.scatter(diagram[:, 0], diagram[:, 1], label=f"Dimension {dim}")
        plt.title("Persistence Diagram")
        plt.xlabel("Birth")
        plt.ylabel("Death")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path / 'persistence_diagram.png')
        plt.close()

        # Opcional: Graficar la persistencia filtrada
        logger.info("Plotting barcode diagram...")
        plt.figure(figsize=(10, 6))
        for dim, diagram in enumerate(diagrams):
            # Graficar las barras de persistencia
            for birth, death in diagram:
                plt.plot([birth, death], [dim, dim], lw=5)
        plt.title("Barcode Diagram")
        plt.xlabel("Value")
        plt.ylabel("Dimension")
        plt.grid(True)
        plt.savefig(output_path / 'barcode_diagram.png')
        plt.close()

        # Mostrar el resumen de los diagramas de persistencia
        logger.info(f"Persistent homology computed. Diagrams saved in {output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    app()
