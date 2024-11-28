from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import typer
from loguru import logger
from ripser import ripser
from scipy.spatial import distance_matrix
from topological_data_analysis_mx_league.config import FIGURES_DIR, INTERIM_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "pca_components.csv",
    output_path: Path = FIGURES_DIR / "persistent_homology",
    maxdim: int = 2,
):
    try:
        logger.info("Loading dataset...")
        df = pd.read_csv(input_path)

        # Seleccionar solo las columnas numéricas
        logger.info("Filtering numeric columns...")
        numeric_columns = df.select_dtypes(include=[float, int]).columns.tolist()
        df_numeric = df[numeric_columns]

        # Validar si hay datos suficientes
        if df_numeric.empty:
            raise ValueError("No numeric data found in the dataset.")

        # Calcular la matriz de distancias
        logger.info("Calculating distance matrix...")
        dist_matrix = distance_matrix(df_numeric, df_numeric)

        # Calcular homología persistente
        logger.info("Computing persistent homology...")
        result = ripser(dist_matrix, maxdim=maxdim, distance_matrix=True)
        diagrams = result['dgms']

        # Visualizar diagrama de persistencia
        logger.info("Plotting persistence diagram...")
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        for dim, diagram in enumerate(diagrams):
            sns.scatterplot(
                x=diagram[:, 0],
                y=diagram[:, 1],
                label=f"Dimension {dim}",
                s=100,
                edgecolor="black"
            )
        plt.plot([0, max(diagram[:, 1].max() for diagram in diagrams)],
                 [0, max(diagram[:, 1].max() for diagram in diagrams)],
                 color="red", linestyle="--", label="y=x (Diagonal)")
        plt.title("Persistence Diagram")
        plt.xlabel("Birth")
        plt.ylabel("Death")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / "persistence_diagram_sns.png")
        plt.close()

        # Graficar barras de persistencia (barcode)
        logger.info("Plotting barcode diagram...")
        plt.figure(figsize=(10, 6))
        for dim, diagram in enumerate(diagrams):
            for birth, death in diagram:
                plt.hlines(
                    y=dim,
                    xmin=birth,
                    xmax=death,
                    color="blue",
                    linewidth=2
                )
        plt.title("Barcode Diagram")
        plt.xlabel("Value")
        plt.ylabel("Dimension")
        plt.tight_layout()
        plt.savefig(output_path / "barcode_diagram_sns.png")
        plt.close()

        logger.success(f"Persistent homology completed successfully. Results saved in {output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    app()

