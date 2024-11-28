from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import typer
from loguru import logger
from tqdm import tqdm

from topological_data_analysis_mx_league.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "non_normalized_liga_mx_data.csv",
    output_dir: Path = FIGURES_DIR / "exploratory_analysis",
):
    # Crear directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar dataset
    logger.info(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)

    # Filtrar columnas numéricas y categóricas
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    logger.info(f"Numeric columns: {numeric_columns}")

    # Crear histogramas para variables numéricas
    logger.info("Generating histograms for numeric columns...")
    for col in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True, bins=20, color='skyblue', alpha=0.8)
        plt.title(f"Distribution of {col}", fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        file_path = output_dir / f"{col}_histogram.png"
        plt.savefig(file_path, dpi=300)
        plt.close()
        logger.info(f"Saved histogram for {col} to {file_path}")

    # Heatmap de correlación
    logger.info("Generating correlation heatmap...")
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={"shrink": 0.8}, linewidths=0.5)
    plt.title('Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    heatmap_path = output_dir / "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    logger.info(f"Saved correlation heatmap to {heatmap_path}")

    # Pairplot para relaciones entre variables numéricas
    logger.info("Generating pairplot for numeric columns...")
    pairplot_path = output_dir / "pairplot_numeric_columns.png"
    pairplot = sns.pairplot(df[numeric_columns], diag_kind="kde", plot_kws={'alpha': 0.6, 's': 40})
    pairplot.fig.suptitle("Pairplot of Numeric Columns", y=1.02, fontsize=16)
    pairplot.savefig(pairplot_path, dpi=300)
    plt.close()
    logger.info(f"Saved pairplot to {pairplot_path}")

    logger.success("EDA visualizations saved as PNG files in the output directory.")

if __name__ == "__main__":
    app()
