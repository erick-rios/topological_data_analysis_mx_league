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
    input_path: Path = PROCESSED_DATA_DIR / "normalized_features.csv",  # ruta de entrada al archivo CSV
    output_dir: Path = FIGURES_DIR,  # directorio para guardar las figuras generadas
):
    # Cargar el dataset
    logger.info(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)

    # Estadísticas descriptivas
    logger.info("Generating descriptive statistics...")
    logger.info(f"Dataset summary:\n{df.describe(include='all')}")

    # Graficar las distribuciones de variables numéricas
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    logger.info(f"Numeric columns found: {numeric_columns}")
    
    for col in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True, color='skyblue', bins=20)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(output_dir / f"{col}_distribution.png")
        plt.close()

    # Graficar Box plots
    logger.info("Generating box plots...")
    for col in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col], color='salmon')
        plt.title(f"Box plot of {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(output_dir / f"{col}_boxplot.png")
        plt.close()

    # Graficar un gráfico de pastel para las variables categóricas
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    logger.info(f"Categorical columns found: {categorical_columns}")
    
    # Preprocesar la columna "Edad" para obtener solo la parte entera de la edad
    if 'Edad' in df.columns:
        # Eliminar los valores "Unknown" o no numéricos antes de proceder
        df['Edad'] = df['Edad'].str.split('-', expand=True)[0]
        df['Edad'] = pd.to_numeric(df['Edad'], errors='coerce')  # Convertir a numérico, convertir errores a NaN
        df = df.dropna(subset=['Edad'])  # Eliminar filas con valores NaN en 'Edad'

    for col in categorical_columns:
        if col == "Edad":
            # Generar gráfico de pastel solo con la edad entera
            plt.figure(figsize=(8, 6))
            df['Edad'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette("Set3", len(df['Edad'].unique())))
            plt.title("Age Distribution")
            plt.ylabel('')
            plt.tight_layout()
            plt.savefig(output_dir / f"{col}_pie_chart.png")
            plt.close()
        else:
            # Para otras columnas categóricas, se generan gráficos de pastel como antes
            plt.figure(figsize=(8, 6))
            df[col].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette("Set3", len(df[col].unique())))
            plt.title(f"Pie chart of {col}")
            plt.ylabel('')
            plt.tight_layout()
            plt.savefig(output_dir / f"{col}_pie_chart.png")
            plt.close()

    # Heatmap de correlación para ver las relaciones lineales
    logger.info("Generating correlation heatmap...")
    correlation_matrix = df[numeric_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()

    # Pairplot para ver relaciones entre las variables
    logger.info("Generating pairplot...")
    sns.pairplot(df[numeric_columns], diag_kind="kde", plot_kws={'alpha':0.5, 's':50})
    plt.suptitle("Pairplot of Numeric Columns", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "pairplot.png")
    plt.close()

    # Tabla de frecuencias para variables categóricas
    logger.info("Generating frequency tables for categorical variables...")
    for col in categorical_columns:
        frequency_table = df[col].value_counts()
        logger.info(f"Frequency table for {col}:\n{frequency_table}")
        frequency_table.to_csv(output_dir / f"{col}_frequency_table.csv")

    logger.success("Exploratory data analysis and visualizations complete.")

if __name__ == "__main__":
    app()
