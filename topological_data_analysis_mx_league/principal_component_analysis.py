from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import typer
from loguru import logger
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from topological_data_analysis_mx_league.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "normalized_features.csv",
    output_path: Path = FIGURES_DIR / "pca/pca_analysis.png",
):
    try:
        logger.info("Loading dataset...")
        df = pd.read_csv(input_path)

        # Inspección inicial de las columnas
        logger.info(f"Loaded dataset with columns: {df.columns.tolist()}")

        # Seleccionar solo las columnas numéricas
        numeric_columns = df.select_dtypes(include=[float, int]).columns.tolist()

        # Imputar valores faltantes y escalar las variables numéricas
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Imputar valores faltantes
            ('scaler', StandardScaler())  # Escalar las variables numéricas
        ])

        # Aplicar preprocesamiento solo a las columnas numéricas
        df_numeric = df[numeric_columns]

        # Aplicar PCA solo a las variables numéricas
        logger.info("Applying PCA...")
        pca = PCA()
        df_pca = numeric_transformer.fit_transform(df_numeric)

        # Ajustar PCA a los datos numéricos procesados
        pca.fit(df_pca)

        # Obtener la varianza explicada por cada componente principal
        explained_variance = pca.explained_variance_ratio_

        # Graficar la varianza explicada por cada componente
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance per Principal Component')
        plt.xticks(range(1, len(explained_variance) + 1))
        plt.savefig(FIGURES_DIR / 'explained_variance.png')
        plt.show()

        # Obtener las dos primeras componentes principales
        pca_components = pca.transform(df_pca)[:, :2]

        # Graficar la proyección de los datos en las primeras dos componentes principales
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], s=100, edgecolor='black')
        plt.title('PCA: First and Second Principal Components')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.savefig(FIGURES_DIR / 'pca_projection.png')
        plt.show()

        # Percentage of Variance Contained in Each Principal Component (Cumulative)
        cumulative_variance = pca.explained_variance_ratio_.cumsum()
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='b')
        plt.xlabel('Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance by Principal Components')
        plt.xticks(range(1, len(cumulative_variance) + 1))
        plt.savefig(FIGURES_DIR / 'cumulative_variance.png')
        plt.show()

        # Guards Distribution in the New Axes: Distribution of variables on the first principal component
        plt.figure(figsize=(10, 6))
        sns.histplot(pca_components[:, 0], kde=True, color='green', bins=30)
        plt.title('Distribution of Data on the First Principal Component')
        plt.xlabel('First Principal Component')
        plt.ylabel('Frequency')
        plt.savefig(FIGURES_DIR / 'guards_distribution_first_component.png')
        plt.show()

        # Forward Distribution: Forward distribution can be interpreted as how the data spreads along the principal components
        plt.figure(figsize=(10, 6))
        sns.histplot(pca_components[:, 1], kde=True, color='red', bins=30)
        plt.title('Distribution of Data on the Second Principal Component')
        plt.xlabel('Second Principal Component')
        plt.ylabel('Frequency')
        plt.savefig(FIGURES_DIR / 'forward_distribution_second_component.png')
        plt.show()

        # Center Case: Plot center of the data, can be represented as mean of the principal components
        center_x, center_y = pca_components.mean(axis=0)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], s=100, edgecolor='black', color='blue', label='Data')
        plt.scatter(center_x, center_y, color='red', s=200, marker='X', label='Center Case')
        plt.title('PCA: Data with Center Case Marked')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        plt.savefig(FIGURES_DIR / 'center_case.png')
        plt.show()

        logger.success("PCA analysis and visualizations complete.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    app()
