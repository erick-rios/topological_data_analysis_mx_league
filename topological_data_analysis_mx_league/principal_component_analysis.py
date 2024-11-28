from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import typer
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from topological_data_analysis_mx_league.config import FIGURES_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR/"cleaned_liga_mx_data.csv",  # Ruta de entrad
    output_csv: Path = INTERIM_DATA_DIR/"pca_components.csv",   # Salida del PC
    output_dir: Path = FIGURES_DIR/"pca_analysis",        # Carpeta de figura
    n_components: int = 2                                    # Número de componentes a conservar
):
    try:
        # Crear directorios de salida
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        # Cargar datos
        logger.info("Loading dataset...")
        df = pd.read_csv(input_path)
        logger.info(f"Dataset loaded with shape {df.shape}")

        # Seleccionar columnas numéricas
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        logger.info(f"Numeric columns: {numeric_columns}")

        # Preprocesamiento: imputación y escalado
        logger.info("Preprocessing data...")
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        df_numeric = df[numeric_columns]
        processed_data = numeric_transformer.fit_transform(df_numeric)

        # Aplicar PCA
        logger.info(f"Applying PCA to retain {n_components} components...")
        pca = PCA(n_components=n_components)
        pca_results = pca.fit_transform(processed_data)

        # Guardar componentes principales en un DataFrame
        pca_columns = [f"PC{i+1}" for i in range(n_components)]
        df_pca = pd.DataFrame(pca_results, columns=pca_columns)
        df_pca.to_csv(output_csv, index=False)
        logger.info(f"PCA results saved to {output_csv}")

        # Gráficos
        # 1. Varianza explicada
        explained_variance = pca.explained_variance_ratio_
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_components + 1), explained_variance, alpha=0.7, color='skyblue')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance per Principal Component')
        plt.xticks(range(1, n_components + 1))
        plt.tight_layout()
        plt.savefig(output_dir / 'explained_variance.png', dpi=300)
        plt.close()

        # 2. Proyección en las dos primeras componentes
        if n_components >= 2:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                x=df_pca["PC1"], y=df_pca["PC2"],
                s=100, edgecolor='black', alpha=0.8
            )
            plt.title('PCA: First and Second Principal Components')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.tight_layout()
            plt.savefig(output_dir / 'pca_projection.png', dpi=300)
            plt.close()

        # 3. Varianza explicada acumulada
        cumulative_variance = explained_variance.cumsum()
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_components + 1), cumulative_variance, marker='o', linestyle='-', color='b')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance by Principal Components')
        plt.xticks(range(1, n_components + 1))
        plt.tight_layout()
        plt.savefig(output_dir / 'cumulative_variance.png', dpi=300)
        plt.close()

        logger.success("PCA analysis and visualizations complete.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    app()
