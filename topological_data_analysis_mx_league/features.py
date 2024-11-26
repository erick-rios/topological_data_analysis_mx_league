from pathlib import Path
import pandas as pd
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from topological_data_analysis_mx_league.config import RAW_DATA_DIR
from topological_data_analysis_mx_league.config import PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "liga_mx_stats.pkl",
    output_path: Path = PROCESSED_DATA_DIR / "normalized_features.csv",
):
    try:
        logger.info(f"Loading dataset from {input_path}...")
        with open(input_path, "rb") as f:
            df = pd.DataFrame(pd.read_pickle(f))

        # Aplanar las columnas de MultiIndex
        df.columns = ["_".join(col).strip() for col in df.columns.values]

        # Inspección de las columnas
        logger.info(f"Flattened dataset columns: {df.columns.tolist()}")

        # Ajustar los nombres de las columnas requeridas al nuevo formato
        required_columns = [
            "Unnamed: 1_level_0_Jugador",
            "Tiempo Jugado_Mín",
            "Rendimiento_Gls.",
            "Rendimiento_Ass",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns in dataset: {missing_columns}")
            return

        logger.info("Performing exploratory data analysis...")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Dataset summary:\n{df.describe(include='all')}")

        # Limpieza de datos
        logger.info("Cleaning dataset...")
        # Eliminar columnas completamente vacías
        df.dropna(how="all", axis=1, inplace=True)

        # Rellenar valores faltantes en columnas numéricas con 0
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_columns] = df[numeric_columns].fillna(0)

        # Rellenar valores faltantes en columnas no numéricas con una etiqueta predeterminada
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
        df[non_numeric_columns] = df[non_numeric_columns].fillna("Unknown")

        # Mostrar valores únicos en columnas no numéricas para identificar errores
        for col in non_numeric_columns:
            logger.info(f"Unique values in column '{col}': {df[col].unique()}")

        # Convertir columnas relevantes a numéricas
        columns_to_normalize = ["Tiempo Jugado_Mín", "Rendimiento_Gls.", "Rendimiento_Ass"]
        for col in columns_to_normalize:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Eliminar filas con valores nulos después de la conversión
        df.dropna(subset=columns_to_normalize, inplace=True)

        # Filtrar filas donde "Tiempo Jugado_Mín" sea mayor a 0
        df = df[df["Tiempo Jugado_Mín"] > 0]

        # Normalización de datos
        logger.info("Normalizing data by minutes played...")
        stats_columns = ["Rendimiento_Gls.", "Rendimiento_Ass"]
        for col in tqdm(stats_columns, desc="Normalizing stats"):
            df[col + "_per_minute"] = df[col] / df["Tiempo Jugado_Mín"]

        # Renombrar columnas
        logger.info("Renaming columns...")
        df.drop(df.columns[0], axis=1, inplace=True)  # Eliminar la columna 0
        df.columns = [
            col.split("_")[-1] if "_" in col else col  # Conservar solo la última parte después del "_"
            for col in df.columns
        ]

        # Filtrar las columnas que te interesan
        columns_to_keep = [
            "Jugador", "País", "Posc", "Equipo", "Edad", "Nacimiento", 
            "PJ", "Titular", "Mín", "90 s", "Gls.", "Ass", "TP", "TA", "TR"
        ]
        df = df[columns_to_keep]

        # Guardar el dataset normalizado
        logger.info(f"Saving normalized dataset to {output_path}...")
        df.to_csv(output_path, index=False)
        logger.success("Feature generation, normalization, and column renaming complete.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    app()

