from pathlib import Path
import pandas as pd
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from topological_data_analysis_mx_league.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "liga_mx_stats.csv",
    output_path: Path = PROCESSED_DATA_DIR / "cleaned_liga_mx_data.csv",
):
    """
    Cleans and preprocesses Liga MX data for topological data analysis. 
    The script:
    - Removes invalid rows (e.g., repeated header rows within the file).
    - Extracts only relevant columns for analysis.
    - Cleans numerical columns (e.g., "Age") and normalizes stats based on minutes played.
    - Detects and handles outliers.
    - Saves the cleaned dataset to a new file.

    Args:
        input_path (Path): Path to the raw dataset (CSV format).
        output_path (Path): Path to save the cleaned dataset (CSV format).
    """
    try:
        logger.info(f"Loading dataset from {input_path}...")

        # Load the dataset while ignoring the first row and fixing headers
        df = pd.read_csv(input_path, skiprows=1)
        logger.info("Dataset loaded successfully.")

        # Remove rows that are repeated header rows
        logger.info("Removing repeated header rows...")
        df = df[df["RL"] != "RL"]

        # Ensure correct column names
        logger.info("Assigning proper column names...")
        df.columns = [
            "RL", "Player", "Country", "Position", "Team", "Age", "Birthdate", 
            "Matches_Played", "Starts", "Minutes", "90s", "Goals", "Assists", 
            "G+A", "G-TP", "Shots_on_Target", "Shots", "Yellow_Cards", 
            "Red_Cards", "xG", "npxG", "xAG", "npxG+xAG", "Progressive_Carries", 
            "Progressive_Passes", "Progressive_Receptions", "Goals_per_90", 
            "Assists_per_90", "G+A_per_90", "G-TP_per_90", "G+A-TP_per_90", 
            "xG_per_90", "xAG_per_90", "xG+xAG_per_90", "npxG_per_90", 
            "npxG+xAG_per_90", "Matches"
        ]

        # Select only relevant columns for analysis
        logger.info("Filtering relevant columns...")
        columns_to_keep = ["Player", "Country", "Position", "Team", "Age", 
                           "Matches_Played", "Minutes", "Goals", "Assists", 
                           "Shots_on_Target", "Yellow_Cards", "Red_Cards"]
        df = df[columns_to_keep]

        # Extract the integer part of the "Age" column (before the "-")
        logger.info("Cleaning 'Age' column...")
        df["Age"] = df["Age"].str.split("-").str[0].astype(float)

        # Ensure all numeric columns are properly formatted
        logger.info("Converting columns to numeric...")
        numeric_columns = [
            "Age", "Matches_Played", "Minutes", "Goals", "Assists", 
            "Shots_on_Target", "Yellow_Cards", "Red_Cards"
        ]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove rows with missing values in numeric columns
        df.dropna(subset=numeric_columns, inplace=True)

        # Filter rows where "Minutes" > 0
        logger.info("Filtering players with valid minutes played...")
        df = df[df["Minutes"] > 0]

        # Detect and handle outliers
        logger.info("Detecting and handling outliers...")
        for col in numeric_columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Cap values outside the bounds
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

        # Normalize numeric stats based on minutes played
        logger.info("Normalizing stats per minute played...")
        stats_columns = ["Goals", "Assists", "Shots_on_Target", "Yellow_Cards", "Red_Cards"]
        for col in tqdm(stats_columns, desc="Normalizing stats"):
            df[f"{col}_per_minute"] = df[col] / df["Minutes"]

        # Remove duplicate entries based on the "Player" column
        logger.info("Removing duplicate player entries...")
        df.drop_duplicates(subset=["Player"], inplace=True)

        # Save the cleaned dataset
        logger.info(f"Saving cleaned dataset to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.success("Dataset cleaning and normalization complete.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    app()
