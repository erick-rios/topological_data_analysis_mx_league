from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pickle

from topological_data_analysis_mx_league.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def scrape_and_save(
    # Replace default paths as appropriate
    output_path: Path = PROCESSED_DATA_DIR / "liga_mx_stats.pkl",
    url: str = "https://fbref.com/es/comps/31/stats/Estadisticas-de-Liga-MX",
    table_id: str = "stats_standard",
):
    """
    Scrapes Liga MX statistics from the given URL and saves them to a pickle file.
    
    Args:
        output_path (Path): Path to save the processed data.
        url (str): URL to scrape the data from.
        table_id (str): ID of the table to scrape in the HTML page.
    """
    logger.info(f"Starting web scraping from {url}")

    try:
        # Fetch the HTML content
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        html_content = response.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch data: {e}")
        return

    try:
        # Parse the table and extract data
        logger.info("Parsing the HTML content...")
        soup = BeautifulSoup(html_content, "html.parser")
        table = soup.find("table", {"id": table_id})
        if not table:
            logger.error(f"Table with ID '{table_id}' not found.")
            return

        # Extract headers
        headers = [th.text for th in table.find_all("th")[1:]]  # Skip the first empty header

        # Extract rows
        rows = table.find_all("tr")[1:]  # Skip the header row
        data = []
        for row in tqdm(rows, desc="Processing rows"):
            cols = row.find_all("td")
            if cols:  # Skip empty rows
                data.append([col.text for col in cols])

        # Create a DataFrame
        df = pd.DataFrame(data, columns=headers)
    except Exception as e:
        logger.error(f"Error parsing data: {e}")
        return

    try:
        # Save the DataFrame to a pickle file
        logger.info(f"Saving data to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as file:
            pickle.dump(df, file)
        logger.success("Data successfully saved.")
    except Exception as e:
        logger.error(f"Failed to save data: {e}")


if __name__ == "__main__":
    app()
