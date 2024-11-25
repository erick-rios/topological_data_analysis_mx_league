from pathlib import Path
from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from bs4 import BeautifulSoup
import pandas as pd
import pickle

from topological_data_analysis_mx_league.config import PROCESSED_DATA_DIR

def scrape_and_save_with_selenium(output_path: Path, url: str, table_id: str):
    """
    Scrapes a webpage using Selenium, extracts an HTML table, and saves it as a DataFrame.

    Args:
        output_path (Path): Path to save the extracted DataFrame.
        url (str): URL of the webpage to scrape.
        table_id (str): The ID of the HTML table to extract.
    """
    logger.info(f"Starting web scraping from {url}")
    
    service = ChromeService(executable_path="/usr/bin/chromedriver")  
    options = ChromeOptions()
    options.add_argument("--headless") 
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    
    # Inicia el controlador de Chrome
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        driver.get(url)
        logger.info("Page loaded successfully.")
        logger.info("Parsing the HTML content...")
        
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        
        table = soup.find("table", {"id": table_id})
        if not table:
            logger.error(f"Table with ID '{table_id}' not found.")
            return
        
        
        df = pd.read_html(str(table))[0]
        logger.info(f"Table with ID '{table_id}' extracted successfully. Saving to {output_path}")
        
        
        with open(output_path, "wb") as f:
            pickle.dump(df, f)
        logger.success(f"Data saved to {output_path}")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    finally:
        driver.quit()

if __name__ == "__main__":
    scrape_and_save_with_selenium(
        output_path=PROCESSED_DATA_DIR / "liga_mx_stats.pkl",
        url="https://fbref.com/es/comps/31/stats/Estadisticas-de-Liga-MX",
        table_id="stats_standard",
    )
