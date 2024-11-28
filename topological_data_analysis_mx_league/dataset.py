import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService

from topological_data_analysis_mx_league.config import RAW_DATA_DIR


def scrape_and_save_with_selenium(output_path: Path, url: str, table_id: str, driver_path: str = "/usr/bin/chromedriver"):
    """
    Scrapes a webpage using Selenium, extracts an HTML table, and saves it as a DataFrame.

    Args:
        output_path (Path): Path to save the extracted DataFrame.
        url (str): URL of the webpage to scrape.
        table_id (str): The ID of the HTML table to extract.
        driver_path (str): Path to the ChromeDriver executable.
    """
    logger.info(f"Starting web scraping from {url}")
    
    # Configuración de Selenium
    service = ChromeService(executable_path=driver_path)
    options = ChromeOptions()
    options.add_argument("--headless")  # Ejecutar en modo sin interfaz gráfica
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        logger.info("Loading the webpage...")
        driver.get(url)
        logger.info("Page loaded successfully. Parsing the HTML content...")
        
        # Parsear contenido HTML
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Buscar la tabla por ID
        table = soup.find("table", {"id": table_id})
        if not table:
            logger.error(f"Table with ID '{table_id}' not found.")
            return
        
        # Convertir la tabla a un DataFrame de pandas
        df = pd.read_html(str(table))[0]
        
        # Validar que el DataFrame no esté vacío
        if df.empty:
            logger.error("Extracted table is empty. Aborting save.")
            return
        
        # Mostrar dimensiones de la tabla extraída
        logger.info(f"Extracted table dimensions: {df.shape[0]} rows, {df.shape[1]} columns.")
        
        # Guardar los datos en formato CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.success(f"Data successfully saved to {output_path}")
    
    except Exception as e:
        logger.error(f"An error occurred during scraping: {e}")
    
    finally:
        driver.quit()
        logger.info("Selenium driver closed.")

if __name__ == "__main__":
    # Parámetros del scraping
    scrape_and_save_with_selenium(
        output_path=RAW_DATA_DIR / "liga_mx_stats.csv",
        url="https://fbref.com/es/comps/31/stats/Estadisticas-de-Liga-MX",
        table_id="stats_standard",
        driver_path="/usr/bin/chromedriver"  # Cambiar si está en otra ruta
    )
