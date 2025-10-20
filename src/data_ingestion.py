# src/data_ingestion.py
import pandas as pd
import os

def fetch_data(url: str, save_path: str):
    """Downloads data from a URL and saves it to a CSV file.
       Descarga datos de una URL y los guarda en un archivo CSV."""
    print(f"Fetching data from {url}...")
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df = pd.read_csv(url, dtype={'cmplnt_num': str})
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")
    return save_path