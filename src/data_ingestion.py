import pandas as pd
import requests
from prefect import task
from pathlib import Path

@task(retries=3, retry_delay_seconds=10)
def download_data(url: str, save_path: Path):
    """Descarga datos de una URL y los guarda."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Descargando datos desde {url}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Datos guardados en {save_path}")
        return save_path
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar desde {url}: {e}")
        return None

@task
def load_data(path: Path) -> pd.DataFrame:
    """Carga un archivo CSV a un DataFrame de pandas."""
    if path and path.exists():
        print(f"Cargando datos desde {path}...")
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()

@task
def ingest_and_load_data(urls: dict) -> dict:
    """
    Orquesta la descarga y carga de múltiples datasets.
    Retorna un diccionario de DataFrames.
    """
    dataframes = {}
    for name, url in urls.items():
        save_path = Path(f"data/raw/{name}.csv")
        downloaded_path = download_data(url, save_path)
        dataframes[name] = load_data(downloaded_path)
    return dataframes