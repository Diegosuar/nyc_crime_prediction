# src/pipeline.py
from prefect import task, flow
from .config import COMPLAINTS_URL, RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET, MODEL_PATH, PREPROCESSOR_PATH
from .data_ingestion import fetch_data
from .preprocessing import preprocess
from .train import train_model
from .evaluate import evaluate

@task
def fetch_data_task():
    return fetch_data(url=COMPLAINTS_URL, save_path=RAW_DATA_PATH)

@task
def preprocess_data_task(raw_path):
    return preprocess(raw_data_path=raw_path, processed_data_path=PROCESSED_DATA_PATH, target=TARGET)

@task
def train_model_task(processed_path):
    return train_model(processed_data_path=processed_path, model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH)

@task
def evaluate_model_task(processed_path, model_path):
    evaluate(processed_data_path=processed_path, model_path=model_path)

@flow(name="NYC Crime Prediction Flow")
def crime_prediction_flow():
    """Main pipeline to run the full ML workflow.
       Pipeline principal para ejecutar el flujo de trabajo de ML completo."""
    raw_path = fetch_data_task()
    processed_path = preprocess_data_task(raw_path)
    model_path_result = train_model_task(processed_path)
    evaluate_model_task(processed_path, model_path_result)

if __name__ == "__main__":
    crime_prediction_flow()