# src/pipeline.py (ACTUALIZADO PARA LA HIPÓTESIS POLICIAL)

from prefect import flow
from src.data_ingestion import ingest_and_load_data
from src.preprocessing import preprocess_and_fuse_data # Renombramos la función para claridad
from src.train import train_model_task
from src.evaluate import evaluate_model_task
import os
import pandas as pd

@flow(name="NYC Crime Full ETL and Training")
def crime_prediction_pipeline():
    # ...

    # URLs de los datasets
# URLs de los datasets (CON TODAS LAS URLs CORRECTAS)

    urls = {
        "complaints": "https://data.cityofnewyork.us/resource/qgea-i56i.csv",
        "arrests": "https://data.cityofnewyork.us/resource/uip8-fykc.csv",
        # ¡ESTA ES LA URL MÁS RECIENTE Y ACTIVA PARA LAS PARADAS!
        "vehicle_stops": "https://data.cityofnewyork.us/resource/fe2c-6g96.csv" 
    }

    # Asegurarse de que los directorios necesarios existan
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    # --- 1. INGESTIÓN DE DATOS ---
    raw_dataframes = ingest_and_load_data(urls)

    # --- 2. PREPROCESAMIENTO Y FUSIÓN DE DATOS ---
    X_train, X_test, y_train, y_test, label_encoder, location_analysis = preprocess_and_fuse_data(
        complaints_df=raw_dataframes["complaints"],
        arrests_df=raw_dataframes["arrests"],
        vehicle_stops_df=pd.DataFrame() # Pasamos un DF vacío, ya no lo usamos
    )

    model = train_model_task(X_train, y_train)

    # --- 4. EVALUACIÓN DEL MODELO ---
    evaluate_model_task(model, X_test, y_test, label_encoder, location_analysis)

    print("--- PIPELINE COMPLETADO EXITOSAMENTE ---")


if __name__ == "__main__":
    crime_prediction_pipeline()