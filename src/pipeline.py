from prefect import flow
from src.data_ingestion import ingest_and_load_data
from src.preprocessing import preprocess_and_fuse_data
from src.train import train_model_task
from src.evaluate import evaluate_model_task
import os
import pandas as pd

@flow(name="NYC Crime Full ETL and Training")
def crime_prediction_pipeline():
    urls = {
        "complaints": "https://data.cityofnewyork.us/resource/qgea-i56i.csv?$limit=50000", 
        "arrests": "https://data.cityofnewyork.us/resource/uip8-fykc.csv",
        "vehicle_stops": "https://data.cityofnewyork.us/resource/fe2c-6g96.csv" 
    }
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True) 
    os.makedirs('app/static/img', exist_ok=True) 

    print("Iniciando ingestión de datos...")
    raw_dataframes = ingest_and_load_data(urls)
    print("Ingestión de datos completada.")

    # PREPROCESAMIENTO Y FUSIÓN DE DATOS
    print("Llamando a la tarea preprocess_and_fuse_data...")
    # Almacena el resultado en una variable primero
    preprocess_result = preprocess_and_fuse_data(
        complaints_df=raw_dataframes.get("complaints", pd.DataFrame()), 
        arrests_df=raw_dataframes.get("arrests", pd.DataFrame()),
        vehicle_stops_df=raw_dataframes.get("vehicle_stops", pd.DataFrame())
    )
    print("Tarea de preprocesamiento finalizada. Desempaquetando resultados...")

    try:
        X_train, X_test, y_train, y_test, label_encoder, location_analysis = preprocess_result
        print(" -> Éxito al desempaquetar 6 valores.")
    except ValueError as e:
        print(f" ERROR durante el desempaquetado: {e}")
        print(f" -> Tipo devuelto por preprocess_and_fuse_data: {type(preprocess_result)}")
        if isinstance(preprocess_result, tuple):
            print(f" -> Longitud de la tupla devuelta: {len(preprocess_result)}")
        raise e
    except TypeError as e:
         print(f" ERROR: El resultado podría no ser desempaquetable (¿no es una tupla/secuencia?): {e}")
         print(f" -> Tipo devuelto por preprocess_and_fuse_data: {type(preprocess_result)}")
         raise e

    # ENTRENAMIENTO DEL MODELO
    print("Llamando a la tarea train_model_task...")
    model = train_model_task(X_train, y_train)
    print("Entrenamiento del modelo completado.")

    #EVALUACIÓN DEL MODELO
    print("Llamando a la tarea evaluate_model_task...")
    evaluate_model_task(model, X_test, y_test, label_encoder, location_analysis)
    print("Evaluación del modelo completada.")

    print("\n--- PIPELINE COMPLETADO EXITOSAMENTE ---")


if __name__ == "__main__":
    crime_prediction_pipeline()