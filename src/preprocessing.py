# src/preprocessing.py
import pandas as pd
import numpy as np # Necesitamos numpy

def preprocess(raw_data_path: str, processed_data_path: str, target: str):
    """Cleans data, engineers features, and creates the binary target variable."""
    print("Starting preprocessing for regression task...")
    df = pd.read_csv(raw_data_path, low_memory=False)

    print("Columns found in raw data:", df.columns.tolist())
    
    col_date = 'cmplnt_fr_dt'
    col_time = 'cmplnt_fr_tm'
    col_premise = 'prem_typ_desc'
    col_offense = 'ofns_desc' # Columna clave para definir "robo"

    # --- Creación de la Variable Objetivo (Target) ---
    # Creamos la columna 'is_robbery'. Será 1 si la descripción es 'ROBBERY', y 0 en otro caso.
    # Usamos .str.contains() para ser más robustos por si hay variaciones.
    df[target] = np.where(df[col_offense].str.contains('ROBBERY', na=False), 1, 0)
    
    # Limpieza básica
    critical_cols = ['latitude', 'longitude', col_date, col_time, col_premise]
    df.dropna(subset=critical_cols, inplace=True)
    
    # Ingeniería de Características (Feature Engineering)
    df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
    df[col_time] = pd.to_datetime(df[col_time], format='%H:%M:%S', errors='coerce').dt.hour
    df.dropna(subset=[col_date, col_time], inplace=True)

    df.rename(columns={col_time: 'hour'}, inplace=True)
    df['day_of_week'] = df[col_date].dt.day_name()
    
    top_premises = df[col_premise].value_counts().nlargest(10).index
    df[col_premise] = df[col_premise].apply(lambda x: x if x in top_premises else 'OTHER')

    # Seleccionar columnas finales y guardar
    final_cols = [target] + ['latitude', 'longitude', col_premise, 'hour', 'day_of_week']
    df_processed = df[final_cols]
    
    df_processed.to_csv(processed_data_path, index=False)
    print(f"Preprocessing complete. Data saved to {processed_data_path}")
    return processed_data_path