# src/preprocessing.py
import pandas as pd
import numpy as np

def preprocess(raw_data_path: str, processed_data_path: str, target: str):
    """Limpia datos y crea características para la tarea de clasificación multiclase."""
    print(f"Starting preprocessing for multiclass task (Target: {target})...")
    df = pd.read_csv(raw_data_path, low_memory=False, dtype={'cmplnt_num': str})

    col_date = 'cmplnt_fr_dt'
    col_time = 'cmplnt_fr_tm'
    col_premise = 'prem_typ_desc'
    # --- CAMBIO DE LÓGICA ---
    # 'law_cat_cd' (nuestro target) ahora se llama 'target'
    col_law_cat = target 

    # Limpieza básica
    critical_cols = ['latitude', 'longitude', col_date, col_time, col_premise, col_law_cat]
    df.dropna(subset=critical_cols, inplace=True)
    
    # Ingeniería de Características (Feature Engineering)
    df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
    df[col_time] = pd.to_datetime(df[col_time], format='%H:%M:%S', errors='coerce').dt.hour
    
    # Nos aseguramos de que solo se usen las 3 clases principales
    valid_targets = ['FELONY', 'MISDEMEANOR', 'VIOLATION']
    df = df[df[col_law_cat].isin(valid_targets)]
    
    df.dropna(subset=[col_date, col_time], inplace=True)

    df.rename(columns={col_time: 'hour'}, inplace=True)
    df['day_of_week'] = df[col_date].dt.day_name()
    
    # Agrupar premisas poco comunes en 'OTHER'
    top_premises = df[col_premise].value_counts().nlargest(10).index
    df[col_premise] = df[col_premise].apply(lambda x: x if x in top_premises else 'OTHER')

    # Seleccionar columnas finales y guardar
    # (Ya no necesitamos crear 'is_robbery')
    final_cols = [target] + ['latitude', 'longitude', col_premise, 'hour', 'day_of_week']
    df_processed = df[final_cols]
    
    df_processed.to_csv(processed_data_path, index=False)
    print(f"Preprocessing complete. Data saved to {processed_data_path}")
    return processed_data_path