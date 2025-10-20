# src/preprocessing.py (VERSIÓN FINAL PARA VIOLENTO vs. NO VIOLENTO)

import pandas as pd
import numpy as np
from prefect import task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from joblib import dump
from imblearn.over_sampling import SMOTE

@task
def preprocess_and_fuse_data(complaints_df: pd.DataFrame, arrests_df: pd.DataFrame, vehicle_stops_df: pd.DataFrame):
    """
    Implementa un pipeline avanzado para predecir si un crimen es Violento vs. No Violento:
    1. Limpia y prepara el dataset de denuncias.
    2. Crea características geoespaciales (clusters) y de contexto.
    3. Define la variable objetivo binaria 'is_violent'.
    4. Aplica SMOTE para balancear las clases.
    """
    print("Iniciando preprocesamiento AVANZADO para predicción Violento vs. No Violento...")

    # --- 1. Estandarizar y Preparar DataFrame de Denuncias ---
    if not complaints_df.empty:
        complaints_df.columns = complaints_df.columns.str.lower()
    else:
        # Si no hay datos de denuncias, no podemos continuar
        raise ValueError("El DataFrame de denuncias está vacío o no se cargó correctamente.")
        
    complaints_df['cmplnt_fr_dt'] = pd.to_datetime(complaints_df['cmplnt_fr_dt'], errors='coerce')
    # Nos aseguramos de tener las columnas necesarias para características y objetivo
    required_cols = ['cmplnt_fr_dt', 'latitude', 'longitude', 'prem_typ_desc', 'ofns_desc', 'law_cat_cd']
    complaints_df.dropna(subset=required_cols, inplace=True)
    complaints_df = complaints_df[complaints_df['cmplnt_fr_dt'].dt.year >= 2020].copy()
    print(" -> DataFrame de denuncias preparado.")

    # --- 2. Ingeniería de Características ---
    df_processed = complaints_df.copy() # Usamos un nuevo nombre

    # Características cíclicas y de contexto
    df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['cmplnt_fr_dt'].dt.hour/24.0)
    df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['cmplnt_fr_dt'].dt.hour/24.0)
    df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['cmplnt_fr_dt'].dt.month/12.0)
    df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['cmplnt_fr_dt'].dt.month/12.0)
    df_processed['is_weekend'] = df_processed['cmplnt_fr_dt'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Clustering Geoespacial
    print(" -> Creando clusters geoespaciales...")
    kmeans = KMeans(n_clusters=50, random_state=42, n_init='auto') # Usamos n_init='auto'
    df_processed['crime_cluster'] = kmeans.fit_predict(df_processed[['latitude', 'longitude']])
    
    # --- 3. Crear Variable Objetivo Binaria: 'is_violent' ---
    # Asumimos que 'FELONY' corresponde a crímenes más violentos/graves
    print(" -> Creando variable objetivo binaria 'is_violent' (basada en 'law_cat_cd')...")
    df_processed['is_violent'] = df_processed['law_cat_cd'].apply(lambda x: 1 if x == 'FELONY' else 0)
    print(f" -> Proporción de crímenes violentos (Felony): {df_processed['is_violent'].mean():.2%}")

    # Análisis de lugares (se mantiene para el dashboard)
    print(" -> Realizando análisis de lugares...")
    location_crime_counts = df_processed['prem_typ_desc'].value_counts()
    location_analysis = {
        "most_dangerous": location_crime_counts.nlargest(5).to_dict(),
        "least_dangerous": location_crime_counts.nsmallest(5).to_dict() # <-- ¡Asegúrate que esta línea esté!
    }

    # --- 4. Preparación Final para el Modelo ---
    features = ['latitude', 'longitude', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_weekend']
    # El tipo de crimen original y el cluster ahora son características
    categorical_features = ['prem_typ_desc', 'crime_cluster', 'ofns_desc'] 
    
    df_final = df_processed[features + categorical_features + ['is_violent']].copy()
    
    # Aplicamos get_dummies a todas las categóricas
    df_final = pd.get_dummies(df_final, columns=categorical_features, prefix='cat', drop_first=True)
    print(" -> Ingeniería de características final completada.")

    X = df_final.drop(columns=['is_violent'])
    y = df_final['is_violent']

    le = LabelEncoder().fit(['No Violento', 'Violento'])
    dump(le, 'models/label_encoder.joblib') # Guardamos el encoder para la app

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f" -> Antes de SMOTE, tamaño de X_train: {X_train.shape}")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f" -> Después de SMOTE, tamaño de X_train: {X_train.shape}")
    
    numerical_cols = ['latitude', 'longitude', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    dump(scaler, 'models/scaler.joblib')
    
    return X_train, X_test, y_train, y_test, le, location_analysis