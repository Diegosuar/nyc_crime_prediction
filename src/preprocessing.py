# src/preprocessing.py (VERSIÓN FINAL COMPLETA - Violento vs. No Violento + Análisis Completo)

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
    4. Calcula análisis de lugares y distritos para el dashboard.
    5. Aplica SMOTE para balancear las clases.
    6. Escala los datos y los retorna listos para el entrenamiento.
    """
    print("Iniciando preprocesamiento AVANZADO para predicción Violento vs. No Violento...")

    # --- 1. Estandarizar y Preparar DataFrame de Denuncias ---
    if not complaints_df.empty:
        complaints_df.columns = complaints_df.columns.str.lower()
    else:
        raise ValueError("El DataFrame de denuncias está vacío o no se cargó correctamente.")

    # Convertir fecha y verificar columnas necesarias
    complaints_df['cmplnt_fr_dt'] = pd.to_datetime(complaints_df['cmplnt_fr_dt'], errors='coerce')
    required_cols = ['cmplnt_fr_dt', 'latitude', 'longitude', 'prem_typ_desc', 'ofns_desc', 'law_cat_cd', 'boro_nm'] # Añadido boro_nm
    complaints_df.dropna(subset=required_cols, inplace=True)
    complaints_df = complaints_df[complaints_df['cmplnt_fr_dt'].dt.year >= 2020].copy()
    print(" -> DataFrame de denuncias preparado.")

    # --- 2. Ingeniería de Características ---
    df_processed = complaints_df.copy()

    # Características cíclicas y de contexto
    df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['cmplnt_fr_dt'].dt.hour/24.0)
    df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['cmplnt_fr_dt'].dt.hour/24.0)
    df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['cmplnt_fr_dt'].dt.month/12.0)
    df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['cmplnt_fr_dt'].dt.month/12.0)
    df_processed['is_weekend'] = df_processed['cmplnt_fr_dt'].dt.dayofweek.isin([5, 6]).astype(int)

    # Clustering Geoespacial
    print(" -> Creando clusters geoespaciales...")
    kmeans = KMeans(n_clusters=50, random_state=42, n_init='auto')
    df_processed['crime_cluster'] = kmeans.fit_predict(df_processed[['latitude', 'longitude']])
    print(" -> Característica 'crime_cluster' creada.")

    # --- 3. Crear Variable Objetivo Binaria: 'is_violent' ---
    print(" -> Creando variable objetivo binaria 'is_violent' (basada en 'law_cat_cd')...")
    df_processed['is_violent'] = df_processed['law_cat_cd'].apply(lambda x: 1 if x == 'FELONY' else 0)
    print(f" -> Proporción de crímenes violentos (Felony): {df_processed['is_violent'].mean():.2%}")

    # --- 4. Calcular Análisis para el Dashboard (Lugares y Distritos) ---
    # Análisis de Lugares
    print(" -> Realizando análisis de lugares...")
    location_crime_counts = df_processed['prem_typ_desc'].value_counts()
    location_analysis = {
        "most_dangerous": location_crime_counts.nlargest(5).to_dict(),
        "least_dangerous": location_crime_counts.nsmallest(5).to_dict()
    }

    # Análisis de Distritos (Boroughs)
    print(" -> Realizando análisis por distrito (Borough)...")
    if 'boro_nm' in df_processed.columns:
        borough_crime_counts = df_processed['boro_nm'].value_counts()
        total_crimes = borough_crime_counts.sum()
        borough_percentage = (borough_crime_counts / total_crimes * 100).round(2) if total_crimes > 0 else borough_crime_counts * 0

        most_dangerous_boroughs = {
            borough: f"{count} ({percentage}%)"
            for borough, count, percentage in zip(borough_crime_counts.nlargest(5).index,
                                                  borough_crime_counts.nlargest(5).values,
                                                  borough_percentage.loc[borough_crime_counts.nlargest(5).index])
        }
        least_dangerous_boroughs = {
            borough: f"{count} ({percentage}%)"
            for borough, count, percentage in zip(borough_crime_counts.nsmallest(5).index,
                                                  borough_crime_counts.nsmallest(5).values,
                                                  borough_percentage.loc[borough_crime_counts.nsmallest(5).index])
        }
        borough_analysis = {
            "most_dangerous": most_dangerous_boroughs,
            "least_dangerous": least_dangerous_boroughs
        }
    else:
        print(" -> ADVERTENCIA: Columna 'boro_nm' no encontrada. No se pudo realizar análisis por distrito.")
        borough_analysis = {} # Devolver vacío si falta la columna

    # --- 5. Preparación Final para el Modelo ---
    features = ['latitude', 'longitude', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_weekend']
    # El tipo de crimen original, cluster y tipo de lugar son características categóricas
    categorical_features = ['prem_typ_desc', 'crime_cluster', 'ofns_desc']

    df_final = df_processed[features + categorical_features + ['is_violent']].copy()

    # Filtrar solo tipos de crimen comunes para evitar demasiadas columnas dummy
    common_crimes = df_final['ofns_desc'].value_counts()
    common_crimes = common_crimes[common_crimes > 10].index
    df_final = df_final[df_final['ofns_desc'].isin(common_crimes)]

    df_final = pd.get_dummies(df_final, columns=categorical_features, prefix='cat', drop_first=True)
    print(" -> Ingeniería de características final completada.")

    X = df_final.drop(columns=['is_violent'])
    y = df_final['is_violent']

    le = LabelEncoder().fit(['No Violento', 'Violento'])
    dump(le, 'models/label_encoder.joblib')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- 6. Balanceo de Clases con SMOTE ---
    print(f" -> Antes de SMOTE, tamaño de X_train: {X_train.shape}")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f" -> Después de SMOTE, tamaño de X_train: {X_train.shape}")

    # --- 7. Escalado de Datos ---
    numerical_cols = ['latitude', 'longitude', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    scaler = StandardScaler()

    # Asegurarnos de escalar solo las columnas que existen después de get_dummies
    cols_to_scale = [col for col in numerical_cols if col in X_train.columns]
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    # Usamos .loc para evitar SettingWithCopyWarning
    X_test.loc[:, cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    dump(scaler, 'models/scaler.joblib')
    print(" -> Escalado de datos completado.")

    # Retornamos todos los resultados necesarios
    return X_train, X_test, y_train, y_test, le, location_analysis, borough_analysis