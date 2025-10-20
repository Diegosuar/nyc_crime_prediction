# src/preprocessing.py (VERSIÓN FINAL CON TODAS LAS MEJORAS)

import pandas as pd
import numpy as np
from prefect import task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import dump
from imblearn.over_sampling import SMOTE

@task
def preprocess_and_fuse_data(complaints_df: pd.DataFrame, arrests_df: pd.DataFrame, vehicle_stops_df: pd.DataFrame):
    """
    Limpia, agrega actividad policial, crea características avanzadas, fusiona los datos,
    aplica SMOTE para balancear clases y prepara los datos para el entrenamiento.
    """
    print("Iniciando preprocesamiento y fusión de datos...")

    # --- 1. Estandarizar Nombres de Columnas ---
    for df in [complaints_df, arrests_df, vehicle_stops_df]:
        if not df.empty:
            df.columns = df.columns.str.lower()
    print(" -> Nombres de columnas estandarizados.")

    # --- 2. Preparar cada DataFrame ---
    # Denuncias
    complaints_df['cmplnt_fr_dt'] = pd.to_datetime(complaints_df['cmplnt_fr_dt'], errors='coerce')
    complaints_df.dropna(subset=['cmplnt_fr_dt', 'addr_pct_cd'], inplace=True)
    complaints_df['year_month'] = complaints_df['cmplnt_fr_dt'].dt.to_period('M')
    complaints_df['precinct'] = complaints_df['addr_pct_cd'].astype(int)

    # Arrestos
    arrests_df['arrest_date'] = pd.to_datetime(arrests_df['arrest_date'], errors='coerce')
    arrests_df.dropna(subset=['arrest_date', 'arrest_precinct'], inplace=True)
    arrests_df['year_month'] = arrests_df['arrest_date'].dt.to_period('M')
    arrests_df['precinct'] = arrests_df['arrest_precinct'].astype(int)

    # Paradas de Vehículos (con manejo de fallos)
    stops_agg = pd.DataFrame()
    if not vehicle_stops_df.empty:
        print(" -> Procesando datos de paradas de vehículos...")
        vehicle_stops_df['stop_frisk_date'] = pd.to_datetime(vehicle_stops_df['stop_frisk_date'], errors='coerce')
        vehicle_stops_df.dropna(subset=['stop_frisk_date', 'precinct'], inplace=True)
        vehicle_stops_df['year_month'] = vehicle_stops_df['stop_frisk_date'].dt.to_period('M')
        vehicle_stops_df['precinct'] = vehicle_stops_df['precinct'].astype(int)
        stops_agg = vehicle_stops_df.groupby(['precinct', 'year_month']).size().reset_index(name='monthly_stops')
    else:
        print(" -> ADVERTENCIA: No se procesaron datos de paradas de vehículos (DataFrame vacío).")
    
    print(" -> DataFrames individuales preparados.")

    # --- 3. Agregar y Fusionar ---
    arrests_agg = arrests_df.groupby(['precinct', 'year_month']).size().reset_index(name='monthly_arrests')
    df_merged = pd.merge(complaints_df, arrests_agg, on=['precinct', 'year_month'], how='left')
    if not stops_agg.empty:
        df_merged = pd.merge(df_merged, stops_agg, on=['precinct', 'year_month'], how='left')
        df_merged['monthly_stops'] = df_merged['monthly_stops'].fillna(0)
    else:
        df_merged['monthly_stops'] = 0
    df_merged['monthly_arrests'] = df_merged['monthly_arrests'].fillna(0)
    print(" -> Datasets fusionados.")

    # --- 4. Ingeniería de Características (Feature Engineering) ---
    df_merged.dropna(subset=['latitude', 'longitude', 'prem_typ_desc', 'ofns_desc'], inplace=True)
    df_merged = df_merged[df_merged['cmplnt_fr_dt'].dt.year >= 2020].copy()

    # Características cíclicas y de contexto
    df_merged['hour_sin'] = np.sin(2 * np.pi * df_merged['cmplnt_fr_dt'].dt.hour/24.0)
    df_merged['hour_cos'] = np.cos(2 * np.pi * df_merged['cmplnt_fr_dt'].dt.hour/24.0)
    df_merged['month_sin'] = np.sin(2 * np.pi * df_merged['cmplnt_fr_dt'].dt.month/12.0)
    df_merged['month_cos'] = np.cos(2 * np.pi * df_merged['cmplnt_fr_dt'].dt.month/12.0)
    df_merged['is_weekend'] = df_merged['cmplnt_fr_dt'].dt.dayofweek.isin([5, 6]).astype(int)
    df_merged['is_night'] = ((df_merged['cmplnt_fr_dt'].dt.hour >= 22) | (df_merged['cmplnt_fr_dt'].dt.hour <= 6)).astype(int)
    
    top_crimes = df_merged['ofns_desc'].value_counts().nlargest(10).index
    df_final = df_merged[df_merged['ofns_desc'].isin(top_crimes)].copy()

    # --- 5. Simplificación de Clases (Reagrupación de Crímenes) ---
    print(" -> Reagrupando crímenes en categorías más amplias...")
    crime_mapping = {
        'PETIT LARCENY': 'ROBO',
        'GRAND LARCENY': 'ROBO',
        'GRAND LARCENY OF MOTOR VEHICLE': 'ROBO DE VEHÍCULO',
        'ROBBERY': 'ROBO',
        'BURGLARY': 'ROBO',
        'HARRASSMENT 2': 'ACOSO Y OFENSAS',
        'ASSAULT 3 & RELATED OFFENSES': 'ASALTO',
        'FELONY ASSAULT': 'ASALTO',
        'CRIMINAL MISCHIEF & RELATED OF': 'DAÑO A LA PROPIEDAD',
        'OFF. AGNST PUB ORD SENSBLTY &': 'OFENSAS PÚBLICAS'
        # Nota: 'VEHICLE AND TRAFFIC LAWS' se omite o puedes asignarlo a 'OFENSAS PÚBLICAS'
    }
    df_final['crime_category'] = df_final['ofns_desc'].map(crime_mapping)
    df_final.dropna(subset=['crime_category'], inplace=True) # Elimina filas que no se mapearon

    # Análisis de lugares (antes de eliminar la columna original)
    print(" -> Realizando análisis de lugares...")
    location_crime_counts = df_final['prem_typ_desc'].value_counts()
    location_analysis = {
        "most_dangerous": location_crime_counts.nlargest(5).to_dict(),
        "least_dangerous": location_crime_counts.nsmallest(5).to_dict()
    }

    # Definimos la lista final de características a usar
    features = ['latitude', 'longitude', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                'is_weekend', 'is_night', 'monthly_arrests', 'monthly_stops']
    categorical_features = ['prem_typ_desc']
    df_final_processed = df_final[features + categorical_features + ['crime_category']].copy()
    
    df_final_processed = pd.get_dummies(df_final_processed, columns=categorical_features, prefix='prem_typ', drop_first=True)
    print(" -> Ingeniería de características final completada.")

    # --- 6. Preparación para el Modelo (usando 'crime_category') ---
    X = df_final_processed.drop(columns=['crime_category'])
    y_raw = df_final_processed['crime_category']

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    dump(le, 'models/label_encoder.joblib')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- 7. Balanceo de Clases con SMOTE ---
    print(f" -> Antes de SMOTE, tamaño de X_train: {X_train.shape}")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f" -> Después de SMOTE, tamaño de X_train: {X_train.shape}")
    
    # --- 8. Escalado de Datos ---
    numerical_cols = X_train.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    dump(scaler, 'models/scaler.joblib')
    print(" -> Escalado de datos completado.")

    return X_train, X_test, y_train, y_test, le, location_analysis