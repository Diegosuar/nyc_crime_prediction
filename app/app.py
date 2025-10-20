# app/app.py (VERSIÓN FINAL, COMPLETA Y ORDENADA)

# --- 1. Importaciones ---
from flask import Flask, render_template, request # Asegúrate que Flask y otras funciones necesarias estén importadas
import matplotlib
matplotlib.use('Agg') # Configura el backend ANTES de importar pyplot
import pandas as pd
import numpy as np
from joblib import load
import os
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import json

# --- 2. Creación de la Aplicación Flask ---
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- 3. Carga de Archivos (se ejecuta una sola vez al iniciar) ---
model, scaler, label_encoder, df, model_columns, metrics = [None] * 6
try:
    model = load('models/crime_predictor_model.joblib')
    scaler = load('models/scaler.joblib')
    label_encoder = load('models/label_encoder.joblib')
    # Guardamos los nombres de las columnas que el modelo espera para usarlos en la predicción
    model_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None

    df = pd.read_csv('data/raw/complaints.csv', low_memory=False)
    df.columns = df.columns.str.lower() # Estandarizar columnas

    # Cargar las métricas del dashboard generadas por el pipeline
    with open('reports/dashboard_metrics.json', 'r') as f:
        metrics = json.load(f)

    print("✅ Modelo, datos y métricas cargados exitosamente.")
except FileNotFoundError as e:
    print(f"❌ Error: No se encontró el archivo {e.filename}. Asegúrate de ejecutar el pipeline (`python -m src.pipeline`) primero.")
except Exception as e:
    print(f"❌ Error inesperado al cargar archivos: {e}")

# --- 4. Definición de la Función para Generar Gráficos ---
def generate_plots():
    """Genera y guarda los gráficos para el dashboard."""
    if df is None or df.empty:
        print(" -> ADVERTENCIA: No se pueden generar gráficos, DataFrame 'df' vacío o no cargado.")
        return

    try:
        img_dir = os.path.join('app', 'static', 'img')
        os.makedirs(img_dir, exist_ok=True)

        # Gráfico 1: Distribución de Crímenes
        plt.figure(figsize=(10, 6))
        # Asegúrate que 'ofns_desc' exista antes de usarlo
        if 'ofns_desc' in df.columns:
            sns.countplot(
                y='ofns_desc',
                data=df,
                order=df['ofns_desc'].value_counts().nlargest(10).index,
                palette='viridis',
                hue='ofns_desc',
                legend=False
            )
            plt.title('Top 10 Crímenes Más Comunes')
            plt.xlabel('Número de Incidentes')
            plt.ylabel('Tipo de Crimen')
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'crime_distribution.png'))
            plt.close()
        else:
             print(" -> ADVERTENCIA: Columna 'ofns_desc' no encontrada para el gráfico de distribución.")


        # Gráfico 2: Crímenes por Día de la Semana
        if 'cmplnt_fr_dt' in df.columns:
            df['cmplnt_fr_dt'] = pd.to_datetime(df['cmplnt_fr_dt'], errors='coerce')
            df.dropna(subset=['cmplnt_fr_dt'], inplace=True)
            df['day_of_week'] = df['cmplnt_fr_dt'].dt.day_name()
            plt.figure(figsize=(10, 6))
            sns.countplot(
                x='day_of_week',
                data=df,
                order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                palette='plasma',
                hue='day_of_week',
                legend=False
            )
            plt.title('Incidentes por Día de la Semana')
            plt.xlabel('Día')
            plt.ylabel('Número de Incidentes')
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'crimes_by_day.png'))
            plt.close()
            print("✅ Gráficos generados y guardados correctamente.")
        else:
            print(" -> ADVERTENCIA: Columna 'cmplnt_fr_dt' no encontrada para el gráfico por día.")

    except Exception as e:
        print(f"❌ Error al generar gráficos: {e}")

# --- 5. Llamada a la Función (¡Ahora está después de la definición!) ---
generate_plots()

# --- 6. Rutas de la Aplicación ---
@app.route('/')
def home():
    """Muestra el dashboard principal con gráficos y métricas."""
    return render_template('index.html', metrics=metrics)

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    """Maneja el formulario y la predicción de arresto."""
    predictions = None
    input_values = request.form.to_dict() if request.method == 'POST' else {} # Guarda los valores para rellenar el form

    if request.method == 'POST':
        try:
            # Validaciones cruciales antes de predecir
            if model is None or scaler is None or label_encoder is None or model_columns is None:
                 raise ValueError("El modelo o sus componentes no se cargaron correctamente.")

            form_data = request.form.to_dict()
            input_df = pd.DataFrame(columns=model_columns)
            input_df.loc[0] = 0 # Inicializar fila con ceros

            # Llenar datos numéricos y cíclicos
            input_df['latitude'] = float(form_data['latitude'])
            input_df['longitude'] = float(form_data['longitude'])
            hour = int(form_data['hour'])
            month = int(form_data['month'])
            input_df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
            input_df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
            input_df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
            input_df['month_cos'] = np.cos(2 * np.pi * month / 12.0)
            input_df['is_weekend'] = int(form_data['is_weekend'])

            # Manejar las columnas categóricas (one-hot encoded)
            # Asegúrate que los prefijos coincidan con los usados en preprocessing.py ('cat_')
            prem_typ_col = f"cat_{form_data['prem_typ_desc']}"
            if prem_typ_col in input_df.columns:
                input_df[prem_typ_col] = 1

            ofns_desc_col = f"cat_{form_data['ofns_desc']}"
            if ofns_desc_col in input_df.columns:
                input_df[ofns_desc_col] = 1
                
            # NOTA: Ignoramos 'crime_cluster' y actividad policial ('monthly_*') en la app por simplicidad

            # Escalar los datos numéricos correctos
            numerical_cols_to_scale = ['latitude', 'longitude', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
            valid_cols_to_scale = [col for col in numerical_cols_to_scale if col in input_df.columns]
            input_df[valid_cols_to_scale] = scaler.transform(input_df[valid_cols_to_scale])

            # Asegurar el orden de columnas antes de predecir
            input_df = input_df[model_columns]

            # Predecir probabilidades (asumiendo clasificación binaria: 0=No Arresto, 1=Arresto)
            proba_arrest = model.predict_proba(input_df)[0][1]
            predictions = [{"label": "Probabilidad de Arresto", "probability": round(proba_arrest * 100, 2)}]

        except KeyError as e:
             print(f"Error de clave durante la predicción: Falta la columna {e} en el DataFrame de entrada o en el modelo.")
             predictions = [{"label": f"Error: Dato faltante o inválido ({e})", "probability": 0}]
        except ValueError as e:
             print(f"Error de valor durante la predicción: {e}")
             predictions = [{"label": f"Error: {e}", "probability": 0}]
        except Exception as e:
            print(f"Error inesperado durante la predicción: {e}")
            predictions = [{"label": f"Error inesperado: {e}", "probability": 0}]

    # Preparamos la lista de crímenes para el dropdown
    crime_types = []
    if df is not None and 'ofns_desc' in df.columns:
         try:
             # Usamos los crímenes comunes definidos en el preprocesamiento si es posible
             common_crimes_list = df['ofns_desc'].value_counts()
             common_crimes_list = common_crimes_list[common_crimes_list > 10].index.tolist()
             crime_types = sorted(common_crimes_list)
         except Exception as e:
             print(f"Error al obtener lista de crímenes: {e}")
             crime_types = ['Error al cargar tipos'] # Valor por defecto

    return render_template('forecast.html', predictions=predictions, input_values=input_values, crime_types=crime_types)

@app.route('/map')
def map_view():
    """Muestra el mapa de calor de densidad de crímenes."""
    folium_map = None
    try:
        if df is not None and not df.empty and 'latitude' in df.columns and 'longitude' in df.columns:
            df_clean = df.dropna(subset=['latitude', 'longitude'])
            sample_size = min(10000, len(df_clean)) # Tamaño de muestra seguro
            map_sample = df_clean.sample(n=sample_size, random_state=42)

            m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles='CartoDB positron') # Usamos un tile más limpio
            heat_data = [[row['latitude'], row['longitude']] for index, row in map_sample.iterrows()]
            HeatMap(heat_data, radius=12, blur=15).add_to(m)
            folium_map = m._repr_html_()
        else:
            print(" -> ADVERTENCIA: No se puede generar mapa, DataFrame 'df' vacío o sin columnas de coordenadas.")
    except Exception as e:
        print(f"❌ Error al generar el mapa: {e}")

    return render_template('map.html', folium_map=folium_map)

# --- 7. Punto de Entrada para Ejecución ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)