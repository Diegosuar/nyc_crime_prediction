# app/app.py (VERSIÓN FINAL COMPLETA - Predicción Violento vs. No Violento + Dashboard Completo)

# --- 1. Importaciones ---
from flask import Flask, render_template, request # Importaciones clave de Flask
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
import json # Para cargar las métricas

# --- 2. Creación de la Aplicación Flask ---
# Le decimos a Flask dónde encontrar las plantillas y archivos estáticos
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- 3. Definir Variables Globales con Valores por Defecto ---
# Esto evita NameErrors si la carga falla
model, scaler, label_encoder, df, model_columns = [None] * 5
metrics = {}            # Diccionario vacío por defecto para métricas del modelo
location_analysis = {}  # Diccionario vacío por defecto para análisis de lugares
borough_analysis = {}   # Diccionario vacío por defecto para análisis de distritos
plot_paths = {}         # Diccionario vacío por defecto para rutas de gráficos

# --- 4. Carga de Archivos (se ejecuta una sola vez al iniciar) ---
try:
    print("Iniciando carga de modelos, datos y métricas...")
    model = load('models/crime_predictor_model.joblib')
    scaler = load('models/scaler.joblib')
    label_encoder = load('models/label_encoder.joblib')
    # Guardamos los nombres de las columnas que el modelo espera
    model_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None

    # Cargamos los datos crudos de denuncias para visualizaciones
    df = pd.read_csv('data/raw/complaints.csv', low_memory=False)
    df.columns = df.columns.str.lower() # Estandarizar columnas

    # Cargamos las métricas y rutas de gráficos generadas por el pipeline
    with open('reports/dashboard_metrics.json', 'r') as f:
        metrics_data = json.load(f)
        # Extraemos las partes relevantes y las asignamos a las variables globales
        metrics = metrics_data.get("model_performance", {})
        location_analysis = metrics_data.get("location_analysis", {})
        borough_analysis = metrics_data.get("borough_analysis", {}) # Cargar análisis de distritos
        plot_paths = metrics_data.get("plot_paths", {})

    print("✅ Modelo, datos, métricas y análisis cargados exitosamente.")
except FileNotFoundError as e:
    print(f"❌ Error Crítico: No se encontró el archivo {e.filename}. ")
    print("   Asegúrate de haber ejecutado el pipeline (`python -m src.pipeline`) exitosamente primero.")
except Exception as e:
    print(f"❌ Error inesperado al cargar archivos: {e}")
    print("   La aplicación podría no funcionar correctamente.")

# --- 5. Definición de la Función para Generar Gráficos del Dashboard ---
def generate_plots():
    """Genera y guarda los gráficos estáticos para el dashboard principal."""
    if df is None or df.empty:
        print(" -> ADVERTENCIA: No se pueden generar gráficos, DataFrame 'df' vacío o no cargado.")
        return

    try:
        print(" -> Generando gráficos del dashboard...")
        img_dir = os.path.join('app', 'static', 'img')
        os.makedirs(img_dir, exist_ok=True)

        # Gráfico 1: Distribución de Crímenes (Top 10 Originales)
        if 'ofns_desc' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(
                y='ofns_desc',
                data=df,
                order=df['ofns_desc'].value_counts().nlargest(10).index,
                palette='viridis',
                hue='ofns_desc', # Para evitar FutureWarning
                legend=False
            )
            plt.title('Top 10 Crímenes (Tipos Originales)')
            plt.xlabel('Número de Incidentes')
            plt.ylabel('Tipo de Crimen')
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'crime_distribution.png'))
            plt.close()
        else:
             print(" -> ADVERTENCIA: Columna 'ofns_desc' no encontrada para gráfico de distribución.")


        # Gráfico 2: Crímenes por Día de la Semana
        if 'cmplnt_fr_dt' in df.columns:
             # Crear una columna temporal para evitar modificar df global innecesariamente
            df_temp = df.copy()
            df_temp['cmplnt_fr_dt_graph'] = pd.to_datetime(df_temp['cmplnt_fr_dt'], errors='coerce')
            df_temp.dropna(subset=['cmplnt_fr_dt_graph'], inplace=True)
            df_temp['day_of_week_graph'] = df_temp['cmplnt_fr_dt_graph'].dt.day_name()
            plt.figure(figsize=(10, 6))
            sns.countplot(
                x='day_of_week_graph',
                data=df_temp, # Usar el DataFrame temporal
                order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                palette='plasma',
                hue='day_of_week_graph', # Para evitar FutureWarning
                legend=False
            )
            plt.title('Incidentes por Día de la Semana')
            plt.xlabel('Día')
            plt.ylabel('Número de Incidentes')
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'crimes_by_day.png'))
            plt.close()
            print("✅ Gráficos del dashboard generados y guardados correctamente.")
        else:
            print(" -> ADVERTENCIA: Columna 'cmplnt_fr_dt' no encontrada para gráfico por día.")

    except Exception as e:
        print(f"❌ Error al generar gráficos del dashboard: {e}")

# --- 6. Llamada a la Función (después de definirla) ---
generate_plots()

# --- 7. Rutas de la Aplicación ---
@app.route('/')
def home():
    """Muestra el dashboard principal con gráficos y métricas."""
    # Pasamos todas las variables globales a la plantilla
    return render_template('index.html',
                           metrics=metrics,
                           location_analysis=location_analysis,
                           borough_analysis=borough_analysis, # Añadido
                           plot_paths=plot_paths)

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    """Maneja el formulario y la predicción de crimen violento."""
    predictions = None
    input_values = request.form.to_dict() if request.method == 'POST' else {} # Guarda los valores para rellenar el form

    if request.method == 'POST':
        try:
            # Validaciones cruciales
            if model is None or scaler is None or label_encoder is None or model_columns is None:
                 raise ValueError("El modelo o sus componentes no se cargaron. Ejecuta el pipeline primero.")

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
            prem_typ_col = f"cat_{form_data['prem_typ_desc']}"
            if prem_typ_col in input_df.columns:
                input_df[prem_typ_col] = 1

            ofns_desc_col = f"cat_{form_data['ofns_desc']}"
            if ofns_desc_col in input_df.columns:
                input_df[ofns_desc_col] = 1

            # NOTA: Ignoramos 'crime_cluster' en la app por simplicidad.

            # Escalar los datos numéricos correctos
            numerical_cols_to_scale = ['latitude', 'longitude', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
            valid_cols_to_scale = [col for col in numerical_cols_to_scale if col in input_df.columns]
            input_df.loc[:, valid_cols_to_scale] = scaler.transform(input_df[valid_cols_to_scale])

            # Asegurar el orden de columnas antes de predecir
            input_df = input_df[model_columns]

            # Predecir probabilidades (0=No Violento, 1=Violento)
            proba_violent = model.predict_proba(input_df)[0][1] # Probabilidad de la clase '1'
            predicted_class_index = np.argmax(model.predict_proba(input_df)[0])
            predicted_class_label = label_encoder.classes_[predicted_class_index]

            predictions = [
                {"label": "Probabilidad de ser Violento (Felony)", "probability": round(proba_violent * 100, 2)},
                {"label": "Clase Predominante", "class": predicted_class_label}
            ]

        except KeyError as e:
             error_msg = f"Error: Característica faltante o inválida ({e}). Asegúrate de que todas las entradas sean válidas."
             print(f"KeyError en predicción: {e}")
             predictions = [{"label": error_msg, "probability": 0}]
        except ValueError as e:
             error_msg = f"Error en los datos de entrada: {e}. Revisa los valores numéricos."
             print(f"ValueError en predicción: {e}")
             predictions = [{"label": error_msg, "probability": 0}]
        except Exception as e:
            error_msg = f"Error inesperado durante la predicción: {e}"
            print(f"Error inesperado en predicción: {e}")
            predictions = [{"label": error_msg, "probability": 0}]

    # Preparamos la lista de crímenes comunes para el dropdown
    crime_types = []
    if df is not None and 'ofns_desc' in df.columns:
         try:
             common_crimes_list = df['ofns_desc'].value_counts()
             common_crimes_list = common_crimes_list[common_crimes_list > 10].index.tolist()
             crime_types = sorted(common_crimes_list)
         except Exception as e:
             print(f"Error al obtener lista de crímenes para el formulario: {e}")
             crime_types = ['Error al cargar tipos']

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

            m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles='CartoDB positron')
            heat_data = [[row['latitude'], row['longitude']] for index, row in map_sample.iterrows()]
            HeatMap(heat_data, radius=12, blur=15).add_to(m)
            folium_map = m._repr_html_()
        else:
            print(" -> ADVERTENCIA: No se puede generar mapa, DataFrame 'df' vacío o sin columnas de coordenadas.")
    except Exception as e:
        print(f"❌ Error al generar el mapa: {e}")

    return render_template('map.html', folium_map=folium_map)

# --- 8. Punto de Entrada para Ejecución ---
if __name__ == '__main__':
    # debug=True es útil para desarrollo, recuerda quitarlo para producción
    app.run(debug=True, port=5000)