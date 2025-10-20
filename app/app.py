# app/app.py (VERSIÓN FINAL, COMPLETA Y ORDENADA)

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from joblib import load
import os
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import json

app = Flask(__name__, template_folder='templates', static_folder='static')

# --- 1. Carga de Archivos (se ejecuta una sola vez al iniciar) ---
model, scaler, label_encoder, df, model_columns, metrics = [None] * 6
try:
    model = load('models/crime_predictor_model.joblib')
    scaler = load('models/scaler.joblib')
    label_encoder = load('models/label_encoder.joblib')
    model_columns = model.feature_names_in_
    df = pd.read_csv('data/raw/complaints.csv', low_memory=False)
    df.columns = df.columns.str.lower()
    
    with open('reports/dashboard_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    print("✅ Modelo, datos y métricas cargados exitosamente.")
except Exception as e:
    print(f"❌ Error al cargar archivos: {e}")

# --- 2. Definición de la Función para Generar Gráficos ---
def generate_plots():
    if df is None or df.empty:
        print(" -> No se pueden generar gráficos, DataFrame vacío.")
        return

    img_dir = os.path.join('app', 'static', 'img')
    os.makedirs(img_dir, exist_ok=True)

    # Gráfico 1: Distribución de Crímenes
    plt.figure(figsize=(10, 6))
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

    # Gráfico 2: Crímenes por Día de la Semana
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

# --- 3. Llamada a la Función (ahora que ya está definida) ---
generate_plots()

# --- 4. Rutas de la Aplicación ---
@app.route('/')
def home():
    return render_template('index.html', metrics=metrics)

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    predictions = None
    input_values = request.form.to_dict() if request.method == 'POST' else {}

    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            input_df = pd.DataFrame(columns=model_columns)
            input_df.loc[0] = 0
            
            input_df['latitude'] = float(form_data['latitude'])
            input_df['longitude'] = float(form_data['longitude'])
            input_df['hour'] = int(form_data['hour'])
            input_df['day_of_week'] = int(form_data['day_of_week'])
            input_df['month'] = int(form_data['month'])
            
            numerical_cols = scaler.feature_names_in_
            input_df.loc[:, numerical_cols] = scaler.transform(input_df[numerical_cols])

            probabilities = model.predict_proba(input_df)[0]
            top3_indices = np.argsort(probabilities)[-3:][::-1]
            predictions = [{"crime": label_encoder.classes_[i], "probability": round(probabilities[i] * 100, 2)} for i in top3_indices]
        except Exception as e:
            predictions = [{"crime": f"Error en la predicción: {e}", "probability": 0}]

    return render_template('forecast.html', predictions=predictions, input_values=input_values)

@app.route('/map')
def map_view():
    folium_map = None
    if df is not None and not df.empty:
        df_clean = df.dropna(subset=['latitude', 'longitude'])
        sample_size = min(10000, len(df_clean))
        map_sample = df_clean.sample(n=sample_size, random_state=42)
        
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
        heat_data = [[row['latitude'], row['longitude']] for index, row in map_sample.iterrows()]
        HeatMap(heat_data, radius=12).add_to(m)
        folium_map = m._repr_html_()

    return render_template('map.html', folium_map=folium_map)

# --- 5. Punto de Entrada para Ejecución ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)