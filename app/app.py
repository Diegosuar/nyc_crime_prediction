# app/app.py
from flask import Flask, render_template, url_for, request
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')  
import seaborn as sns
import matplotlib.pyplot as plt
import os
import folium
from folium.plugins import HeatMap


# --- Configuración Inicial ---
app = Flask(__name__)
STATIC_FOLDER = 'app/static/img'
os.makedirs(STATIC_FOLDER, exist_ok=True) # Asegura que la carpeta para imágenes exista

# --- Cargar Datos y Modelo ---
try:
    df = pd.read_csv('data/processed/processed_complaints.csv')
    model = joblib.load('models/random_forest_classifier.joblib')
    print("✅ Modelo y datos cargados exitosamente.")
except Exception as e:
    print(f"🛑 Error cargando los archivos: {e}.")
    df = pd.DataFrame()
    model = None

# --- Función para generar gráficos ---
def generate_plots():
    if df.empty:
        return

    # Gráfico 1: Distribución de Tipos de Crimen
    plt.figure(figsize=(8, 5))
    # CORRECCIÓN: Se añade el DataFrame completo con data=df
    sns.countplot(
        data=df, 
        y='law_cat_cd', 
        order=df['law_cat_cd'].value_counts().index, 
        palette='viridis', 
        hue='law_cat_cd', 
        legend=False
    )
    plt.title('Cantidad de Crímenes por Categoría')
    plt.xlabel('Número de Denuncias')
    plt.ylabel('Categoría de Crimen')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_FOLDER, 'crime_distribution.png'))
    plt.close()

    # Gráfico 2: Crímenes por Día de la Semana
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    plt.figure(figsize=(10, 6))
    # CORRECCIÓN: Se añade el DataFrame completo con data=df
    sns.countplot(
        data=df, 
        x='day_of_week', 
        order=days_order, 
        palette='mako', 
        hue='day_of_week', 
        legend=False
    )
    plt.title('Denuncias de Crímenes por Día de la Semana')
    plt.xlabel('Día de la Semana')
    plt.ylabel('Número de Denuncias')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_FOLDER, 'crimes_by_day.png'))
    plt.close()
    
    print("✅ Gráficos generados y guardados correctamente.")

# --- Ruta para el Dashboard Principal ---
@app.route('/')
def dashboard():
    if df.empty or model is None:
        return "<h3>Error: Archivos no encontrados.</h3><p>Ejecuta primero el pipeline de Prefect: <code>python -m src.pipeline</code></p>"

    # Métricas clave
    X = df.drop('law_cat_cd', axis=1)
    y = df['law_cat_cd']
    accuracy = model.score(X, y) * 100
    num_datos = f"{len(df):,}"
    
    # Análisis de Peligrosidad por Lugar
    # Usamos 'prem_typ_desc' como un proxy para "área" o "lugar"
    premise_counts = df['prem_typ_desc'].value_counts()
    most_dangerous = premise_counts.head(5).to_dict()
    least_dangerous = premise_counts.tail(5).to_dict()

    return render_template('dashboard.html',
                           accuracy=f"{accuracy:.2f}%",
                           num_datos=num_datos,
                           most_dangerous=most_dangerous,
                           least_dangerous=least_dangerous,
                           plot1_url=url_for('static', filename='img/crime_distribution.png'),
                           plot2_url=url_for('static', filename='img/crimes_by_day.png'))


BOROUGH_COORDINATES = {
    "MANHATTAN": {"lat": 40.7831, "lon": -73.9712},
    "BROOKLYN": {"lat": 40.6782, "lon": -73.9442},
    "QUEENS": {"lat": 40.7282, "lon": -73.7949},
    "BRONX": {"lat": 40.8448, "lon": -73.8648},
    "STATEN ISLAND": {"lat": 40.5795, "lon": -74.1502}
}

# --- RUTA DE PRONÓSTICO ACTUALIZADA PARA REGRESIÓN ---
@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    premise_types = sorted(df['prem_typ_desc'].unique())
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    borough_names = list(BOROUGH_COORDINATES.keys())
    
    robbery_probability = None
    input_values = {}

    if request.method == 'POST':
        selected_borough = request.form['borough']
        coords = BOROUGH_COORDINATES[selected_borough]

        input_values = {
            'latitude': coords['lat'],
            'longitude': coords['lon'],
            'hour': int(request.form['hour']),
            'day_of_week': request.form['day_of_week'],
            'prem_typ_desc': request.form['premise_type'],
            'borough': selected_borough
        }

        predict_df = pd.DataFrame([{
            'latitude': input_values['latitude'],
            'longitude': input_values['longitude'],
            'hour': input_values['hour'],
            'day_of_week': input_values['day_of_week'],
            'prem_typ_desc': input_values['prem_typ_desc']
        }])

        # --- Predicción de Probabilidad ---
        # model.predict_proba(df) devuelve [[prob_clase_0, prob_clase_1]]
        # Queremos la probabilidad de la clase 1 (es robo)
        prob_is_robbery = model.predict_proba(predict_df)[0][1] * 100
        robbery_probability = f"{prob_is_robbery:.2f}"

    return render_template('forecast.html', 
                           premise_types=premise_types, 
                           days=days_of_week,
                           boroughs=borough_names,
                           probability=robbery_probability,
                           input_values=input_values)


@app.route('/heatmap')
def heatmap():
    if df.empty:
        return "<h3>Error: No se pudieron cargar los datos.</h3><p>Por favor, ejecuta el pipeline de Prefect primero.</p>"

    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="CartoDB positron")

    # --- CORRECCIÓN: Limpiar los datos de coordenadas antes de usarlos ---
    # 1. Crear un DataFrame temporal solo con las coordenadas
    coords_df = df[['latitude', 'longitude']].copy()
    
    # 2. Eliminar cualquier fila que tenga valores nulos en latitud o longitud
    coords_df.dropna(subset=['latitude', 'longitude'], inplace=True)
    
    # 3. Convertir los datos limpios a una lista
    heat_data = coords_df.values.tolist()
    
    # 4. (Opcional pero recomendado) Verificar si la lista no está vacía
    if heat_data:
        # Añadir la capa de mapa de calor solo si hay datos válidos
        HeatMap(heat_data, radius=12, blur=15).add_to(m)
        print(f"✅ Mapa de calor generado con {len(heat_data)} puntos.")
    else:
        print("🛑 Advertencia: No se encontraron coordenadas válidas para generar el mapa de calor.")


    # Cargar la capa de distritos (sin cambios)
    boroughs_geojson_url = 'https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&format=GeoJSON'
    try:
        folium.GeoJson(
            boroughs_geojson_url,
            style_function=lambda feature: {
                'color': '#007bff', 'weight': 1, 'fillOpacity': 0.1,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['boro_name'], aliases=['Distrito:'],
                style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
            )
        ).add_to(m)
        print("✅ Capa de distritos (boroughs) cargada desde la API.")
    except Exception as e:
        print(f"🛑 Error al cargar el GeoJSON desde la URL: {e}")

    map_html = m._repr_html_()
    return render_template('heatmap.html', map_html=map_html)
# --- Generar los gráficos una vez al iniciar la app ---
generate_plots()

