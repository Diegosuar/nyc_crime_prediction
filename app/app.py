# app/app.py
from flask import Flask, render_template, url_for, request
import pandas as pd
import joblib
import json  # <-- Asegúrate de que esto esté importado
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
os.makedirs(STATIC_FOLDER, exist_ok=True) 

# --- Cargar Artefactos del Pipeline ---
try:
    # Cargar los datos procesados por el pipeline
    df = pd.read_csv('data/processed/processed_complaints.csv')
    
    # Cargar el pipeline COMPLETO (preprocesador + modelo)
    # Este único archivo .joblib contiene todo lo que necesitamos
    model = joblib.load('models/random_forest_classifier.joblib')
    
    # --- MEJORA: Cargar métricas desde el pipeline ---
    # Leemos los resultados guardados por evaluate.py
    with open('dashboard_metrics.json', 'r') as f:
        metrics = json.load(f)
        
    print("✅ Modelo, datos y métricas cargados exitosamente.")

except Exception as e:
    print(f"🛑 Error cargando los archivos: {e}.")
    print("🛑 Asegúrate de ejecutar 'python -m src.pipeline' primero.")
    df = pd.DataFrame()
    model = None
    metrics = {
        "accuracy_test_set": "Error",
        "weighted_f1_test_set": "Error",
        "total_records_analyzed": "Error"
    }

# --- Función para generar gráficos (Corregida) ---
def generate_plots():
    if df.empty:
        return

    # Gráfico 1: Distribución de Tipos de Crimen
    plt.figure(figsize=(8, 5))
    law_counts = df['law_cat_cd'].value_counts()
    sns.countplot(
        data=df, 
        y='law_cat_cd', 
        order=law_counts.index, 
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
    sns.countplot(
        data=df, 
        x='day_of_week', 
        order=days_order, 
        palette='mako', 
        hue='day_of_week',
        hue_order=days_order,
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

# --- Ruta para el Dashboard Principal (ACTUALIZADA) ---
@app.route('/')
def dashboard():
    if df.empty or model is None:
        return "<h3>Error: Archivos no encontrados.</h3><p>Ejecuta primero el pipeline de Prefect: <code>python -m src.pipeline</code></p>"

    # --- Usar métricas del pipeline ---
    f1_score = metrics.get('weighted_f1_test_set', 'N/A')
    num_datos = metrics.get('total_records_analyzed', 'N/A')
    
    # Análisis de Peligrosidad por Lugar
    premise_counts = df['prem_typ_desc'].value_counts()
    most_dangerous = premise_counts.head(5).to_dict()
    least_dangerous = premise_counts.tail(5).to_dict()

    return render_template('dashboard.html',
                           model_metric=f1_score,
                           metric_name="F1-Score Ponderado (Test Set)", # Nuevo nombre para la métrica
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

# --- RUTA DE PRONÓSTICO (ACTUALIZADA PARA MULTICLASE) ---
@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if df.empty or model is None:
        return "<h3>Error: Archivos no encontrados.</h3><p>Ejecuta primero el pipeline de Prefect: <code>python -m src.pipeline</code></p>"
        
    premise_types = sorted(df['prem_typ_desc'].unique())
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    borough_names = list(BOROUGH_COORDINATES.keys())
    
    # --- ESTA ES LA LÓGICA NUEVA ---
    prediction_results = None # Esta es la variable que espera el HTML
    input_values = {}

    if request.method == 'POST':
        try:
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

            # Definimos las columnas en el orden que espera el pipeline
            features_order = [
                'latitude', 'longitude', 'prem_typ_desc', 'hour', 'day_of_week'
            ]
            
            # Creamos el DataFrame para predecir
            predict_df = pd.DataFrame([input_values], columns=features_order)

            # --- Predicción Multiclase ---
            probabilities = model.predict_proba(predict_df)[0]
            
            # Mapeamos las probabilidades a los nombres de las clases
            prediction_results = []
            for i, class_name in enumerate(model.classes_):
                prediction_results.append({
                    "name": class_name,
                    # Esta variable ahora se llama 'probability' para el HTML
                    "probability": f"{probabilities[i] * 100:.2f}" 
                })
            
            # Ordenamos por probabilidad descendente para la UI
            prediction_results = sorted(prediction_results, key=lambda x: float(x['probability']), reverse=True)

        except Exception as e:
            print(f"🛑 Error durante la predicción: {e}")
            prediction_results = [{"name": "Error", "probability": "0.00"}]

    return render_template('forecast.html', 
                           premise_types=premise_types, 
                           days=days_of_week,
                           boroughs=borough_names,
                           # Pasamos la lista de resultados
                           prediction_results=prediction_results, # Esta variable ahora existe
                           input_values=input_values)

# --- RUTA DE MAPA DE CALOR (Sin cambios) ---
@app.route('/heatmap')
def heatmap():
    if df.empty:
        return "<h3>Error: No se pudieron cargar los datos.</h3><p>Por favor, ejecuta el pipeline de Prefect primero.</p>"

    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="CartoDB positron")

    coords_df = df[['latitude', 'longitude']].copy()
    coords_df.dropna(subset=['latitude', 'longitude'], inplace=True)
    heat_data = coords_df.values.tolist()
    
    if heat_data:
        HeatMap(heat_data, radius=12, blur=15).add_to(m)
        print(f"✅ Mapa de calor generado con {len(heat_data)} puntos.")
    else:
        print("🛑 Advertencia: No se encontraron coordenadas válidas para generar el mapa de calor.")

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
if __name__ == "__main__":
    print("Generando gráficos estáticos para el dashboard...")
    generate_plots() # Genera los gráficos estáticos antes de arrancar
    print("Iniciando la aplicación Flask en http://127.0.0.1:5000")
    # Si 'run_app.py' usa puerto 5001, puedes cambiarlo aquí
    app.run(debug=True, port=5000)