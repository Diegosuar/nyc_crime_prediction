# src/evaluate.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster, HeatMap
import os
import json
# Importamos todas las variables de configuración necesarias
from .config import (
    FEATURES, TARGET, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, 
    REPORT_PATH, CONFUSION_MATRIX_PATH, FEATURE_IMPORTANCE_PATH
)

def evaluate(processed_data_path: str, model_path: str):
    """Evalúa el modelo multiclase y genera un reporte HTML interactivo."""
    print("Evaluating multiclass model and generating report...")
    
    os.makedirs(os.path.dirname(CONFUSION_MATRIX_PATH), exist_ok=True)
    
    model = joblib.load(model_path)
    df = pd.read_csv(processed_data_path)

    X = df[FEATURES]
    y = df[TARGET]
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    y_pred = model.predict(X_test)
    
    # Obtenemos las clases del modelo (ej. ['FELONY', 'MISDEMEANOR', 'VIOLATION'])
    class_labels = model.classes_
    
    # 1. Classification Report (como texto)
    report_str = classification_report(y_test, y_pred, labels=class_labels)
    print("Classification Report:\n", report_str)

    # 2. Matriz de Confusión Normalizada (como imagen)
    # Normalizamos por fila ('true') para ver el % de acierto por clase
    cm = confusion_matrix(y_test, y_pred, labels=class_labels, normalize='true')
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_labels, 
                yticklabels=class_labels)
    plt.title('Matriz de Confusión Normalizada (por fila)')
    plt.ylabel('Clase Actual')
    plt.xlabel('Clase Predicha')
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")
    plt.close()

    # 3. Importancia de Características (como imagen)
    try:
        preprocessor_step = model.named_steps['preprocessor']
        cat_features_out = preprocessor_step.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)
        feature_names = NUMERICAL_FEATURES + list(cat_features_out)
        
        importances = model.named_steps['classifier'].feature_importances_
        forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=forest_importances.values, y=forest_importances.index, palette='viridis', hue=forest_importances.index, legend=False)
        plt.title("Top 15 Características más Importantes")
        plt.xlabel("Reducción Media de Impureza")
        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_PATH)
        print(f"Feature importance plot saved to {FEATURE_IMPORTANCE_PATH}")
        plt.close()
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")


    # 4. Mapa Interactivo de Folium (para multiclase)
    print("Generating interactive performance map...")
    test_sample = X_test.copy()
    test_sample['actual'] = y_test
    test_sample['predicted'] = y_pred
    
    map_sample = test_sample.sample(n=min(2000, len(test_sample)), random_state=42)

    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="CartoDB positron")
    
    # --- MEJORA DE ESTÉTICA: Colores y clusters por gravedad ---
    colors = {'FELONY': 'red', 'MISDEMEANOR': 'orange', 'VIOLATION': 'blue'}
    clusters = {
        'FELONY': MarkerCluster(name="Actual: FELONY").add_to(m),
        'MISDEMEANOR': MarkerCluster(name="Actual: MISDEMEANOR").add_to(m),
        'VIOLATION': MarkerCluster(name="Actual: VIOLATION").add_to(m)
    }

    for idx, row in map_sample.iterrows():
        actual_class = row.actual
        predicted_class = row.predicted
        
        color = colors.get(actual_class, 'gray') # Color por el valor real
        cluster = clusters.get(actual_class)
        
        # El popup ahora muestra la discrepancia (si existe)
        popup_html = f"""
        <b>Lugar:</b> {row.prem_typ_desc}<br>
        <b>Día:</b> {row.day_of_week}, <b>Hora:</b> {int(row.hour)}<br>
        <hr>
        <b>Actual:</b> <span style='color:{color}; font-weight:bold;'>{actual_class}</span><br>
        <b>Predicho:</b> {predicted_class}
        """
        
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(cluster)

    # Añadimos un Heatmap de todos los DELITOS GRAVES (FELONY)
    felonies = df[df[TARGET] == 'FELONY'][['latitude', 'longitude']].values
    HeatMap(felonies, name="Densidad de Delitos Graves (Todos)", radius=10, blur=12).add_to(m)

    folium.LayerControl().add_to(m)
        
    # 5. Crear el contenido HTML para el reporte
    html_content = f"""
    <html>
    <head>
        <title>Reporte de Predicción de Gravedad del Crimen</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ font-family: sans-serif; padding: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 10px; margin-top: 30px; }}
            pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin-top: 10px; }}
            .container {{ max-width: 1200px; margin: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mt-4">Reporte de Predicción de Gravedad del Crimen en NYC</h1>
            <p>Este reporte detalla el rendimiento de un modelo (Random Forest + SMOTE) 
               entrenado para predecir la gravedad de un incidente: 
               <b>FELONY</b> (Delito Grave), <b>MISDEMEANOR</b> (Delito Menor), o <b>VIOLATION</b> (Infracción).</p>
            
            <h2>Rendimiento del Modelo</h2>
            <p>Se utiliza el <b>F1-Score ponderado (weighted avg)</b> como la métrica principal,
               ya que las clases están naturalmente desbalanceadas.</p>
            
            <h3>Reporte de Clasificación</h3>
            <pre>{report_str}</pre>
            
            <h3>Matriz de Confusión (Normalizada)</h3>
            <p>Esto muestra el porcentaje de predicciones correctas e incorrectas para cada clase. 
               El eje Y representa la verdad real; el eje X la predicción del modelo.</p>
            <img src='figures/confusion_matrix.png' class="img-fluid">
            
            <h3>Importancia de Características</h3>
            <p>Los 15 factores más importantes que usó el modelo para tomar sus decisiones.</p>
            <img src='figures/feature_importance.png' class="img-fluid">

            <h2 class="mt-5">Mapa Interactivo de Rendimiento</h2>
            <p>Muestra un muestreo de 2,000 predicciones en los datos de prueba. 
               Los clusters están coloreados por la <b>gravedad real</b> del incidente. 
               (Rojo=FELONY, Naranja=MISDEMEANOR, Azul=VIOLATION).
               Puedes activar y desactivar las capas en la esquina superior derecha.</p>
            {m._repr_html_()}
        </div>
    </body>
    </html>
    """
    
    report_file_path = REPORT_PATH
    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    
    print(f"\n¡Éxito! Reporte interactivo guardado en {report_file_path}")

    # --- PASO FINAL: Guardar métricas clave para la App Flask ---
    import json
    from sklearn.metrics import accuracy_score, f1_score

    # Calcular métricas clave
    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Obtener conteo de datos
    total_records = len(df)
    
    metrics = {
        "accuracy_test_set": f"{accuracy*100:.2f}%",
        "weighted_f1_test_set": f"{weighted_f1*100:.2f}%",
        "total_records_analyzed": f"{total_records:,}"
    }
    
    # Guardar en un archivo JSON que la app pueda leer
    metrics_path = "dashboard_metrics.json" 
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Métricas del dashboard guardadas en {metrics_path}")