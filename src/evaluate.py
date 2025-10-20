# src/evaluate.py (VERSIÓN FINAL)

import pandas as pd
from prefect import task
from sklearn.metrics import classification_report, accuracy_score
import json

@task
def evaluate_model_task(model, X_test: pd.DataFrame, y_test: pd.Series, label_encoder, location_analysis: dict):
    """
    Evalúa el modelo y compila las métricas y análisis en un solo archivo.
    """
    print(" -> Evaluando el modelo y generando reporte final...")

    # --- Evaluación del Modelo ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    report['accuracy'] = accuracy

    # --- Importancia de Características ---
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    feature_importances_dict = importances.sort_values(ascending=False).to_dict()

    # --- Compilar todo en un solo archivo JSON ---
    final_metrics = {
        "model_performance": report,
        "location_analysis": location_analysis, # Usamos el diccionario que recibimos
        "feature_importances": feature_importances_dict
    }

    with open('reports/dashboard_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)

    print(" -> Métricas y análisis del dashboard guardados en 'reports/dashboard_metrics.json'")
    print(f" -> Exactitud del modelo (Accuracy): {accuracy:.4f}")

    return final_metrics