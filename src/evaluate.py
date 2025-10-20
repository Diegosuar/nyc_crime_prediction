# src/evaluate.py (VERSIÓN FINAL CON GRÁFICOS Y MÉTRICAS COMPLETAS)

import pandas as pd
import numpy as np
from prefect import task
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os # Importar os para manejo de directorios

@task
def evaluate_model_task(model, X_test: pd.DataFrame, y_test: pd.Series, label_encoder, location_analysis: dict):
    """
    Evalúa el modelo, genera gráficos (matriz de confusión, importancia de características)
    y compila todo en un archivo JSON.
    """
    print(" -> Evaluando el modelo y generando reporte final con gráficos...")

    # --- Evaluación del Modelo ---
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilidad de la clase positiva (Violento=1)

    accuracy = accuracy_score(y_test, y_pred)
    # Calculamos AUC-ROC
    try:
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        print(f" -> AUC-ROC Score: {auc_roc:.4f}")
    except ValueError as e:
        print(f" -> ADVERTENCIA: No se pudo calcular AUC-ROC: {e}")
        auc_roc = None # O manejarlo como 0 o NaN

    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    report['accuracy'] = accuracy
    if auc_roc is not None:
        report['auc_roc'] = auc_roc

    # Extraemos el F1-Score específico para la clase "Violento" (asumiendo que es la clase '1')
    f1_violent = report.get('Violento', {}).get('f1-score', None)

    # --- Generación de Gráficos ---
    img_dir = os.path.join('app', 'static', 'img') # Directorio donde guardar las imágenes
    os.makedirs(img_dir, exist_ok=True)

    # 1. Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    cm_path = os.path.join(img_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f" -> Matriz de confusión guardada en: {cm_path}")

    # 2. Importancia de Características
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=X_test.columns).nlargest(15) # Top 15
        plt.figure(figsize=(10, 8))
        sns.barplot(x=importances.values, y=importances.index, palette='viridis')
        plt.title('Top 15 Características Más Importantes')
        plt.xlabel('Importancia Relativa')
        plt.ylabel('Característica')
        plt.tight_layout()
        fi_path = os.path.join(img_dir, 'feature_importance.png')
        plt.savefig(fi_path)
        plt.close()
        print(f" -> Gráfico de importancia de características guardado en: {fi_path}")
        feature_importances_dict = importances.to_dict()
    else:
        print(" -> ADVERTENCIA: El modelo no tiene el atributo 'feature_importances_'.")
        feature_importances_dict = {}
        fi_path = None


    # --- Compilar todo en un solo archivo JSON ---
    final_metrics = {
        "model_performance": report,
        "location_analysis": location_analysis,
        "feature_importances": feature_importances_dict,
        # Guardamos las rutas relativas a la carpeta 'static' para usarlas en HTML
        "plot_paths": {
             "confusion_matrix": f"img/confusion_matrix.png" if cm_path else None,
             "feature_importance": f"img/feature_importance.png" if fi_path else None
        }
    }

    with open('reports/dashboard_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)

    print(" -> Métricas, análisis y rutas de gráficos guardados en 'reports/dashboard_metrics.json'")
    print(f" -> Exactitud del modelo (Accuracy): {accuracy:.4f}")
    if f1_violent is not None:
         print(f" -> F1-Score para 'Violento': {f1_violent:.4f}")


    return final_metrics