# src/evaluate.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster

from .config import FEATURES, TARGET, REPORT_PATH, CONFUSION_MATRIX_PATH, FEATURE_IMPORTANCE_PATH

def evaluate(processed_data_path: str, model_path: str):
    """Evaluates the model and generates an interactive HTML report with a map.
       Evalúa el modelo y genera un reporte HTML interactivo con un mapa."""
    print("Evaluating model and generating report...")
    model = joblib.load(model_path)
    df = pd.read_csv(processed_data_path)

    X = df[FEATURES]
    y = df[TARGET]
    
    # We need a test set to evaluate
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    y_pred = model.predict(X_test)
    
    # 1. Classification Report (as text)
    report_str = classification_report(y_test, y_pred)
    print("Classification Report:\n", report_str)

    # 2. Confusion Matrix (as image)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")

    # 3. Feature Importance (as image)
    try:
        importances = model.named_steps['classifier'].feature_importances_
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=forest_importances, y=forest_importances.index)
        plt.title("Feature Importances")
        plt.xlabel("Mean decrease in impurity")
        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_PATH)
        print(f"Feature importance plot saved to {FEATURE_IMPORTANCE_PATH}")
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")


    # 4. Interactive Map with Folium
    # Use a sample to avoid crashing the browser
    test_sample = X_test.copy()
    test_sample['actual'] = y_test
    test_sample['predicted'] = y_pred
    map_sample = test_sample.sample(n=min(2000, len(test_sample)), random_state=42)

    # Map centered on NYC
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
    
    # Create marker clusters for performance
    marker_cluster = MarkerCluster().add_to(m)

    colors = {'FELONY': 'red', 'MISDEMEANOR': 'orange', 'VIOLATION': 'blue'}

    for idx, row in map_sample.iterrows():
        popup_html = f"""
        <b>Location:</b> ({row.latitude:.4f}, {row.longitude:.4f})<br>
        <b>Premise:</b> {row.prem_typ_desc}<br>
        <b>Time:</b> Day {row.day_of_week}, Hour {int(row.hour)}<br>
        <hr>
        <b>Actual Crime:</b> {row.actual}<br>
        <b>Predicted Crime:</b> {row.predicted}
        """
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=5,
            color=colors.get(row.actual, 'gray'),
            fill=True,
            fill_color=colors.get(row.actual, 'gray'),
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(marker_cluster)
        
    # Create HTML content for the report
    html_content = f"""
    <html>
    <head><title>Crime Prediction Report</title></head>
    <body style="font-family: sans-serif;">
        <h1>NYC Crime Prediction Report</h1>
        <p>This report details the performance of a Random Forest model trained to predict crime categories.</p>
        
        <h2>Model Performance</h2>
        <h3>Classification Report</h3>
        <pre>{report_str}</pre>
        
        <h3>Confusion Matrix</h3>
        <p>This shows the number of correct and incorrect predictions for each class.</p>
        <img src='figures/confusion_matrix.png' width='600'>
        
        <h3>Feature Importances</h3>
        <p>Which factors were most important for the model's predictions.</p>
        <img src='figures/feature_importance.png' width='700'>

        <h2>Interactive Prediction Map</h2>
        <p>A map showing a sample of 2,000 predictions on the test set. Markers are colored by the <b>actual</b> crime type. Click a marker for details.</p>
        {m._repr_html_()}
    </body>
    </html>
    """
    
    with open(REPORT_PATH, 'w') as f:
        f.write(html_content)
    
    print(f"Interactive report saved to {REPORT_PATH}")