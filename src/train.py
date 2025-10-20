# src/train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

# --- MEJORA DE PRECISIÓN: Importar SMOTE y Pipeline de imblearn ---
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from .config import FEATURES, TARGET, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, MODEL_PARAMS

def train_model(processed_data_path: str, model_path: str, preprocessor_path: str):
    """Entrena el modelo usando SMOTE para desbalance y GridSearchCV."""
    print("Starting model training with SMOTE...")
    df = pd.read_csv(processed_data_path)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Pipeline de Preprocesamiento (igual que antes)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ], remainder='passthrough')

    # Guardamos el preprocesador (sigue siendo una buena práctica)
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")

    # --- MEJORA DE PRECISIÓN 3: Pipeline con SMOTE ---
    # Usamos el Pipeline de 'imblearn' que permite incluir SMOTE
    
    # NOTA: quitamos 'class_weight' de RandomForest, ya que SMOTE se encarga del desbalance.
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)), # <-- ¡AQUÍ ESTÁ LA MAGIA!
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    # 2. Hyperparameter Tuning (igual que antes)
    param_grid = {
        f'classifier__{key}': value for key, value in MODEL_PARAMS.items()
    }
    
    # Seguimos usando 'f1_weighted'
    print("Starting GridSearchCV with SMOTE...")
    grid_search = GridSearchCV(
        model_pipeline, 
        param_grid, 
        cv=3, 
        n_jobs=-1, 
        scoring='f1_weighted', 
        verbose=2
    )

    # Entrenar el GridSearch
    grid_search.fit(X_train, y_train)

    print(f"GridSearchCV complete. Best F1-score: {grid_search.best_score_}")
    print(f"Best parameters found: {grid_search.best_params_}")

    # Guardamos el mejor modelo
    joblib.dump(grid_search.best_estimator_, model_path)
    
    print(f"Best model saved to {model_path}")
    return model_path