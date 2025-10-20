# src/train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

from .config import FEATURES, TARGET, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

def train_model(processed_data_path: str, model_path: str, preprocessor_path: str):
    """Trains the model using a preprocessing pipeline and saves both.
       Entrena el modelo usando un pipeline de preprocesamiento y guarda ambos."""
    print("Starting model training...")
    df = pd.read_csv(processed_data_path)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create a preprocessing pipeline
    # This pipeline handles scaling for numbers and one-hot encoding for categories
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ])

    # Create the full model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Save the fitted preprocessor and the full model pipeline
    joblib.dump(preprocessor.fit(X_train), preprocessor_path)
    joblib.dump(model_pipeline, model_path)
    
    print(f"Model training complete. Model saved to {model_path}")
    return model_path