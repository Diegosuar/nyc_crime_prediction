import pandas as pd
from prefect import task
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump

@task
def train_model_task(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Busca los mejores hiperparámetros para un XGBClassifier y entrena el modelo final.
    """
    print(" -> Iniciando búsqueda de hiperparámetros con XGBoost...")
    
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
    
    random_search = RandomizedSearchCV(
        estimator=XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        param_distributions=param_distributions,
        n_iter=20, 
        cv=3,     
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f" -> Mejores parámetros encontrados: {random_search.best_params_}")
    
    best_model = random_search.best_estimator_
    
    dump(best_model, 'models/crime_predictor_model.joblib')
    print(" -> Mejor modelo (XGBoost) guardado en 'models/crime_predictor_model.joblib'")
    
    return best_model