# Data source URLs
# CORRECCIÓN FINAL: Esta es la URL correcta para el dataset de "NYPD Complaint Data Historic".
COMPLAINTS_URL = "https://data.cityofnewyork.us/resource/qgea-i56i.csv?$limit=50000" # Limit to 50k for speed

# File paths
RAW_DATA_PATH = "data/raw/complaints.csv"
PROCESSED_DATA_PATH = "data/processed/processed_complaints.csv"
MODEL_PATH = "models/random_forest_classifier.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
REPORT_PATH = "reports/crime_prediction_report.html"
CONFUSION_MATRIX_PATH = "reports/figures/confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = "reports/figures/feature_importance.png"


# Feature engineering and modeling config
# Features to use for prediction
FEATURES = [
    'latitude', 
    'longitude',
    'prem_typ_desc', # Premise Type Description (e.g., STREET, RESIDENCE)
    'hour', 
    'day_of_week'
]

# The variable we want to predict
TARGET = 'is_robbery'

# Features that need categorical encoding
CATEGORICAL_FEATURES = ['prem_typ_desc', 'day_of_week']

# Features that are numerical
NUMERICAL_FEATURES = ['latitude', 'longitude', 'hour']

# Model parameters for GridSearchCV (optional, but good practice)
MODEL_PARAMS = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}