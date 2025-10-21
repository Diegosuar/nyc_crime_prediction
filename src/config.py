COMPLAINTS_URL = "https://data.cityofnewyork.us/resource/qgea-i56i.csv?$limit=50000" 

RAW_DATA_PATH = "data/raw/complaints.csv"
PROCESSED_DATA_PATH = "data/processed/processed_complaints.csv"
MODEL_PATH = "models/random_forest_classifier.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
REPORT_PATH = "reports/crime_prediction_report.html"
CONFUSION_MATRIX_PATH = "reports/figures/confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = "reports/figures/feature_importance.png"

FEATURES = [
    'latitude', 
    'longitude',
    'prem_typ_desc', 
    'hour', 
    'day_of_week'
]

TARGET = 'is_robbery'

# Features that need categorical encoding
CATEGORICAL_FEATURES = ['prem_typ_desc', 'day_of_week']

# Features that are numerical
NUMERICAL_FEATURES = ['latitude', 'longitude', 'hour']

# Model parameters for GridSearchCV
MODEL_PARAMS = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}