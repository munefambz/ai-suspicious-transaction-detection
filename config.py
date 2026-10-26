import os

class Config:
    # Secret Key for session signing
    SECRET_KEY = os.environ.get('SECRET_KEY', 'aml-secure-prototype-key-2026')
    
    # Database Configuration (supports switching to MySQL by setting DATABASE_URL env var)
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DEFAULT_DB_PATH = os.path.join(BASE_DIR, 'data', 'aml_system.db')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', f'sqlite:///{DEFAULT_DB_PATH}')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Machine Learning Settings
    MODEL_DIR = os.path.join(BASE_DIR, 'data')
    MODELS = {
        'Logistic Regression': os.path.join(MODEL_DIR, 'logistic_regression_model.joblib'),
        'Decision Tree': os.path.join(MODEL_DIR, 'decision_tree_model.joblib'),
        'Random Forest': os.path.join(MODEL_DIR, 'random_forest_model.joblib')
    }
    DEFAULT_MODEL = 'Random Forest'
    
    # AML Alert Thresholds (model output probabilities)
    MEDIUM_RISK_THRESHOLD = 0.3
    HIGH_RISK_THRESHOLD = 0.7

    # Generative AI (LLM) Integration Settings (OpenRouter & NVIDIA NIM)
    OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
    OPENROUTER_MODEL = os.environ.get('OPENROUTER_MODEL', 'meta-llama/llama-3.1-8b-instruct:free')
    
    NVIDIA_NIM_API_KEY = os.environ.get('NVIDIA_NIM_API_KEY', '')
    NVIDIA_NIM_MODEL = os.environ.get('NVIDIA_NIM_MODEL', 'meta/llama-3.1-70b-instruct')

