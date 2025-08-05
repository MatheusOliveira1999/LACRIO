"""
Configurações globais do sistema de downscaling
"""

import os
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# Diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
IMG_FOLDER = os.path.join(STATIC_FOLDER, 'img')

# Flask
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB

# Configurações padrão
DEFAULT_MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 3,
    'optimize_hyperparams': True,
    'feature_selection': True,
    'max_features': 100,
    'ensemble_size': 3,
    'fast_optimization': True,
    'use_random_search': False,
    'random_search_iter': 20,
    'show_top_params': True,
    'verbose': True,
    'optimize_all_models': True,
    'use_optuna': False
}

# Modelos disponíveis
AVAILABLE_MODELS = [
    'LinearRegression', 'Ridge', 'Lasso', 'RandomForest', 'ExtraTrees',
    'GradientBoosting', 'XGBoost', 'SVR', 'MLP'
]

# Variáveis climáticas
CLIMATE_VARIABLES = ['temperature', 'precipitation']

# Diretórios necessários
REQUIRED_DIRS = [UPLOAD_FOLDER, MODELS_FOLDER, RESULTS_FOLDER, IMG_FOLDER]

def create_directories():
    """Cria diretórios necessários"""
    for directory in REQUIRED_DIRS:
        if not os.path.exists(directory):
            os.makedirs(directory)
