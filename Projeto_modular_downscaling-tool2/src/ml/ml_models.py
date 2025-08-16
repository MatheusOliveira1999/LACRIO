"""
ml_models.py
Módulo com definições dos modelos de Machine Learning
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import pandas as pd

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost não disponível. Instale com: pip install xgboost")
    XGBOOST_AVAILABLE = False


class ModelBuilder:
    """Constrói e configura modelos de ML"""
    
    def __init__(self, variable='temperature', config=None):
        self.variable = variable
        self.config = config or {}
        self.random_state = self.config.get('random_state', 42)
    
    def get_models(self):
        """Define modelos para cada variável"""
        models = {}
        
        # Modelos básicos
        models['LinearRegression'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
        
        models['Ridge'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0, random_state=self.random_state))
        ])
        
        # Modelos baseados em árvores
        models['RandomForest'] = Pipeline([
            ('model', RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=5,
                max_depth=None,
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])
        
        models['ExtraTrees'] = Pipeline([
            ('model', ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])
        
        models['GradientBoosting'] = Pipeline([
            ('model', GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            ))
        ])
        
        # SVM
        models['SVR'] = Pipeline([
            ('scaler', RobustScaler()),
            ('model', SVR(kernel='rbf', C=10, gamma='scale'))
        ])
        
        # Rede Neural
        models['MLP'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=5000,
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=50,
                random_state=self.random_state,
                verbose=False
            ))
        ])
        
        # XGBoost (se disponível)
        if XGBOOST_AVAILABLE:
            if self.variable == 'temperature':
                models['XGBoost'] = Pipeline([
                    ('model', XGBRegressor(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.1,
                        objective='reg:squarederror',
                        random_state=self.random_state,
                        n_jobs=-1
                    ))
                ])
            else:  # precipitation
                models['XGBoost'] = Pipeline([
                    ('model', XGBRegressor(
                        n_estimators=300,
                        max_depth=8,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='reg:squarederror',
                        random_state=self.random_state,
                        n_jobs=-1
                    ))
                ])
        
        return models