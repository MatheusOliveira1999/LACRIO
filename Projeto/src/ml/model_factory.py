"""
Factory para criação de modelos de Machine Learning
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, 
                            VotingRegressor, GradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from typing import Dict, Any, List
import warnings

# XGBoost (opcional)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')


class ModelFactory:
    """Factory para criar modelos de ML configurados"""
    
    def __init__(self, variable: str = 'temperature', random_state: int = 42):
        self.variable = variable
        self.random_state = random_state
    
    def create_all_models(self, selected_models: List[str] = None) -> Dict[str, Pipeline]:
        """Cria todos os modelos disponíveis"""
        
        models = {}
        
        # Linear Regression
        models['LinearRegression'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
        
        # Ridge Regression
        models['Ridge'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0, random_state=self.random_state))
        ])
        
        # Random Forest
        models['RandomForest'] = Pipeline([
            ('model', RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=5,
                max_depth=None,
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])
        
        # Extra Trees
        models['ExtraTrees'] = Pipeline([
            ('model', ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])
        
        # Gradient Boosting
        models['GradientBoosting'] = Pipeline([
            ('model', GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            ))
        ])
        
        # SVR
        models['SVR'] = Pipeline([
            ('scaler', RobustScaler()),
            ('model', SVR(kernel='rbf', C=10, gamma='scale'))
        ])
        
        # MLP
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
        
        # Filtrar modelos selecionados
        if selected_models:
            filtered_models = {
                name: model for name, model in models.items() 
                if name in selected_models
            }
            return filtered_models
        
        return models
    
    def get_available_models(self) -> List[str]:
        """Retorna lista de modelos disponíveis"""
        available = [
            'LinearRegression', 'Ridge', 'RandomForest', 'ExtraTrees', 
            'GradientBoosting', 'SVR', 'MLP'
        ]
        
        if XGBOOST_AVAILABLE:
            available.append('XGBoost')
        
        return available
    
    def recommend_models_for_variable(self, variable: str) -> List[str]:
        """Recomenda modelos baseado na variável"""
        if variable == 'temperature':
            return ['LinearRegression', 'Ridge', 'RandomForest', 'ExtraTrees', 'MLP']
        elif variable == 'precipitation':
            return ['Ridge', 'RandomForest', 'ExtraTrees', 'MLP']
        else:
            return self.get_available_models()
