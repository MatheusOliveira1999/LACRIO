"""
Otimização de hiperparâmetros
"""

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import numpy as np


class HyperparameterOptimizer:
    """Classe para otimização de hiperparâmetros"""
    
    def __init__(self, variable: str, config: dict):
        self.variable = variable
        self.config = config
    
    def optimize_model(self, model_name: str, pipeline, X_train, y_train):
        """Otimiza hiperparâmetros básicos"""
        print(f"  ⚙️ Otimizando {model_name}...")
        
        param_grids = {
            'RandomForest': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [None, 10, 20]
            },
            'Ridge': {
                'model__alpha': [0.1, 1.0, 10.0]
            }
        }
        
        if model_name in param_grids:
            try:
                ts_cv = TimeSeriesSplit(n_splits=3)
                grid_search = GridSearchCV(
                    pipeline, param_grids[model_name],
                    cv=ts_cv, scoring='neg_mean_squared_error', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                return grid_search.best_estimator_
            except:
                pass
        
        return pipeline
