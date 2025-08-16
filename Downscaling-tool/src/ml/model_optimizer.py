"""
model_optimizer.py
Módulo para otimização de hiperparâmetros
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, PowerTransformer


class ModelOptimizer:
    """Otimiza hiperparâmetros dos modelos"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.robust_scaler = None
        self.power_transformer = None
        self.skewed_features = []
    
    def optimize_model(self, name, pipeline, X_train, y_train):
        """Otimiza hiperparâmetros do modelo"""
        print(f"  Otimizando hiperparâmetros para {name}...")
        
        param_grid = self._get_param_grid(name)
        
        if param_grid is None:
            print(f"  Sem parâmetros para otimizar em {name}")
            return pipeline
        
        # Configurar CV baseado na resolução temporal
        cv_folds = self.config.get('cv_folds', 3)
        temporal_resolution = self.config.get('temporal_resolution', 'daily')
        
        # Para dados horários, limitar cv_folds para evitar overfitting
        if temporal_resolution == 'hourly' or (temporal_resolution == 'auto' and cv_folds > 4):
            cv_folds = min(cv_folds, 4)
            print(f"  Ajustando CV para dados horários: {cv_folds} folds")
        
        ts_cv = TimeSeriesSplit(n_splits=cv_folds, gap=0)
        
        try:
            use_random_search = self.config.get('use_random_search', False)
            n_iter = self.config.get('random_search_iter', 20)
            
            models_for_random_search = ['RandomForest', 'ExtraTrees', 'GradientBoosting', 
                                       'XGBoost', 'MLP', 'SVR']
            
            if use_random_search and name in models_for_random_search:
                print(f"  Usando RandomizedSearchCV com {n_iter} iterações")
                
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_grid,
                    cv=ts_cv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1 if self.config.get('verbose', False) else 0,
                    n_iter=n_iter,
                    random_state=self.config.get('random_state', 42)
                )
            else:
                n_combinations = 1
                for param, values in param_grid.items():
                    n_combinations *= len(values)
                
                print(f"  Usando GridSearchCV (busca completa com {n_combinations} combinações)")
                
                search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    cv=ts_cv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1 if self.config.get('verbose', False) else 0,
                    return_train_score=False
                )
            
            print(f"  Iniciando otimização...")
            search.fit(X_train, y_train)
            
            print(f"  ✓ Melhores parâmetros: {search.best_params_}")
            print(f"  ✓ Melhor score CV: {-search.best_score_:.4f}")
            
            if hasattr(search, 'cv_results_') and self.config.get('show_top_params', False):
                results_df = pd.DataFrame(search.cv_results_)
                top_3 = results_df.nsmallest(3, 'rank_test_score')
                print("\n  Top 3 combinações:")
                for idx, row in top_3.iterrows():
                    print(f"    {row['rank_test_score']}. Score: {-row['mean_test_score']:.4f}")
                    print(f"       Params: {row['params']}")
            
            return search.best_estimator_
            
        except Exception as e:
            print(f"  ⚠️ Erro na otimização: {str(e)}")
            print(f"  Usando modelo com parâmetros padrão")
            return pipeline
    
    def _get_param_grid(self, name):
        """Retorna grid de parâmetros para cada modelo"""
        
        if self.config.get('fast_optimization', True):
            # Grids reduzidos para otimização rápida
            param_grids = {
                'LinearRegression': {
                    'model__fit_intercept': [True, False]
                },
                'Ridge': {
                    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    'model__fit_intercept': [True, False]
                },
                'RandomForest': {
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [None, 10, 20],
                    'model__min_samples_leaf': [1, 5, 10],
                    'model__max_features': ['auto', 'sqrt', 0.7]
                },
                'ExtraTrees': {
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [None, 10, 20],
                    'model__min_samples_leaf': [1, 5, 10],
                    'model__max_features': ['auto', 'sqrt', 0.7]
                },
                'GradientBoosting': {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [5, 7, 10],
                    'model__learning_rate': [0.05, 0.1, 0.15],
                    'model__subsample': [0.8, 0.9, 1.0]
                },
                'XGBoost': {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [3, 6, 9],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__subsample': [0.8, 1.0]
                },
                'SVR': {
                    'model__C': [1, 10, 100],
                    'model__gamma': ['scale', 0.01, 0.1],
                    'model__kernel': ['rbf', 'linear']
                },
                'MLP': {
                    'model__hidden_layer_sizes': [(100,), (100, 50)],
                    'model__activation': ['relu'],
                    'model__alpha': [0.001, 0.01],
                    'model__max_iter': [2000]
                }
            }
        else:
            # Grids completos para otimização extensiva
            param_grids = {
                'LinearRegression': {
                    'model__fit_intercept': [True, False],
                    'model__positive': [True, False]
                },
                'Ridge': {
                    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                    'model__fit_intercept': [True, False],
                    'model__solver': ['auto', 'svd', 'cholesky', 'lsqr']
                },
                'RandomForest': {
                    'model__n_estimators': [50, 100, 200, 300, 500],
                    'model__max_depth': [None, 5, 10, 15, 20, 30],
                    'model__min_samples_split': [2, 5, 10, 20],
                    'model__min_samples_leaf': [1, 2, 5, 10],
                    'model__max_features': ['auto', 'sqrt', 'log2', 0.5, 0.7],
                    'model__bootstrap': [True, False]
                },
                'ExtraTrees': {
                    'model__n_estimators': [50, 100, 200, 300, 500],
                    'model__max_depth': [None, 5, 10, 15, 20, 30],
                    'model__min_samples_split': [2, 5, 10, 20],
                    'model__min_samples_leaf': [1, 2, 5, 10],
                    'model__max_features': ['auto', 'sqrt', 'log2', 0.5, 0.7],
                    'model__bootstrap': [True, False]
                },
                'GradientBoosting': {
                    'model__n_estimators': [50, 100, 200, 300],
                    'model__max_depth': [3, 5, 7, 10, 15],
                    'model__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 3, 5, 10]
                },
                'XGBoost': {
                    'model__n_estimators': [50, 100, 200, 300],
                    'model__max_depth': [3, 5, 7, 10],
                    'model__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                    'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'model__gamma': [0, 0.1, 0.2, 0.5],
                    'model__reg_alpha': [0, 0.01, 0.1, 1],
                    'model__reg_lambda': [0, 0.01, 0.1, 1]
                },
                'SVR': {
                    'model__C': [0.1, 1, 10, 100, 1000],
                    'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'model__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                    'model__epsilon': [0.01, 0.1, 0.2, 0.5]
                },
                'MLP': {
                    'model__hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 75)],
                    'model__activation': ['relu', 'tanh'],
                    'model__solver': ['adam', 'lbfgs'],
                    'model__alpha': [0.0001, 0.001, 0.01],
                    'model__learning_rate': ['adaptive'],
                    'model__learning_rate_init': [0.001, 0.01],
                    'model__max_iter': [2000, 5000]
                }
            }
        
        return param_grids.get(name)
    
    def robust_preprocessing(self, X, fit=True):
        """Pré-processamento robusto para evitar problemas de convergência"""
        
        if fit:
            self.robust_scaler = RobustScaler()
            self.power_transformer = PowerTransformer(method='yeo-johnson')
            
            skewed_features = []
            for col in X.columns:
                skewness = X[col].skew()
                if abs(skewness) > 2:
                    skewed_features.append(col)
            
            self.skewed_features = skewed_features
            
            X_processed = X.copy()
            
            if skewed_features:
                X_processed[skewed_features] = self.power_transformer.fit_transform(X[skewed_features])
            
            X_processed = pd.DataFrame(
                self.robust_scaler.fit_transform(X_processed),
                columns=X.columns,
                index=X.index
            )
        else:
            X_processed = X.copy()
            
            if hasattr(self, 'skewed_features') and self.skewed_features:
                X_processed[self.skewed_features] = self.power_transformer.transform(X[self.skewed_features])
            
            X_processed = pd.DataFrame(
                self.robust_scaler.transform(X_processed),
                columns=X.columns,
                index=X.index
            )
        
        return X_processed