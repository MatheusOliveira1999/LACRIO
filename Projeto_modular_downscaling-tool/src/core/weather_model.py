"""
weather_model.py
Classe principal do modelo de downscaling
"""

import numpy as np
import pandas as pd
import xarray as xr
import joblib
import warnings
import os
import json
import sys
from datetime import datetime

from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression

# Imports locais usando imports absolutos
from src.ml.ml_models import ModelBuilder
from src.ml.feature_engineering import FeatureEngineer
from src.ml.model_optimizer import ModelOptimizer
from src.core.data_processor import DataProcessor

warnings.filterwarnings('ignore')


class WeatherDownscalingModel:
    """
    Classe principal para downscaling de dados climáticos usando Machine Learning
    """
    
    def __init__(self, variable='temperature', config=None):
        """
        Inicializa o modelo de downscaling
        
        Parameters:
        -----------
        variable : str
            'temperature' ou 'precipitation'
        config : dict
            Configurações opcionais do modelo incluindo otimização
        """
        self.variable = variable
        self.config = config or {}
        self.models = {}
        self.results = {}
        self.best_model = None
        self.features = None
        self.data = None
        self.scaler = None
        self.feature_selector = None
        
        # Configurações padrão
        self.default_config = {
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
        
        # Mesclar configurações
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Inicializar helpers
        self.data_processor = DataProcessor(self.variable)
        self.feature_engineer = FeatureEngineer(self.variable)
        self.model_builder = ModelBuilder(self.variable, self.config)
        self.model_optimizer = ModelOptimizer(self.config)
    
    def load_and_merge_data(self, era5_path, station_path, dem_path, lat, lon):
        """Carrega e mescla todos os dados necessários"""
        self.data = self.data_processor.load_and_merge_data(
            era5_path, station_path, dem_path, lat, lon
        )
        return self.data
    
    def create_features(self):
        """Cria todas as features para o modelo"""
        self.data = self.feature_engineer.create_all_features(self.data)
        return self.data
    
    def prepare_data(self, split_date='2018-01-01'):
        """Prepara dados para modelagem"""
        print("Preparando dados para modelagem...")
        
        # Definir variável alvo
        target_col = 'temp_obs' if self.variable == 'temperature' else 'prec_obs'
        
        # Selecionar features (excluir colunas não numéricas e alvo)
        exclude_cols = ['date', 'temp_obs', 'prec_obs']
        self.features = [c for c in self.data.columns if c not in exclude_cols]
        
        # Filtrar apenas features numéricas
        numeric_features = []
        for col in self.features:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                numeric_features.append(col)
        self.features = numeric_features
        
        print(f"Features selecionadas: {len(self.features)}")
        
        # Preparar X e y
        X = self.data[self.features].copy()
        y = self.data[target_col].copy()

        # Remover valores infinitos e NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)

        # Verificar se há NaN
        nan_mask = X.isna().any(axis=1) | y.isna()
        if nan_mask.sum() > 0:
            print(f"⚠️ Removendo {nan_mask.sum()} linhas com valores faltantes/infinitos")
            X = X[~nan_mask]
            y = y[~nan_mask]
            self.data = self.data[~nan_mask]

        # Verificar escala dos dados
        for col in X.columns:
            col_std = X[col].std()
            col_mean = X[col].mean()
            if col_std > 1000 or abs(col_mean) > 1000:
                print(f"⚠️ Feature '{col}' tem escala muito grande (mean={col_mean:.1f}, std={col_std:.1f})")
        
        # Tratamento especial para precipitação
        if self.variable == 'precipitation':
            y = np.log1p(y)
        
        # Dividir em treino e teste
        split_date = pd.to_datetime(split_date)
        train_mask = self.data['date'] < split_date
        
        self.X_train = X[train_mask]
        self.X_test = X[~train_mask]
        self.y_train = y[train_mask]
        self.y_test = y[~train_mask]
        
        # Guardar datas
        self.train_dates = self.data[train_mask]['date']
        self.test_dates = self.data[~train_mask]['date']
        
        # Seleção de features (se habilitada)
        if self.config['feature_selection']:
            self._select_features()
        
        print(f"Dados preparados - Treino: {self.X_train.shape}, Teste: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _select_features(self):
        """Seleciona as melhores features"""
        print("Selecionando features...")
        
        max_features = min(self.config['max_features'], len(self.features))
        
        self.feature_selector = SelectKBest(score_func=f_regression, k=max_features)
        
        X_train_selected = self.feature_selector.fit_transform(self.X_train, self.y_train)
        X_test_selected = self.feature_selector.transform(self.X_test)
        
        selected_features = [self.features[i] for i in self.feature_selector.get_support(indices=True)]
        
        self.X_train = pd.DataFrame(X_train_selected, columns=selected_features, index=self.X_train.index)
        self.X_test = pd.DataFrame(X_test_selected, columns=selected_features, index=self.X_test.index)
        self.features = selected_features
        
        print(f"Features selecionadas: {len(self.features)}")
    
    def get_models(self):
        """Define modelos para cada variável"""
        return self.model_builder.get_models()
    
    def train_models(self, selected_models=None, optimize=None):
        """Treina todos os modelos selecionados"""
        if optimize is None:
            optimize = self.config['optimize_hyperparams']
        
        print(f"\nTreinando modelos para {self.variable}...")
        
        all_models = self.get_models()
        
        # Aplicar pré-processamento se necessário
        if 'MLP' in (selected_models or all_models.keys()):
            print("Aplicando pré-processamento robusto para MLP...")
            self.X_train_robust = self.model_optimizer.robust_preprocessing(self.X_train, fit=True)
            self.X_test_robust = self.model_optimizer.robust_preprocessing(self.X_test, fit=False)
        
        if selected_models:
            self.models = {k: v for k, v in all_models.items() if k in selected_models}
        else:
            self.models = all_models
        
        print(f"Modelos a treinar: {list(self.models.keys())}")
        
        if self.config['optimize_all_models']:
            models_to_optimize = list(self.models.keys())
        else:
            models_to_optimize = ['RandomForest', 'XGBoost', 'Ridge', 'GradientBoosting', 
                                'ExtraTrees', 'SVR', 'MLP']
        
        for name, pipeline in self.models.items():
            print(f"\n{name}...")
            
            try:
                if optimize and name in models_to_optimize:
                    pipeline = self.model_optimizer.optimize_model(
                        name, pipeline, self.X_train, self.y_train
                    )
                else:
                    print(f"  Usando parâmetros padrão")

                if name == 'MLP' and hasattr(self, 'X_train_robust'):
                    X_train_temp = self.X_train_robust
                    X_test_temp = self.X_test_robust
                else:
                    X_train_temp = self.X_train
                    X_test_temp = self.X_test
                                
                print(f"  Treinando modelo final...")
                pipeline.fit(X_train_temp, self.y_train)
                
                y_pred = pipeline.predict(X_test_temp)
                
                if self.variable == 'precipitation':
                    y_pred = np.expm1(y_pred)
                    y_test_original = np.expm1(self.y_test)
                else:
                    y_test_original = self.y_test
                
                metrics = self.calculate_metrics(y_test_original, y_pred)
                
                self.results[name] = {
                    'model': pipeline,
                    'predictions': y_pred,
                    'metrics': metrics
                }
                
                print(f"  ✓ RMSE: {metrics['RMSE']:.4f}")
                print(f"  ✓ R²: {metrics['R2']:.4f}")
                
            except Exception as e:
                print(f"  ❌ Erro ao treinar {name}: {str(e)}")
                continue
    
    def calculate_metrics(self, y_true, y_pred):
        """Calcula métricas de avaliação"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        
        ss_clim = 1 - (rmse**2 / np.var(y_true))
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Bias': bias,
            'Skill_Score': ss_clim,
            'Correlation': corr
        }
        
        if self.variable == 'precipitation':
            rain_threshold = 0.1
            rain_true = y_true > rain_threshold
            rain_pred = y_pred > rain_threshold
            
            if len(rain_true) > 0:
                rain_accuracy = np.mean(rain_true == rain_pred) * 100
                
                if np.sum(rain_true) > 0:
                    pod = np.sum(rain_true & rain_pred) / np.sum(rain_true)
                else:
                    pod = np.nan
                
                if np.sum(rain_pred) > 0:
                    far = np.sum(~rain_true & rain_pred) / np.sum(rain_pred)
                else:
                    far = np.nan
                
                metrics.update({
                    'Rain_Detection_Accuracy': rain_accuracy,
                    'POD': pod,
                    'FAR': far
                })
            
            if np.sum(rain_true) > 10:
                rain_days_mask = rain_true
                r2_rain = r2_score(y_true[rain_days_mask], y_pred[rain_days_mask])
                metrics['R2_Rain_Days'] = r2_rain
        
        return metrics
    
    def create_ensemble(self, ensemble_size=None):
        """Cria modelo ensemble dos melhores modelos"""
        if ensemble_size is None:
            ensemble_size = self.config['ensemble_size']
        
        print(f"\nCriando modelo ensemble com {ensemble_size} modelos...")
        
        if len(self.results) < 2:
            print("Insufficient models for ensemble creation")
            return None
        
        best_models = sorted(self.results.items(), 
                           key=lambda x: x[1]['metrics']['RMSE'])[:ensemble_size]
        
        print(f"Modelos selecionados: {[name for name, _ in best_models]}")
        
        ensemble_models = [(name, result['model']) for name, result in best_models]
        ensemble = VotingRegressor(ensemble_models)
        
        ensemble.fit(self.X_train, self.y_train)
        
        y_pred = ensemble.predict(self.X_test)
        
        if self.variable == 'precipitation':
            y_pred = np.expm1(y_pred)
            y_test_original = np.expm1(self.y_test)
        else:
            y_test_original = self.y_test
        
        metrics = self.calculate_metrics(y_test_original, y_pred)
        
        self.results['Ensemble'] = {
            'model': ensemble,
            'predictions': y_pred,
            'metrics': metrics
        }
        
        print(f"Ensemble RMSE: {metrics['RMSE']:.4f}")
        print(f"Ensemble R²: {metrics['R2']:.4f}")
        
        return ensemble
    
    def save_models(self, directory=None):
        """Salva todos os modelos treinados"""
        if directory is None:
            # Caminho absoluto para a pasta models na raiz do projeto
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            directory = os.path.join(project_root, 'models')
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        saved_models = []
        
        for name, result in self.results.items():
            model_path = os.path.join(directory, f'{name.lower()}_{self.variable}.pkl')
            try:
                joblib.dump(result['model'], model_path)
                saved_models.append(model_path)
                print(f"Modelo salvo: {model_path}")
            except Exception as e:
                print(f"Erro ao salvar {name}: {str(e)}")
        
        model_info = {
            'variable': self.variable,
            'features': self.features,
            'config': self.config,
            'results': {name: result['metrics'] for name, result in self.results.items()},
            'best_model': min(self.results.items(), key=lambda x: x[1]['metrics']['RMSE'])[0] if self.results else None,
            'data_info': {
                'train_size': len(self.X_train),
                'test_size': len(self.X_test),
                'n_features': len(self.features)
            }
        }
        
        info_path = os.path.join(directory, f'model_info_{self.variable}.json')
        try:
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
            print(f"Informações salvas: {info_path}")
        except Exception as e:
            print(f"Erro ao salvar informações: {str(e)}")
        
        return saved_models
    
    def load_model(self, model_path, info_path=None):
        """Carrega modelo salvo"""
        try:
            model = joblib.load(model_path)
            print(f"Modelo carregado: {model_path}")
            
            if info_path and os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    info = json.load(f)
                self.features = info['features']
                self.config = info['config']
                print(f"Informações carregadas: {info_path}")
            
            return model
        except Exception as e:
            print(f"Erro ao carregar modelo: {str(e)}")
            return None
    
    def predict_new_data(self, new_data_path, model_name=None):
        """Faz predições em novos dados"""
        if not self.results:
            print("Nenhum modelo treinado disponível")
            return None
        
        if model_name is None or model_name not in self.results:
            model_name = min(self.results.items(), key=lambda x: x[1]['metrics']['RMSE'])[0]
        
        print(f"Usando modelo: {model_name}")
        
        try:
            new_data = pd.read_csv(new_data_path, parse_dates=['date'])
            print(f"Novos dados carregados: {new_data.shape}")
            
            missing_features = [f for f in self.features if f not in new_data.columns]
            if missing_features:
                print(f"Features faltantes: {missing_features}")
                return None
            
            X_new = new_data[self.features]
            
            model = self.results[model_name]['model']
            predictions = model.predict(X_new)
            
            if self.variable == 'precipitation':
                predictions = np.expm1(predictions)
            
            results_df = pd.DataFrame({
                'date': new_data['date'],
                f'{self.variable}_predicted': predictions
            })
            
            print(f"Predições concluídas: {len(predictions)} registros")
            return results_df
            
        except Exception as e:
            print(f"Erro ao fazer predições: {str(e)}")
            return None
    
    def _get_unit(self):
        """Retorna a unidade de medida"""
        return '°C' if self.variable == 'temperature' else 'mm'