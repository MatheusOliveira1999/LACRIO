"""
Sistema Completo de Downscaling Climático usando Machine Learning
Integra dados ERA5/ERA5-Land, estações in-situ e modelos de elevação
para melhorar dados climáticos em regiões montanhosas
"""

# Configuração do matplotlib ANTES de qualquer import do pyplot
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para evitar erros com Tkinter

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
import json
from datetime import datetime
from scipy.stats import gaussian_kde
from werkzeug.utils import secure_filename



# Flask imports
from flask import Flask, render_template, request, jsonify, send_file

# ML imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, 
                            VotingRegressor, GradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression

# XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost não disponível. Instale com: pip install xgboost")
    XGBOOST_AVAILABLE = False

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
        
        # Configurações padrão expandidas
        self.default_config = {
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 3,
            'optimize_hyperparams': True,
            'feature_selection': True,
            'max_features': 100,
            'ensemble_size': 3,
            # NOVAS configurações de otimização
            'fast_optimization': True,  # Usar grids reduzidos para economia de tempo
            'use_random_search': False,  # Usar RandomizedSearchCV ao invés de GridSearchCV
            'random_search_iter': 20,  # Número de iterações para RandomizedSearchCV
            'show_top_params': True,  # Mostrar top 3 combinações de parâmetros
            'verbose': True,  # Mostrar progresso da otimização
            'optimize_all_models': True,  # Otimizar TODOS os modelos, não apenas alguns
            'use_optuna': False  # Usar Optuna para otimização (requer instalação)
        }
        
        # Mesclar configurações
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def load_and_merge_data(self, era5_path, station_path, dem_path, lat, lon):
        """
        Carrega e mescla todos os dados necessários
        
        Parameters:
        -----------
        era5_path : str
            Caminho para dados ERA5/ERA5-Land
        station_path : str
            Caminho para dados da estação
        dem_path : str
            Caminho para modelo de elevação digital
        lat, lon : float
            Coordenadas da estação
        """
        try:
            print(f"Carregando dados para {self.variable}...")
            
            # Carregar ERA5
            era5_df = pd.read_csv(era5_path, parse_dates=['date'])
            print(f"ERA5 carregado: {era5_df.shape}")
            
            # Carregar estação
            station_df = pd.read_csv(station_path, parse_dates=['data'])
            print(f"Estação carregada: {station_df.shape}")
            
            # Padronizar nomes das colunas da estação
            station_df = self._standardize_station_columns(station_df)
            
            # Mesclar dados
            self.data = pd.merge(era5_df, station_df, on='date', how='inner')
            print(f"Dados mesclados: {self.data.shape}")
            
            if self.data.empty:
                raise ValueError("Nenhum dado encontrado após a mesclagem. Verifique as datas.")
            
            # Carregar e processar DEM
            self._process_elevation_data(dem_path, lat, lon)
            
            # Validar dados
            self._validate_data()
            
            print(f"Dados carregados com sucesso: {self.data.shape}")
            return self.data
            
        except Exception as e:
            print(f"Erro ao carregar dados: {str(e)}")
            raise
    
    def _standardize_station_columns(self, station_df):
        """Padroniza nomes das colunas da estação"""
        # Mapeamento de possíveis nomes de colunas
        column_mapping = {
            'data': 'date',
            'Data': 'date',
            'DATE': 'date',
            'Temperatura': 'temp_obs',
            'temperatura': 'temp_obs',
            'Temperature': 'temp_obs',
            'TEMP': 'temp_obs',
            'Precipitação': 'prec_obs',
            'precipitacao': 'prec_obs',
            'Precipitation': 'prec_obs',
            'PREC': 'prec_obs',
            'Rain': 'prec_obs'
        }
        
        # Renomear colunas
        station_df = station_df.rename(columns=column_mapping)
        
        # Garantir que temos as colunas necessárias
        if 'temp_obs' not in station_df.columns:
            station_df['temp_obs'] = np.nan
        if 'prec_obs' not in station_df.columns:
            station_df['prec_obs'] = 0.0
            
        return station_df
    
    def _process_elevation_data(self, dem_path, lat, lon):
        """Processa dados de elevação"""
        try:
            dem = xr.open_dataset(dem_path)
            
            # Buscar possíveis nomes para elevação
            elevation_vars = ['HGT', 'elevation', 'dem', 'z', 'height']
            elevation_var = None
            
            for var in elevation_vars:
                if var in dem.variables:
                    elevation_var = var
                    break
            
            if elevation_var is None:
                print("Variável de elevação não encontrada. Usando elevação padrão.")
                elev_local = 3000  # Elevação padrão para regiões montanhosas
            else:
                elev_local = float(dem[elevation_var].sel(lat=lat, lon=lon, method='nearest'))
            
            # Calcular diferença de altitude com ERA5
            if 'z' in self.data.columns:
                self.data['alt_ERA5'] = self.data['z'] / 9.80665  # Conversão geopotencial para metros
                self.data['alt_diff'] = elev_local - self.data['alt_ERA5']
            else:
                self.data['alt_diff'] = 0  # Se não tiver dados de geopotencial
            
            self.data['elevation'] = elev_local
            print(f"Elevação local: {elev_local:.1f}m")
            
        except Exception as e:
            print(f"Erro ao processar elevação: {str(e)}")
            self.data['alt_diff'] = 0
            self.data['elevation'] = 3000
    
    def _validate_data(self):
        """Valida os dados carregados"""
        # Verificar se temos dados suficientes
        min_records = 365  # Mínimo 1 ano de dados
        if len(self.data) < min_records:
            raise ValueError(f"Dados insuficientes: {len(self.data)} registros (mínimo: {min_records})")
        
        # Verificar variável alvo
        target_col = 'temp_obs' if self.variable == 'temperature' else 'prec_obs'
        if target_col not in self.data.columns:
            raise ValueError(f"Coluna alvo '{target_col}' não encontrada")
        
        # Verificar dados válidos
        valid_data = self.data[target_col].notna().sum()
        if valid_data < min_records:
            raise ValueError(f"Dados válidos insuficientes: {valid_data} registros")
        
        print(f"Validação concluída: {valid_data} registros válidos")
    
    def create_features(self):
        """Cria todas as features para o modelo"""
        print(f"Criando features para {self.variable}...")

        # Normalizar variáveis de escala muito grande
        large_scale_vars = ['sp', 'z', 'z_300.0', 'z_400.0', 'z_500.0', 'ssrd', 'strd']
        for var in large_scale_vars:
            if var in self.data.columns:
                # Converter pressão de Pa para hPa
                if var in ['sp', 'z', 'z_300.0', 'z_400.0', 'z_500.0']:
                    self.data[var] = self.data[var] / 100.0
                # Converter radiação acumulada para média
                elif var in ['ssrd', 'strd']:
                    self.data[var] = self.data[var] / 86400.0  # J/m² para W/m²
        
        # Features temporais
        self.data = self._create_temporal_features(self.data)
        
        # Features de lag
        self.data = self._create_lag_features(self.data)
        
        # Features de média móvel
        self.data = self._create_rolling_features(self.data)
        
        # Features de interação
        self.data = self._create_interaction_features(self.data)
        
        # Features estatísticas
        self.data = self._create_statistical_features(self.data)
        
        # Features específicas para regiões montanhosas
        self.data = self._create_topographic_features(self.data)
        
        # Remover NaN
        initial_shape = self.data.shape
        self.data = self.data.dropna()
        final_shape = self.data.shape
        
        print(f"Features criadas. Shape: {initial_shape} -> {final_shape}")
        print(f"Registros removidos por NaN: {initial_shape[0] - final_shape[0]}")
        
        return self.data
    
    def _create_temporal_features(self, df):
        """Cria features temporais"""
        df = df.copy()
        
        # Features básicas de tempo
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['season'] = df['month'].apply(lambda x: (x % 12) // 3 + 1)
        
        # Features cíclicas (importantes para capturar sazonalidade)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Tendência temporal
        start_date = df['date'].min()
        df['days_since_start'] = (df['date'] - start_date).dt.days
        
        return df
    
    def _create_lag_features(self, df):
        """Cria features de lag temporal"""
        df = df.copy()
        
        if self.variable == 'temperature':
            meteo_cols = ['t2m', 'tp', 'sp', 'u10', 'v10', 'd2m', 'r', 'ssrd', 'strd']
            lags = [1, 2, 3, 7, 14]
        else:  # precipitation
            meteo_cols = ['tp', 'tcwv', 'cape', 'cin', 'sp', 'u10', 'v10', 'q', 'pev']
            lags = [1, 2, 3, 7, 14, 30]
        
        # Filtrar colunas existentes
        meteo_cols = [col for col in meteo_cols if col in df.columns]
        
        for col in meteo_cols:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def _create_rolling_features(self, df):
        """Cria features de média móvel e estatísticas deslizantes"""
        df = df.copy()
        
        if self.variable == 'temperature':
            meteo_cols = ['t2m', 'tp', 'sp', 'u10', 'v10', 'd2m', 'r']
            windows = [3, 7, 14, 30]
        else:  # precipitation
            meteo_cols = ['tp', 'tcwv', 'cape', 'sp', 'u10', 'v10']
            windows = [3, 7, 14, 30, 60]
        
        # Filtrar colunas existentes
        meteo_cols = [col for col in meteo_cols if col in df.columns]
        
        for col in meteo_cols:
            for window in windows:
                # Média móvel
                df[f'{col}_ma_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                
                # Desvio padrão móvel
                df[f'{col}_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                
                # Mínimo e máximo móveis
                df[f'{col}_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                
                # Para precipitação, adicionar soma acumulada
                if self.variable == 'precipitation' and col == 'tp':
                    df[f'{col}_sum_{window}'] = df[col].rolling(window=window, min_periods=1).sum()
        
        return df
    
    def _create_interaction_features(self, df):
        """Cria features de interação entre variáveis"""
        df = df.copy()
        
        # Velocidade do vento (para todas as variáveis)
        if 'u10' in df.columns and 'v10' in df.columns:
            df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
            df['wind_direction'] = np.arctan2(df['v10'], df['u10'])
        
        # Interações específicas por variável
        if self.variable == 'temperature':
            if 't2m' in df.columns and 'r' in df.columns:
                df['t2m_r_interaction'] = df['t2m'] * df['r']
            if 'sp' in df.columns and 't2m' in df.columns:
                df['sp_t2m_interaction'] = df['sp'] * df['t2m']
            if 'd2m' in df.columns and 't2m' in df.columns:
                df['dewpoint_spread'] = df['t2m'] - df['d2m']
        else:  # precipitation
            if 'tcwv' in df.columns and 'cape' in df.columns:
                df['tcwv_cape_interaction'] = df['tcwv'] * df['cape']
            if 'q' in df.columns and 'sp' in df.columns:
                df['q_sp_interaction'] = df['q'] * df['sp']
        
        return df
    
    def _create_statistical_features(self, df):
        """Cria features estatísticas avançadas"""
        df = df.copy()
        
        # Selecionar variáveis meteorológicas principais
        if self.variable == 'temperature':
            main_vars = ['t2m', 'sp', 'r', 'd2m']
        else:
            main_vars = ['tp', 'tcwv', 'cape', 'sp']
        
        main_vars = [col for col in main_vars if col in df.columns]
        
        # Features estatísticas com janela de 30 dias
        for col in main_vars:
            window = 30
            rolling = df[col].rolling(window=window, min_periods=1)
            
            # Percentis
            df[f'{col}_p25_{window}'] = rolling.quantile(0.25)
            df[f'{col}_p75_{window}'] = rolling.quantile(0.75)
            
            # Skewness e kurtosis (quando possível)
            try:
                df[f'{col}_skew_{window}'] = rolling.skew()
            except:
                pass
        
        return df
    
    def _create_topographic_features(self, df):
        """Cria features específicas para regiões montanhosas"""
        df = df.copy()
        
        # Features baseadas na diferença de altitude
        if 'alt_diff' in df.columns:
            # Correção de temperatura baseada na altitude (lapse rate)
            if 't2m' in df.columns:
                lapse_rate = 0.0065  # K/m (taxa de diminuição padrão)
                df['t2m_corrected'] = df['t2m'] - (lapse_rate * df['alt_diff'])
            
            # Efeito orográfico na precipitação
            if 'tp' in df.columns and 'wind_speed' in df.columns:
                df['orographic_effect'] = df['alt_diff'] * df['wind_speed'] * df['tp']
        
        # Features sazonais específicas para montanhas
        if self.variable == 'temperature':
            # Amplitude térmica diária estimada (maior em altitudes elevadas)
            if 'ssrd' in df.columns:  # Radiação solar
                df['thermal_amplitude'] = df['ssrd'] * 0.1  # Simplificado
        
        return df
    
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
            # Log transform para precipitação (melhor distribuição)
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
        
        # Usar SelectKBest com f_regression
        self.feature_selector = SelectKBest(score_func=f_regression, k=max_features)
        
        # Ajustar e transformar dados de treino
        X_train_selected = self.feature_selector.fit_transform(self.X_train, self.y_train)
        X_test_selected = self.feature_selector.transform(self.X_test)
        
        # Obter nomes das features selecionadas
        selected_features = [self.features[i] for i in self.feature_selector.get_support(indices=True)]
        
        # Atualizar dados
        self.X_train = pd.DataFrame(X_train_selected, columns=selected_features, index=self.X_train.index)
        self.X_test = pd.DataFrame(X_test_selected, columns=selected_features, index=self.X_test.index)
        self.features = selected_features
        
        print(f"Features selecionadas: {len(self.features)}")
    
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
            ('model', Ridge(alpha=1.0, random_state=self.config['random_state']))
        ])
        
        models['RandomForest'] = Pipeline([
            ('model', RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=5,
                max_depth=None,
                random_state=self.config['random_state'],
                n_jobs=-1
            ))
        ])
        
        models['ExtraTrees'] = Pipeline([
            ('model', ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=5,
                random_state=self.config['random_state'],
                n_jobs=-1
            ))
        ])
        
        models['GradientBoosting'] = Pipeline([
            ('model', GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config['random_state']
            ))
        ])
        
        models['SVR'] = Pipeline([
            ('scaler', RobustScaler()),
            ('model', SVR(kernel='rbf', C=10, gamma='scale'))
        ])
        
        models['MLP'] = Pipeline([
            ('scaler', StandardScaler()),  # IMPORTANTE: Normalização obrigatória
            ('model', MLPRegressor(
                hidden_layer_sizes=(100, 50),  # Arquitetura mais simples
                max_iter=5000,  # Aumentar significativamente as iterações
                activation='relu',
                solver='adam',  # Adam geralmente converge melhor
                alpha=0.001,  # Regularização
                learning_rate='adaptive',  # Taxa adaptativa
                learning_rate_init=0.001,
                early_stopping=True,  # Parar quando não melhorar
                validation_fraction=0.15,  # 15% para validação
                n_iter_no_change=50,  # Paciência para early stopping
                random_state=self.config['random_state'],
                verbose=False  # Desativar output verboso
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
                        random_state=self.config['random_state'],
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
                        random_state=self.config['random_state'],
                        n_jobs=-1
                    ))
                ])
        
        return models
    
    def train_models(self, selected_models=None, optimize=None):
        """Treina todos os modelos selecionados"""
        if optimize is None:
            optimize = self.config['optimize_hyperparams']
        
        print(f"\nTreinando modelos para {self.variable}...")
        
        all_models = self.get_models()
        

        # Pré-processar dados se MLP estiver na lista
        if 'MLP' in self.models:
            print("Aplicando pré-processamento robusto para MLP...")
            self.X_train_robust = self._robust_preprocessing(self.X_train, fit=True)
            self.X_test_robust = self._robust_preprocessing(self.X_test, fit=False)
        # Filtrar modelos selecionados
        if selected_models:
            self.models = {k: v for k, v in all_models.items() if k in selected_models}
        else:
            self.models = all_models
        
        print(f"Modelos a treinar: {list(self.models.keys())}")
        
        # Lista de modelos para otimizar
        if self.config['optimize_all_models']:
            models_to_optimize = list(self.models.keys())
        else:
            # Lista original (apenas alguns modelos)
            models_to_optimize = ['RandomForest', 'XGBoost', 'Ridge', 'GradientBoosting', 
                                'ExtraTrees', 'SVR', 'MLP']
        
        for name, pipeline in self.models.items():
            print(f"\n{name}...")
            
            try:
                # Otimização de hiperparâmetros
                if optimize and name in models_to_optimize:
                    if self.config.get('use_optuna', False):
                        pipeline = self._optimize_model_optuna(name, pipeline)
                    else:
                        pipeline = self._optimize_model(name, pipeline)
                else:
                    print(f"  Usando parâmetros padrão")

                # Usar dados pré-processados para MLP
                if name == 'MLP' and hasattr(self, 'X_train_robust'):
                    X_train_temp = self.X_train_robust
                    X_test_temp = self.X_test_robust
                else:
                    X_train_temp = self.X_train
                    X_test_temp = self.X_test
                                
                # Treinar modelo
                print(f"  Treinando modelo final...")
                pipeline.fit(self.X_train, self.y_train)
                
                # Fazer predições
                y_pred = pipeline.predict(self.X_test)
                
                # Reverter transformação para precipitação
                if self.variable == 'precipitation':
                    y_pred = np.expm1(y_pred)
                    y_test_original = np.expm1(self.y_test)
                else:
                    y_test_original = self.y_test
                
                # Calcular métricas
                metrics = self.calculate_metrics(y_test_original, y_pred)
                
                # Armazenar resultados
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

    
    def _optimize_model(self, name, pipeline):
        """Otimiza hiperparâmetros do modelo"""
        print(f"  Otimizando hiperparâmetros para {name}...")
        
        # Grids de hiperparâmetros para TODOS os modelos
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
            
            'Lasso': {
                'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                'model__fit_intercept': [True, False],
                'model__max_iter': [1000, 2000, 5000],
                'model__selection': ['cyclic', 'random']
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
                'model__min_samples_leaf': [1, 3, 5, 10],
                'model__max_features': ['auto', 'sqrt', 'log2', 0.5]
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
                'model__epsilon': [0.01, 0.1, 0.2, 0.5],
                'model__degree': [2, 3, 4]  # Para kernel poly
            },
            
            'MLP': {
            'model__hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 75)],
            'model__activation': ['relu', 'tanh'],
            'model__solver': ['adam', 'lbfgs'],
            'model__alpha': [0.0001, 0.001, 0.01],
            'model__learning_rate': ['adaptive'],
            'model__learning_rate_init': [0.001, 0.01],
            'model__max_iter': [2000, 5000],
            'model__early_stopping': [True],
            'model__validation_fraction': [0.15]
            },
            
            'DecisionTree': {
                'model__max_depth': [None, 5, 10, 15, 20, 30],
                'model__min_samples_split': [2, 5, 10, 20, 50],
                'model__min_samples_leaf': [1, 2, 5, 10, 20],
                'model__max_features': ['auto', 'sqrt', 'log2', None],
                'model__criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                'model__splitter': ['best', 'random']
            },
            
            'Ensemble': {
                'weights': [None,
                        [1, 1, 1],
                        [2, 1, 1],
                        [1, 2, 1],
                        [1, 1, 2],
                        [3, 2, 1],
                        [1, 2, 3]]
            }
        }
        
        param_grids_fast = {
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
        
        if name not in param_grids:
            print(f"  Sem parâmetros para otimizar em {name}")
            return pipeline
        
        use_fast_grid = self.config.get('fast_optimization', True)
        
        if use_fast_grid and name in param_grids_fast:
            selected_grid = param_grids_fast[name]
        else:
            selected_grid = param_grids[name]
        
        ts_cv = TimeSeriesSplit(n_splits=self.config.get('cv_folds', 3))
        
        grid_search_params = {
            'estimator': pipeline,
            'param_grid': selected_grid,
            'cv': ts_cv,
            'scoring': 'neg_mean_squared_error',
            'n_jobs': -1,
            'verbose': 1 if self.config.get('verbose', False) else 0,
            'return_train_score': False
        }
        
        try:
            use_random_search = self.config.get('use_random_search', False)
            n_iter = self.config.get('random_search_iter', 20)
            
            # ---- INÍCIO DA CORREÇÃO ----
            
            # Adicionado 'SVR' à lista de modelos elegíveis para RandomSearch
            models_for_random_search = ['RandomForest', 'ExtraTrees', 'GradientBoosting', 'XGBoost', 'MLP', 'SVR']

            if use_random_search and name in models_for_random_search:
                from sklearn.model_selection import RandomizedSearchCV
                # Mensagem de log corrigida
                print(f"  Usando RandomizedSearchCV com {n_iter} iterações (grid {'rápido' if use_fast_grid else 'completo'})")
                
                random_search_params = grid_search_params.copy()
                random_search_params['param_distributions'] = random_search_params.pop('param_grid')
                
                search = RandomizedSearchCV(
                    **random_search_params,
                    n_iter=n_iter,
                    random_state=self.config['random_state']
                )
            else:
                from sklearn.model_selection import GridSearchCV
                n_combinations = 1
                for param, values in selected_grid.items():
                    n_combinations *= len(values)
                
                # Mensagem de log corrigida
                print(f"  Usando GridSearchCV (busca completa com {n_combinations} combinações)")
                
                search = GridSearchCV(**grid_search_params)
            
            # Executar busca
            print(f"  Iniciando otimização...")
            search.fit(self.X_train, self.y_train)
            
            # Resultados
            print(f"  ✓ Melhores parâmetros: {search.best_params_}")
            print(f"  ✓ Melhor score CV: {-search.best_score_:.4f}")
            
            # Opcional: mostrar top 3 combinações
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



    def _robust_preprocessing(self, X, fit=True):
        """
        Pré-processamento robusto para evitar problemas de convergência
        """
        from sklearn.preprocessing import RobustScaler, PowerTransformer
        
        if fit:
            # Criar scalers se não existirem
            self.robust_scaler = RobustScaler()
            self.power_transformer = PowerTransformer(method='yeo-johnson')
            
            # Identificar colunas com distribuição muito assimétrica
            skewed_features = []
            for col in X.columns:
                skewness = X[col].skew()
                if abs(skewness) > 2:
                    skewed_features.append(col)
            
            self.skewed_features = skewed_features
            
            # Aplicar transformações
            X_processed = X.copy()
            
            # Power transform para features assimétricas
            if skewed_features:
                X_processed[skewed_features] = self.power_transformer.fit_transform(X[skewed_features])
            
            # Robust scaling para todas as features
            X_processed = pd.DataFrame(
                self.robust_scaler.fit_transform(X_processed),
                columns=X.columns,
                index=X.index
            )
        else:
            # Apenas transformar
            X_processed = X.copy()
            
            if hasattr(self, 'skewed_features') and self.skewed_features:
                X_processed[self.skewed_features] = self.power_transformer.transform(X[self.skewed_features])
            
            X_processed = pd.DataFrame(
                self.robust_scaler.transform(X_processed),
                columns=X.columns,
                index=X.index
            )
        
        return X_processed
    
    def _optimize_model_optuna(self, name, pipeline):
        """Otimização avançada usando Optuna (biblioteca externa)"""
        try:
            import optuna
            from optuna.samplers import TPESampler
            print(f"  Otimizando {name} com Optuna...")
            
            def objective(trial):
                # Definir espaço de busca para cada modelo
                if name == 'RandomForest':
                    params = {
                        'model__n_estimators': trial.suggest_int('n_estimators', 50, 500),
                        'model__max_depth': trial.suggest_int('max_depth', 5, 50),
                        'model__min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'model__min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'model__max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
                    }
                elif name == 'XGBoost':
                    params = {
                        'model__n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'model__max_depth': trial.suggest_int('max_depth', 3, 15),
                        'model__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'model__subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'model__colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                    }
                # ... adicionar outros modelos conforme necessário
                else:
                    return 0
                
                # Configurar pipeline com parâmetros
                pipeline.set_params(**params)
                
                # Validação cruzada
                ts_cv = TimeSeriesSplit(n_splits=3)
                scores = []
                
                for train_idx, val_idx in ts_cv.split(self.X_train):
                    X_train_cv = self.X_train.iloc[train_idx]
                    y_train_cv = self.y_train.iloc[train_idx]
                    X_val_cv = self.X_train.iloc[val_idx]
                    y_val_cv = self.y_train.iloc[val_idx]
                    
                    pipeline.fit(X_train_cv, y_train_cv)
                    y_pred = pipeline.predict(X_val_cv)
                    mse = mean_squared_error(y_val_cv, y_pred)
                    scores.append(mse)
                
                return np.mean(scores)
            
            # Criar estudo Optuna
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=self.config['random_state'])
            )
            
            # Otimizar
            study.optimize(objective, n_trials=50, show_progress_bar=True)
            
            # Aplicar melhores parâmetros
            best_params = {'model__' + k: v for k, v in study.best_params.items()}
            pipeline.set_params(**best_params)
            pipeline.fit(self.X_train, self.y_train)
            
            print(f"  ✓ Melhores parâmetros (Optuna): {study.best_params}")
            print(f"  ✓ Melhor score: {study.best_value:.4f}")
            
            return pipeline
            
        except ImportError:
            print("  Optuna não instalado. Use: pip install optuna")
            return self._optimize_model(name, pipeline)  # Fallback para GridSearch
       
    def calculate_metrics(self, y_true, y_pred):
        """Calcula métricas de avaliação"""
        # Métricas básicas
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        
        # Skill Score
        ss_clim = 1 - (rmse**2 / np.var(y_true))
        
        # Correlação
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Bias': bias,
            'Skill_Score': ss_clim,
            'Correlation': corr
        }
        
        # Métricas específicas para precipitação
        if self.variable == 'precipitation':
            # Detecção de eventos de chuva
            rain_threshold = 0.1
            rain_true = y_true > rain_threshold
            rain_pred = y_pred > rain_threshold
            
            if len(rain_true) > 0:
                rain_accuracy = np.mean(rain_true == rain_pred) * 100
                
                # Probabilidade de detecção (POD)
                if np.sum(rain_true) > 0:
                    pod = np.sum(rain_true & rain_pred) / np.sum(rain_true)
                else:
                    pod = np.nan
                
                # Taxa de falso alarme (FAR)
                if np.sum(rain_pred) > 0:
                    far = np.sum(~rain_true & rain_pred) / np.sum(rain_pred)
                else:
                    far = np.nan
                
                metrics.update({
                    'Rain_Detection_Accuracy': rain_accuracy,
                    'POD': pod,
                    'FAR': far
                })
            
            # R² apenas para dias com chuva
            if np.sum(rain_true) > 10:
                rain_days_mask = rain_true
                r2_rain = r2_score(y_true[rain_days_mask], y_pred[rain_days_mask])
                metrics['R2_Rain_Days'] = r2_rain
        
        return metrics
    def diagnose_convergence_issues(self):
        """
        Diagnostica problemas de convergência
        """
        print("\n🔍 DIAGNÓSTICO DE CONVERGÊNCIA")
        print("="*50)
        
        # Verificar escala dos dados
        print("\n📊 Estatísticas dos dados de treino:")
        for col in self.X_train.columns[:10]:  # Primeiras 10 features
            mean = self.X_train[col].mean()
            std = self.X_train[col].std()
            min_val = self.X_train[col].min()
            max_val = self.X_train[col].max()
            print(f"{col:30s} | Mean: {mean:10.2f} | Std: {std:10.2f} | Range: [{min_val:.2f}, {max_val:.2f}]")
        
        # Verificar target
        print(f"\n🎯 Variável alvo ({self.variable}):")
        print(f"   Mean: {self.y_train.mean():.2f}")
        print(f"   Std: {self.y_train.std():.2f}")
        print(f"   Range: [{self.y_train.min():.2f}, {self.y_train.max():.2f}]")
        
        # Verificar correlações
        print("\n🔗 Top 10 correlações com a variável alvo:")
        correlations = self.X_train.corrwith(self.y_train).abs().sort_values(ascending=False)
        for feat, corr in correlations.head(10).items():
            print(f"   {feat:30s}: {corr:.3f}")
        
        print("="*50)

    def create_ensemble(self, ensemble_size=None):
        """Cria modelo ensemble dos melhores modelos"""
        if ensemble_size is None:
            ensemble_size = self.config['ensemble_size']
        
        print(f"\nCriando modelo ensemble com {ensemble_size} modelos...")
        
        if len(self.results) < 2:
            print("Insufficient models for ensemble creation")
            return None
        
        # Selecionar os melhores modelos baseado no RMSE
        best_models = sorted(self.results.items(), 
                           key=lambda x: x[1]['metrics']['RMSE'])[:ensemble_size]
        
        print(f"Modelos selecionados: {[name for name, _ in best_models]}")
        
        # Criar ensemble
        ensemble_models = [(name, result['model']) for name, result in best_models]
        ensemble = VotingRegressor(ensemble_models)
        
        # Treinar ensemble
        ensemble.fit(self.X_train, self.y_train)
        
        # Avaliar ensemble
        y_pred = ensemble.predict(self.X_test)
        
        # Reverter transformação para precipitação
        if self.variable == 'precipitation':
            y_pred = np.expm1(y_pred)
            y_test_original = np.expm1(self.y_test)
        else:
            y_test_original = self.y_test
        
        # Calcular métricas
        metrics = self.calculate_metrics(y_test_original, y_pred)
        
        # Armazenar resultado
        self.results['Ensemble'] = {
            'model': ensemble,
            'predictions': y_pred,
            'metrics': metrics
        }
        
        print(f"Ensemble RMSE: {metrics['RMSE']:.4f}")
        print(f"Ensemble R²: {metrics['R2']:.4f}")
        
        return ensemble
    
    def plot_results(self, save_plots=True):
        """Gera visualizações dos resultados"""
        if not self.results:
            print("Nenhum resultado para plotar")
            return
        
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Resultados do Downscaling - {self.variable.capitalize()}', fontsize=16, fontweight='bold')
        
        # 1. Comparação de modelos
        ax = axes[0, 0]
        models = list(self.results.keys())
        rmse_values = [self.results[m]['metrics']['RMSE'] for m in models]
        r2_values = [self.results[m]['metrics']['R2'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.8, color='skyblue')
        ax.set_ylabel('RMSE', fontweight='bold')
        ax.set_xlabel('Modelos', fontweight='bold')
        ax.set_title('Comparação de Modelos', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars1, rmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Eixo secundário para R²
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, r2_values, width, label='R²', alpha=0.8, color='lightcoral')
        ax2.set_ylabel('R²', fontweight='bold')
        ax2.set_ylim(0, 1)
        
        # Adicionar valores nas barras R²
        for bar, value in zip(bars2, r2_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Legendas
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # 2. Melhor modelo - Observado vs Predito
        ax = axes[0, 1]
        best_model_name = min(self.results.items(), 
                             key=lambda x: x[1]['metrics']['RMSE'])[0]
        y_pred = self.results[best_model_name]['predictions']
        
        if self.variable == 'precipitation':
            y_true = np.expm1(self.y_test)
        else:
            y_true = self.y_test
        
        # Scatter plot com densidade de cores
        scatter = ax.scatter(y_true, y_pred, alpha=0.6, s=20, c='blue', edgecolors='none')
        
        # Linha 1:1
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
        
        # Estatísticas no gráfico
        r2 = self.results[best_model_name]['metrics']['R2']
        rmse = self.results[best_model_name]['metrics']['RMSE']
        ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'Observado ({self._get_unit()})', fontweight='bold')
        ax.set_ylabel(f'Predito ({self._get_unit()})', fontweight='bold')
        ax.set_title(f'{best_model_name} - Observado vs Predito', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. Série temporal
        ax = axes[1, 0]
        n_points = min(365, len(self.test_dates))  # Mostrar no máximo 1 ano
        indices = np.linspace(0, len(self.test_dates)-1, n_points, dtype=int)
        
        dates_subset = self.test_dates.iloc[indices]
        y_true_subset = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_subset = y_pred[indices]
        
        ax.plot(dates_subset, y_true_subset, 'b-', alpha=0.7, label='Observado', linewidth=1.5)
        ax.plot(dates_subset, y_pred_subset, 'r-', alpha=0.7, label='Predito', linewidth=1.5)
        ax.set_xlabel('Data', fontweight='bold')
        ax.set_ylabel(f'{self.variable.capitalize()} ({self._get_unit()})', fontweight='bold')
        ax.set_title('Série Temporal (Subset)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotacionar datas
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Distribuição dos resíduos
        ax = axes[1, 1]
        residuals = y_true - y_pred
        
        # Histograma
        n_bins = min(30, len(residuals)//20)
        ax.hist(residuals, bins=n_bins, edgecolor='black', alpha=0.7, color='lightgreen')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(x=np.mean(residuals), color='orange', linestyle='-', linewidth=2, label='Média')
        
        ax.set_xlabel(f'Resíduos ({self._get_unit()})', fontweight='bold')
        ax.set_ylabel('Frequência', fontweight='bold')
        ax.set_title('Distribuição dos Resíduos', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Estatísticas dos resíduos
        bias = np.mean(residuals)
        std_res = np.std(residuals)
        ax.text(0.05, 0.95, f'Bias = {bias:.3f}\nStd = {std_res:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            output_dir = os.path.join('static', 'img')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = os.path.join(output_dir, f'resultados_downscaling_{self.variable}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo: {filename}")

        
        plt.close()
    
    def plot_feature_importance(self, model_name='RandomForest', top_n=20, save_plot=True):
        """Plota importância das features"""
        if model_name not in self.results:
            available_models = [name for name in self.results.keys() 
                              if 'RandomForest' in name or 'ExtraTrees' in name or 'XGBoost' in name]
            if available_models:
                model_name = available_models[0]
            else:
                print("Nenhum modelo baseado em árvore disponível")
                return
        
        model = self.results[model_name]['model']
        
        # Extrair importâncias
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            importances = model.named_steps['model'].feature_importances_
        else:
            print(f"Modelo {model_name} não possui feature_importances_")
            return
        
        # Criar DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plotar
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(feature_importance)), feature_importance['importance'], 
                       color='steelblue', alpha=0.8)
        
        # Personalizar gráfico
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Importância', fontweight='bold', fontsize=12)
        plt.title(f'Top {top_n} Features Mais Importantes - {model_name}\n{self.variable.capitalize()}', 
                 fontweight='bold', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        # Adicionar valores nas barras
        for i, (bar, value) in enumerate(zip(bars, feature_importance['importance'])):
            plt.text(value + value*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        
        if save_plot:
            output_dir = os.path.join('static', 'img')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = os.path.join(output_dir, f'feature_importance_{self.variable}_{model_name.lower()}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo: {filename}")
        plt.close()
        
        return feature_importance
    
    def plot_temporal_analysis(self, save_plot=True):
        """Análise temporal detalhada"""
        if not self.results:
            print("Nenhum resultado disponível")
            return
        
        # Melhor modelo
        best_model_name = min(self.results.items(), 
                             key=lambda x: x[1]['metrics']['RMSE'])[0]
        y_pred = self.results[best_model_name]['predictions']
        
        if self.variable == 'precipitation':
            y_true = np.expm1(self.y_test)
        else:
            y_true = self.y_test
        
        # Criar DataFrame para análise
        temporal_df = pd.DataFrame({
            'date': self.test_dates.values,
            'observed': y_true.values if hasattr(y_true, 'values') else y_true,
            'predicted': y_pred,
            'residual': (y_true.values if hasattr(y_true, 'values') else y_true) - y_pred
        })
        
        temporal_df['month'] = pd.to_datetime(temporal_df['date']).dt.month
        temporal_df['year'] = pd.to_datetime(temporal_df['date']).dt.year
        temporal_df['season'] = temporal_df['month'].apply(lambda x: (x % 12) // 3 + 1)
        
        # Análise mensal
        monthly_stats = temporal_df.groupby('month').agg({
            'observed': ['mean', 'std'],
            'predicted': ['mean', 'std'],
            'residual': ['mean', 'std']
        }).round(3)
        
        # Análise sazonal
        seasonal_stats = temporal_df.groupby('season').agg({
            'observed': ['mean', 'std'],
            'predicted': ['mean', 'std'],
            'residual': ['mean', 'std']
        }).round(3)
        
        # Plotar
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Análise Temporal - {best_model_name} - {self.variable.capitalize()}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Médias mensais
        ax = axes[0, 0]
        months = range(1, 13)
        month_names = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                      'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        
        obs_means = [monthly_stats.loc[m, ('observed', 'mean')] for m in months]
        pred_means = [monthly_stats.loc[m, ('predicted', 'mean')] for m in months]
        obs_stds = [monthly_stats.loc[m, ('observed', 'std')] for m in months]
        pred_stds = [monthly_stats.loc[m, ('predicted', 'std')] for m in months]
        
        ax.plot(months, obs_means, 'o-', label='Observado', linewidth=2, markersize=6, color='blue')
        ax.plot(months, pred_means, 's-', label='Predito', linewidth=2, markersize=6, color='red')
        
        # Barras de erro
        ax.fill_between(months, 
                       np.array(obs_means) - np.array(obs_stds),
                       np.array(obs_means) + np.array(obs_stds),
                       alpha=0.2, color='blue')
        ax.fill_between(months, 
                       np.array(pred_means) - np.array(pred_stds),
                       np.array(pred_means) + np.array(pred_stds),
                       alpha=0.2, color='red')
        
        ax.set_xlabel('Mês', fontweight='bold')
        ax.set_ylabel(f'{self.variable.capitalize()} ({self._get_unit()})', fontweight='bold')
        ax.set_title('Variação Mensal', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(months)
        ax.set_xticklabels(month_names)
        
        # 2. Erro médio mensal
        ax = axes[0, 1]
        residual_means = [monthly_stats.loc[m, ('residual', 'mean')] for m in months]
        residual_stds = [monthly_stats.loc[m, ('residual', 'std')] for m in months]
        
        bars = ax.bar(months, residual_means, yerr=residual_stds, 
                     capsize=5, alpha=0.7, color='lightcoral')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Mês', fontweight='bold')
        ax.set_ylabel(f'Erro Médio ({self._get_unit()})', fontweight='bold')
        ax.set_title('Erro Médio Mensal', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(months)
        ax.set_xticklabels(month_names)
        
        # 3. Análise sazonal
        ax = axes[1, 0]
        seasons = [1, 2, 3, 4]
        season_names = ['Verão', 'Outono', 'Inverno', 'Primavera']
        
        obs_seasonal = [seasonal_stats.loc[s, ('observed', 'mean')] for s in seasons]
        pred_seasonal = [seasonal_stats.loc[s, ('predicted', 'mean')] for s in seasons]
        
        x = np.arange(len(seasons))
        width = 0.35
        
        ax.bar(x - width/2, obs_seasonal, width, label='Observado', alpha=0.8, color='blue')
        ax.bar(x + width/2, pred_seasonal, width, label='Predito', alpha=0.8, color='red')
        
        ax.set_xlabel('Estação', fontweight='bold')
        ax.set_ylabel(f'{self.variable.capitalize()} ({self._get_unit()})', fontweight='bold')
        ax.set_title('Variação Sazonal', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(season_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Boxplot dos resíduos por mês
        ax = axes[1, 1]
        monthly_residuals = [temporal_df[temporal_df['month'] == m]['residual'].values 
                           for m in months]
        
        bp = ax.boxplot(monthly_residuals, labels=month_names, patch_artist=True)
        
        # Colorir boxplots
        colors = plt.cm.viridis(np.linspace(0, 1, 12))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Mês', fontweight='bold')
        ax.set_ylabel(f'Resíduos ({self._get_unit()})', fontweight='bold')
        ax.set_title('Distribuição Mensal dos Resíduos', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = os.path.join('static', 'img')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = os.path.join(output_dir, f'analise_temporal_{self.variable}_{best_model_name.lower()}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo: {filename}")
            plt.close()

        
        return monthly_stats, seasonal_stats
    
    def save_models(self, directory='models'):
        """Salva todos os modelos treinados"""
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
        
        # Salvar informações do modelo
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
    
    def generate_report(self):
            """Gera relatório completo dos resultados para o terminal e para um arquivo .txt"""
            if not self.results:
                print("Nenhum resultado disponível para relatório")
                return

            # --- INÍCIO DA CORREÇÃO ---
            # 1. Construir todo o conteúdo do relatório em uma lista de strings
            report_lines = []

            report_lines.append("="*80)
            report_lines.append(f"RELATÓRIO DE DOWNSCALING CLIMÁTICO - {self.variable.upper()}")
            report_lines.append("="*80)
            report_lines.append(f"Data de geração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Informações dos dados
            report_lines.append("\n📊 INFORMAÇÕES DOS DADOS:")
            report_lines.append(f"   • Total de registros: {len(self.data)}")
            report_lines.append(f"   • Features utilizadas: {len(self.features)}")
            report_lines.append(f"   • Dados de treino: {len(self.X_train)}")
            report_lines.append(f"   • Dados de teste: {len(self.X_test)}")
            report_lines.append(f"   • Período de teste: {self.test_dates.min().strftime('%Y-%m-%d')} a {self.test_dates.max().strftime('%Y-%m-%d')}")

            # Configurações
            report_lines.append("\n⚙️ CONFIGURAÇÕES:")
            for key, value in self.config.items():
                report_lines.append(f"   • {key}: {value}")

            # Resultados dos modelos
            report_lines.append("\n🎯 RESULTADOS DOS MODELOS:")
            report_lines.append("-"*60)

            sorted_results = sorted(self.results.items(), key=lambda x: x[1]['metrics']['RMSE'])

            for i, (name, result) in enumerate(sorted_results, 1):
                metrics = result['metrics']
                report_lines.append(f"\n{i}. {name}")
                report_lines.append(f"   ├─ RMSE: {metrics['RMSE']:.4f} {self._get_unit()}")
                report_lines.append(f"   ├─ MAE: {metrics['MAE']:.4f} {self._get_unit()}")
                report_lines.append(f"   ├─ R²: {metrics['R2']:.4f}")
                report_lines.append(f"   ├─ Correlação: {metrics['Correlation']:.4f}")
                report_lines.append(f"   ├─ Bias: {metrics['Bias']:.4f} {self._get_unit()}")
                report_lines.append(f"   └─ Skill Score: {metrics['Skill_Score']:.4f}")

                if self.variable == 'precipitation':
                    prec_metrics = []
                    if 'Rain_Detection_Accuracy' in metrics:
                        prec_metrics.append(f"   ├─ Detecção de Chuva: {metrics['Rain_Detection_Accuracy']:.1f}%")
                    if 'POD' in metrics and not np.isnan(metrics['POD']):
                        prec_metrics.append(f"   ├─ POD: {metrics['POD']:.3f}")
                    if 'FAR' in metrics and not np.isnan(metrics['FAR']):
                        prec_metrics.append(f"   ├─ FAR: {metrics['FAR']:.3f}")
                    if 'R2_Rain_Days' in metrics and not np.isnan(metrics['R2_Rain_Days']):
                        prec_metrics.append(f"   └─ R² (dias com chuva): {metrics['R2_Rain_Days']:.4f}")
                    
                    # Substitui o último '├─' por '└─' se houver métricas de precipitação
                    if prec_metrics and "└─ Skill Score" in report_lines[-1]:
                        report_lines[-1] = report_lines[-1].replace('└─', '├─')
                    
                    report_lines.extend(prec_metrics)

            # Melhor modelo
            best_model = sorted_results[0]
            report_lines.append(f"\n🏆 MELHOR MODELO: {best_model[0]}")
            report_lines.append(f"   • RMSE: {best_model[1]['metrics']['RMSE']:.4f} {self._get_unit()}")
            report_lines.append(f"   • R²: {best_model[1]['metrics']['R2']:.4f}")
            report_lines.append(f"   • Melhoria sobre climatologia: {best_model[1]['metrics']['Skill_Score']:.1%}")

            # Interpretação dos resultados
            report_lines.append(f"\n📈 INTERPRETAÇÃO:")
            r2 = best_model[1]['metrics']['R2']
            if r2 >= 0.8:
                quality = "Excelente"
            elif r2 >= 0.6:
                quality = "Boa"
            elif r2 >= 0.4:
                quality = "Moderada"
            else:
                quality = "Baixa"

            report_lines.append(f"   • Qualidade da predição: {quality} (R² = {r2:.3f})")

            skill = best_model[1]['metrics']['Skill_Score']
            if skill > 0:
                report_lines.append(f"   • O modelo é {skill:.1%} melhor que a climatologia")
            else:
                report_lines.append(f"   • O modelo não supera a climatologia")
            
            report_lines.append("\n" + "="*80)

            # 2. Juntar as linhas em uma única string de texto
            full_report_text = "\n".join(report_lines)

            # 3. Imprimir o relatório completo no terminal
            print(full_report_text)

            # 4. Salvar o mesmo relatório completo em um arquivo
            output_dir = 'results'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            report_path = os.path.join(output_dir, f'relatorio_downscaling_{self.variable}.txt')
            
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(full_report_text)
                
                print(f"\n📄 Relatório salvo em: {report_path}")

            except Exception as e:
                print(f"Erro ao salvar relatório: {str(e)}")
        # --- FIM DA CORREÇÃO ---
    
    def _get_unit(self):
        """Retorna a unidade de medida"""
        return '°C' if self.variable == 'temperature' else 'mm'
    
    def predict_new_data(self, new_data_path, model_name=None):
        """Faz predições em novos dados"""
        if not self.results:
            print("Nenhum modelo treinado disponível")
            return None
        
        # Selecionar modelo
        if model_name is None or model_name not in self.results:
            model_name = min(self.results.items(), key=lambda x: x[1]['metrics']['RMSE'])[0]
        
        print(f"Usando modelo: {model_name}")
        
        try:
            # Carregar novos dados
            new_data = pd.read_csv(new_data_path, parse_dates=['date'])
            print(f"Novos dados carregados: {new_data.shape}")
            
            # Verificar se temos as features necessárias
            missing_features = [f for f in self.features if f not in new_data.columns]
            if missing_features:
                print(f"Features faltantes: {missing_features}")
                return None
            
            # Preparar dados
            X_new = new_data[self.features]
            
            # Fazer predições
            model = self.results[model_name]['model']
            predictions = model.predict(X_new)
            
            # Reverter transformação para precipitação
            if self.variable == 'precipitation':
                predictions = np.expm1(predictions)
            
            # Criar DataFrame com resultados
            results_df = pd.DataFrame({
                'date': new_data['date'],
                f'{self.variable}_predicted': predictions
            })
            
            print(f"Predições concluídas: {len(predictions)} registros")
            return results_df
            
        except Exception as e:
            print(f"Erro ao fazer predições: {str(e)}")
            return None

# ADICIONE ESTAS FUNÇÕES APÓS A DEFINIÇÃO DA CLASSE WeatherDownscalingModel
# Mas ANTES da seção "# Flask Web Application"

def plot_all_models_results(self, save_plots=True):
    """Gera visualizações para TODOS os modelos treinados"""
    if not self.results:
        print("Nenhum resultado para plotar")
        return
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Para cada modelo treinado
    for model_name, model_data in self.results.items():
        print(f"\nGerando gráficos para {model_name}...")
        
        # Criar figura com subplots para este modelo
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Resultados do Downscaling - {model_name} - {self.variable.capitalize()}', 
                     fontsize=16, fontweight='bold')
        
        y_pred = model_data['predictions']
        metrics = model_data['metrics']
        
        if self.variable == 'precipitation':
            y_true = np.expm1(self.y_test)
        else:
            y_true = self.y_test
        
        # 1. Observado vs Predito
        ax = axes[0, 0]
        scatter = ax.scatter(y_true, y_pred, alpha=0.6, s=20, c='blue', edgecolors='none')
        
        # Linha 1:1
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
        
        # Estatísticas
        r2 = metrics['R2']
        rmse = metrics['RMSE']
        mae = metrics['MAE']
        bias = metrics['Bias']
        
        stats_text = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nBias = {bias:.3f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'Observado ({self._get_unit()})', fontweight='bold')
        ax.set_ylabel(f'Predito ({self._get_unit()})', fontweight='bold')
        ax.set_title('Observado vs Predito', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Série temporal (últimos 90 dias)
        ax = axes[0, 1]
        n_points = min(90, len(self.test_dates))
        indices = np.arange(len(self.test_dates) - n_points, len(self.test_dates))
        
        dates_subset = self.test_dates.iloc[indices]
        y_true_subset = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_subset = y_pred[indices]
        
        ax.plot(dates_subset, y_true_subset, 'b-', alpha=0.7, label='Observado', linewidth=1.5)
        ax.plot(dates_subset, y_pred_subset, 'r-', alpha=0.7, label='Predito', linewidth=1.5)
        ax.set_xlabel('Data', fontweight='bold')
        ax.set_ylabel(f'{self.variable.capitalize()} ({self._get_unit()})', fontweight='bold')
        ax.set_title('Série Temporal (Últimos 90 dias)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Distribuição dos resíduos
        ax = axes[1, 0]
        residuals = y_true - y_pred
        
        # Histograma com KDE
        n_bins = min(50, len(residuals)//10)
        n, bins, patches = ax.hist(residuals, bins=n_bins, density=True, 
                                  alpha=0.7, color='lightgreen', edgecolor='black')
        
        # Adicionar KDE
        if len(residuals) > 10:
            try:
                kde = gaussian_kde(residuals)
                x_range = np.linspace(residuals.min(), residuals.max(), 200)
                ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
            except:
                pass
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(x=np.mean(residuals), color='orange', linestyle='-', linewidth=2, label='Média')
        
        ax.set_xlabel(f'Resíduos ({self._get_unit()})', fontweight='bold')
        ax.set_ylabel('Densidade', fontweight='bold')
        ax.set_title('Distribuição dos Resíduos', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Estatísticas
        bias = np.mean(residuals)
        std_res = np.std(residuals)
        skew = residuals.skew() if hasattr(residuals, 'skew') else 0
        kurt = residuals.kurtosis() if hasattr(residuals, 'kurtosis') else 0
        
        stats_text = f'Bias = {bias:.3f}\nStd = {std_res:.3f}\nSkew = {skew:.3f}\nKurt = {kurt:.3f}'
        ax.text(0.75, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Q-Q Plot
        ax = axes[1, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot dos Resíduos', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            output_dir = os.path.join('static', 'img')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Nome único para cada modelo
            model_name_clean = model_name.lower().replace(' ', '_')
            filename = os.path.join(output_dir, 
                                  f'resultados_{self.variable}_{model_name_clean}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  ✓ Gráfico salvo: {filename}")
        
        plt.close()  # Fechar figura para economizar memória


def plot_models_comparison(self, save_plot=True):
    """Gera gráfico comparativo de todos os modelos"""
    if not self.results:
        print("Nenhum resultado para plotar")
        return
    
    # Configurar figura
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Comparação de Modelos - {self.variable.capitalize()}', 
                 fontsize=16, fontweight='bold')
    
    # Preparar dados
    models = list(self.results.keys())
    metrics_names = ['RMSE', 'MAE', 'R2', 'Correlation', 'Bias', 'Skill_Score']
    
    # 1. Barras de RMSE e MAE
    ax = axes[0, 0]
    x = np.arange(len(models))
    width = 0.35
    
    rmse_values = [self.results[m]['metrics']['RMSE'] for m in models]
    mae_values = [self.results[m]['metrics']['MAE'] for m in models]
    
    bars1 = ax.bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.8)
    bars2 = ax.bar(x + width/2, mae_values, width, label='MAE', alpha=0.8)
    
    ax.set_ylabel('Erro', fontweight='bold')
    ax.set_title('RMSE e MAE por Modelo', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Adicionar valores
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. R² e Correlação
    ax = axes[0, 1]
    r2_values = [self.results[m]['metrics']['R2'] for m in models]
    corr_values = [self.results[m]['metrics']['Correlation'] for m in models]
    
    bars1 = ax.bar(x - width/2, r2_values, width, label='R²', alpha=0.8)
    bars2 = ax.bar(x + width/2, corr_values, width, label='Correlação', alpha=0.8)
    
    ax.set_ylabel('Valor', fontweight='bold')
    ax.set_title('R² e Correlação por Modelo', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Bias
    ax = axes[0, 2]
    bias_values = [self.results[m]['metrics']['Bias'] for m in models]
    colors = ['green' if abs(b) < 0.5 else 'orange' if abs(b) < 1 else 'red' 
              for b in bias_values]
    
    bars = ax.bar(models, bias_values, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel(f'Bias ({self._get_unit()})', fontweight='bold')
    ax.set_title('Bias por Modelo', fontweight='bold')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 4. Skill Score
    ax = axes[1, 0]
    skill_values = [self.results[m]['metrics']['Skill_Score'] for m in models]
    colors = ['green' if s > 0.5 else 'orange' if s > 0 else 'red' for s in skill_values]
    
    bars = ax.bar(models, skill_values, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Skill Score', fontweight='bold')
    ax.set_title('Skill Score por Modelo', fontweight='bold')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 5. Heatmap de todas as métricas
    ax = axes[1, 1]
    metrics_matrix = []
    for model in models:
        row = []
        for metric in metrics_names:
            value = self.results[model]['metrics'].get(metric, np.nan)
            row.append(value)
        metrics_matrix.append(row)
    
    metrics_matrix = np.array(metrics_matrix)
    
    # Normalizar métricas para visualização
    normalized_matrix = np.zeros_like(metrics_matrix)
    for j, metric in enumerate(metrics_names):
        col = metrics_matrix[:, j]
        if metric in ['RMSE', 'MAE', 'Bias']:  # Métricas onde menor é melhor
            normalized_matrix[:, j] = 1 - (col - col.min()) / (col.max() - col.min() + 1e-10)
        else:  # Métricas onde maior é melhor
            normalized_matrix[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-10)
    
    im = ax.imshow(normalized_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metrics_names)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax.set_yticklabels(models)
    ax.set_title('Heatmap de Métricas (Normalizado)', fontweight='bold')
    
    # Adicionar valores
    for i in range(len(models)):
        for j in range(len(metrics_names)):
            text = ax.text(j, i, f'{metrics_matrix[i, j]:.3f}',
                         ha='center', va='center', color='black', fontsize=8)
    
    # 6. Ranking dos modelos
    ax = axes[1, 2]
    
    # Calcular score composto (quanto menor melhor)
    scores = []
    for model in models:
        m = self.results[model]['metrics']
        # Normalizar e ponderar métricas
        score = (
            m['RMSE'] * 0.3 +  # 30% peso
            m['MAE'] * 0.2 +   # 20% peso
            (1 - m['R2']) * 0.3 +  # 30% peso (invertido)
            abs(m['Bias']) * 0.2   # 20% peso
        )
        scores.append(score)
    
    # Ordenar modelos por score
    model_scores = list(zip(models, scores))
    model_scores.sort(key=lambda x: x[1])
    
    ranked_models = [m[0] for m in model_scores]
    ranked_scores = [m[1] for m in model_scores]
    
    # Plotar ranking
    y_pos = np.arange(len(ranked_models))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(ranked_models)))
    
    bars = ax.barh(y_pos, ranked_scores, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{i+1}. {m}" for i, m in enumerate(ranked_models)])
    ax.set_xlabel('Score Composto (menor é melhor)', fontweight='bold')
    ax.set_title('Ranking de Modelos', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Adicionar valores
    for i, (bar, score) in enumerate(zip(bars, ranked_scores)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{score:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_plot:
        output_dir = os.path.join('static', 'img')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, f'comparacao_modelos_{self.variable}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Gráfico de comparação salvo: {filename}")

        plt.close()


def plot_temporal_analysis_all_models(self, save_plots=True):
    """Análise temporal para todos os modelos"""
    if not self.results:
        print("Nenhum resultado disponível")
        return
    
    # Para cada modelo
    for model_name, model_data in self.results.items():
        print(f"\nGerando análise temporal para {model_name}...")
        
        y_pred = model_data['predictions']
        
        if self.variable == 'precipitation':
            y_true = np.expm1(self.y_test)
        else:
            y_true = self.y_test
        
        # Criar DataFrame para análise
        temporal_df = pd.DataFrame({
            'date': self.test_dates.values,
            'observed': y_true.values if hasattr(y_true, 'values') else y_true,
            'predicted': y_pred,
            'residual': (y_true.values if hasattr(y_true, 'values') else y_true) - y_pred
        })
        
        temporal_df['month'] = pd.to_datetime(temporal_df['date']).dt.month
        temporal_df['year'] = pd.to_datetime(temporal_df['date']).dt.year
        temporal_df['day_of_year'] = pd.to_datetime(temporal_df['date']).dt.dayofyear
        
        # Criar figura
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Análise Temporal - {model_name} - {self.variable.capitalize()}',
                     fontsize=16, fontweight='bold')
        
        # 1. Ciclo anual médio
        ax = axes[0, 0]
        monthly_means = temporal_df.groupby('month').agg({
            'observed': ['mean', 'std'],
            'predicted': ['mean', 'std']
        })
        
        months = range(1, 13)
        month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        
        obs_means = monthly_means['observed']['mean'].values
        pred_means = monthly_means['predicted']['mean'].values
        obs_stds = monthly_means['observed']['std'].values
        pred_stds = monthly_means['predicted']['std'].values
        
        ax.plot(months, obs_means, 'o-', label='Observado', linewidth=2, markersize=8, color='blue')
        ax.plot(months, pred_means, 's-', label='Predito', linewidth=2, markersize=8, color='red')
        
        ax.fill_between(months, obs_means - obs_stds, obs_means + obs_stds,
                       alpha=0.2, color='blue')
        ax.fill_between(months, pred_means - pred_stds, pred_means + pred_stds,
                       alpha=0.2, color='red')
        
        ax.set_xlabel('Mês', fontweight='bold')
        ax.set_ylabel(f'{self.variable.capitalize()} ({self._get_unit()})', fontweight='bold')
        ax.set_title('Ciclo Anual Médio', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(months)
        ax.set_xticklabels(month_names)
        
        # 2. Erro por dia do ano
        ax = axes[0, 1]
        daily_errors = temporal_df.groupby('day_of_year')['residual'].agg(['mean', 'std'])
        
        ax.plot(daily_errors.index, daily_errors['mean'], color='darkred', linewidth=1.5)
        ax.fill_between(daily_errors.index,
                       daily_errors['mean'] - daily_errors['std'],
                       daily_errors['mean'] + daily_errors['std'],
                       alpha=0.3, color='red')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        ax.set_xlabel('Dia do Ano', fontweight='bold')
        ax.set_ylabel(f'Erro Médio ({self._get_unit()})', fontweight='bold')
        ax.set_title('Padrão de Erro Anual', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. Série temporal de erros absolutos
        ax = axes[1, 0]
        temporal_df['abs_error'] = np.abs(temporal_df['residual'])
        
        # Média móvel de 30 dias
        rolling_error = temporal_df.set_index('date')['abs_error'].rolling('30D').mean()
        
        ax.plot(temporal_df['date'], temporal_df['abs_error'], alpha=0.3, color='gray', label='Erro absoluto')
        ax.plot(rolling_error.index, rolling_error.values, color='red', linewidth=2, label='Média móvel 30d')
        
        ax.set_xlabel('Data', fontweight='bold')
        ax.set_ylabel(f'Erro Absoluto ({self._get_unit()})', fontweight='bold')
        ax.set_title('Evolução Temporal do Erro', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Análise de extremos
        ax = axes[1, 1]
        
        # Percentis de observações e predições
        percentiles = [10, 25, 50, 75, 90]
        obs_percentiles = [np.percentile(temporal_df['observed'], p) for p in percentiles]
        pred_percentiles = [np.percentile(temporal_df['predicted'], p) for p in percentiles]
        
        x = np.arange(len(percentiles))
        width = 0.35
        
        ax.bar(x - width/2, obs_percentiles, width, label='Observado', alpha=0.8)
        ax.bar(x + width/2, pred_percentiles, width, label='Predito', alpha=0.8)
        
        ax.set_ylabel(f'{self.variable.capitalize()} ({self._get_unit()})', fontweight='bold')
        ax.set_title('Análise de Percentis', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'P{p}' for p in percentiles])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            output_dir = os.path.join('static', 'img')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            model_name_clean = model_name.lower().replace(' ', '_')
            filename = os.path.join(output_dir, 
                                  f'analise_temporal_{self.variable}_{model_name_clean}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  ✓ Análise temporal salva: {filename}")
        
        plt.close()


# IMPORTANTE: Adicionar estas linhas APÓS a definição da classe WeatherDownscalingModel
# Procure pela linha "class WeatherDownscalingModel:" e role até o final da classe
# Então adicione estas linhas ANTES de "# Flask Web Application"

# Adicionar os novos métodos à classe
WeatherDownscalingModel.plot_all_models_results = plot_all_models_results
WeatherDownscalingModel.plot_models_comparison = plot_models_comparison
WeatherDownscalingModel.plot_temporal_analysis_all_models = plot_temporal_analysis_all_models
# Flask Web Application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = 'uploads'

# Criar pastas necessárias
for folder in ['uploads', 'models', 'results', os.path.join('static', 'img')]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Sessão atual
current_session = {}

@app.route('/')
def index():
    """Página inicial"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Upload de arquivos"""
    try:
        files_info = {}

        # Processar arquivos
        for file_type in ['era5', 'station', 'dem']:
            if file_type in request.files:
                file = request.files[file_type]
                if file.filename != '':
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    files_info[file_type] = filepath

        # Parâmetros
        latitude = float(request.form.get('latitude', -9.41))
        longitude = float(request.form.get('longitude', -77.35))
        split_date = request.form.get('split_date', '2018-01-01')

        # Validar arquivos necessários
        required_files = ['era5', 'station', 'dem']
        missing_files = [f for f in required_files if f not in files_info]
        if missing_files:
            return jsonify({
                'status': 'error', 
                'message': f'Arquivos faltantes: {missing_files}'
            }), 400

        # Informações dos dados
        data_info = {
            'status': 'uploaded',
            'files': list(files_info.keys()),
            'latitude': latitude,
            'longitude': longitude,
            'split_date': split_date
        }

        # Armazenar na sessão
        current_session['files'] = files_info
        current_session['params'] = {
            'latitude': latitude,
            'longitude': longitude,
            'split_date': split_date
        }
        current_session['data_info'] = data_info

        return jsonify({'status': 'success', 'data_info': data_info})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/process', methods=['POST'])
def process_models():
    """Processar modelos com configurações avançadas"""
    try:
        if 'files' not in current_session:
            return jsonify({'status': 'error', 'message': 'Nenhum arquivo foi carregado'}), 400

        # Obter configurações do request
        data = request.json or {}
        selected_models = data.get('models', ['RandomForest', 'XGBoost', 'MLP'])
        variables = data.get('variables', ['temperature', 'precipitation'])
        use_ensemble = data.get('use_ensemble', True)
        optimize_params = data.get('optimize_params', True)
        
        # Configurações de otimização
        opt_config = data.get('optimization_config', {})
        
        # Executar processamento com as novas configurações
        results = run_downscaling_pipeline_advanced(
            current_session['files'],
            current_session['params'],
            selected_models,
            variables,
            use_ensemble,
            optimize_params,
            opt_config
        )

        # Contar gráficos gerados
        total_plots = 0
        for var in results:
            if isinstance(results[var], dict) and 'generated_plots' in results[var]:
                plots = results[var]['generated_plots']
                if 'individual_models' in plots:
                    total_plots += len(plots['individual_models'])
                if 'temporal_individual' in plots:
                    total_plots += len(plots['temporal_individual'])
                total_plots += 3  # comparison + summary + feature_importance

        current_session['results'] = results

        return jsonify({
            'status': 'success', 
            'results': results,
            'total_plots_generated': total_plots
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/download_results')
def download_results():
    """Download dos resultados"""
    try:
        if 'results' not in current_session:
            return jsonify({'status': 'error', 'message': 'Nenhum resultado disponível'}), 400

        # Criar arquivo ZIP com resultados
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Adicionar relatórios da pasta 'results'
            results_dir = 'results'
            for variable in current_session['results']:
                report_name = f'relatorio_downscaling_{variable}.txt'
                report_path = os.path.join(results_dir, report_name)
                if os.path.exists(report_path):
                    zip_file.write(report_path, report_name)
            
            # Adicionar gráficos da pasta 'static/img'
            img_dir = os.path.join('static', 'img')
            for variable in current_session['results']:
                for graph_type in ['resultados', 'feature_importance', 'analise_temporal']:
                    graph_name = f'{graph_type}_{variable}.png'
                    graph_path = os.path.join(img_dir, graph_name)
                    if os.path.exists(graph_path):
                        zip_file.write(graph_path, os.path.join('img', graph_name))
        
        zip_buffer.seek(0)
        
        return send_file(
            io.BytesIO(zip_buffer.read()),
            as_attachment=True,
            download_name='resultados_downscaling.zip',
            mimetype='application/zip'
        )

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

def run_downscaling_pipeline_advanced(files, params, selected_models, variables, 
                                    use_ensemble, optimize_params, opt_config):
    """Pipeline principal com configurações avançadas de otimização"""
    lat = params['latitude']
    lon = params['longitude']
    split_date = params['split_date']

    era5_path = files['era5']
    station_path = files['station']
    dem_path = files['dem']

    results_all = {}

    for variable in variables:
        print(f"\n{'='*60}")
        print(f"PROCESSANDO {variable.upper()}")
        print(f"{'='*60}")

        try:
            # Configuração do modelo com parâmetros avançados
            config = {
                'optimize_hyperparams': optimize_params,
                'feature_selection': opt_config.get('feature_selection', True),
                'max_features': 80,
                'ensemble_size': 3,
                # Adicionar configurações de otimização
                'optimize_all_models': opt_config.get('optimize_all_models', True),
                'fast_optimization': opt_config.get('fast_optimization', True),
                'use_random_search': opt_config.get('use_random_search', False),
                'random_search_iter': opt_config.get('random_search_iter', 30),
                'cv_folds': opt_config.get('cv_folds', 3),
                'show_top_params': False,  # Para economizar output no web
                'verbose': False
            }

            # Criar modelo
            model = WeatherDownscalingModel(variable=variable, config=config)

            # Pipeline completo
            model.load_and_merge_data(era5_path, station_path, dem_path, lat, lon)
            model.create_features()
            model.prepare_data(split_date)
            model.train_models(selected_models=selected_models, optimize=optimize_params)

            # Criar ensemble se solicitado
            if use_ensemble and len(model.results) >= 2:
                model.create_ensemble()

            # Gerar visualizações
            generate_individual = opt_config.get('generate_individual_plots', True)
            
            if generate_individual:
                model.plot_all_models_results(save_plots=True)
                model.plot_temporal_analysis_all_models(save_plots=True)
            
            model.plot_models_comparison(save_plot=True)
            model.plot_results(save_plots=True)
            model.plot_feature_importance(save_plot=True)
            model.plot_temporal_analysis(save_plot=True)

            # Gerar relatório
            model.generate_report()

            # Salvar modelos
            model.save_models()

            # Coletar resultados
            plots_info = {
                'comparison': f'comparacao_modelos_{variable}.png',
                'summary': f'resultados_downscaling_{variable}.png'
            }
            
            if generate_individual:
                plots_info.update({
                    'individual_models': [f'resultados_{variable}_{m.lower().replace(" ", "_")}.png' 
                                        for m in model.results.keys()],
                    'temporal_individual': [f'analise_temporal_{variable}_{m.lower().replace(" ", "_")}.png' 
                                          for m in model.results.keys()]
                })

            results_all[variable] = {
                'models': {
                    model_name: {
                        'metrics': data['metrics'],
                        'model_type': type(data['model']).__name__
                    }
                    for model_name, data in model.results.items()
                },
                'best_model': min(model.results.items(), 
                                key=lambda x: x[1]['metrics']['RMSE'])[0],
                'data_info': {
                    'total_records': len(model.data),
                    'train_records': len(model.X_train),
                    'test_records': len(model.X_test),
                    'features_count': len(model.features)
                },
                'generated_plots': plots_info,
                'optimization_config': config
            }

            print(f"\n✅ {variable.upper()} processado com sucesso!")

        except Exception as e:
            print(f"\n❌ Erro ao processar {variable}: {str(e)}")
            results_all[variable] = {'error': str(e)}

    return results_all



# Função principal para uso standalone
def run_weather_downscaling(era5_path, station_path, dem_path, 
                           variables=['temperature', 'precipitation'],
                           lat=-9.41, lon=-77.35, 
                           split_date='2018-01-01',
                           selected_models=None,
                           optimize_models=True,
                           use_ensemble=True,
                           save_results=True,
                           generate_plots=True,
                           generate_individual_plots=True):  # NOVO PARÂMETRO!
    # ... código da função updated_run_weather_downscaling ...
    """
    Executa o pipeline completo de downscaling climático
    
    Parameters:
    -----------
    era5_path : str
        Caminho para dados ERA5/ERA5-Land
    station_path : str
        Caminho para dados da estação
    dem_path : str
        Caminho para modelo de elevação digital
    variables : list
        Lista de variáveis ['temperature', 'precipitation']
    lat, lon : float
        Coordenadas da estação
    split_date : str
        Data para divisão treino/teste
    selected_models : list
        Modelos a treinar (None = todos)
    optimize_models : bool
        Otimizar hiperparâmetros
    use_ensemble : bool
        Criar modelo ensemble
    save_results : bool
        Salvar modelos e resultados
    generate_plots : bool
        Gerar gráficos
    
    Returns:
    --------
    dict : Resultados para cada variável
    """
    
    if selected_models is None:
        selected_models = ['LinearRegression', 'Ridge', 'RandomForest', 
                          'ExtraTrees', 'GradientBoosting', 'MLP']
        if XGBOOST_AVAILABLE:
            selected_models.append('XGBoost')
    
    results = {}
    
    for variable in variables:
        print(f"\n{'='*70}")
        print(f"🌦️  DOWNSCALING CLIMÁTICO - {variable.upper()}")
        print(f"{'='*70}")
        
        try:
            # Configuração
            config = {
                'optimize_hyperparams': optimize_models,
                'feature_selection': True,
                'max_features': 100,
                'ensemble_size': min(3, len(selected_models))
            }
            
            # Criar modelo
            model = WeatherDownscalingModel(variable=variable, config=config)
            
            # Pipeline completo
            print("📊 Carregando dados...")
            model.load_and_merge_data(era5_path, station_path, dem_path, lat, lon)
            
            print("🔧 Criando features...")
            model.create_features()
            
            print("📈 Preparando dados...")
            model.prepare_data(split_date)
            
            print("🤖 Treinando modelos...")
            model.train_models(selected_models=selected_models, optimize=optimize_models)
            
            if use_ensemble and len(model.results) >= 2:
                print("🔗 Criando ensemble...")
                model.create_ensemble()
            
            if generate_plots:
                print("📊 Gerando visualizações...")
                model.plot_results(save_plots=save_results)
                model.plot_feature_importance(save_plot=save_results)
                model.plot_temporal_analysis(save_plot=save_results)
            
            print("📋 Gerando relatório...")
            model.generate_report()
            
            if save_results:
                print("💾 Salvando modelos...")
                model.save_models()
            
            # Coletar resultados
            results[variable] = {
                'model': model,
                'best_model_name': min(model.results.items(), 
                                     key=lambda x: x[1]['metrics']['RMSE'])[0],
                'metrics': {name: data['metrics'] for name, data in model.results.items()},
                'success': True
            }
            
            print(f"✅ {variable.upper()} concluído com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro em {variable}: {str(e)}")
            results[variable] = {'error': str(e), 'success': False}
    
    # Resumo final
    print(f"\n{'='*70}")
    print("📊 RESUMO FINAL")
    print(f"{'='*70}")
    
    for variable, result in results.items():
        if result.get('success', False):
            best_model = result['best_model_name']
            best_metrics = result['metrics'][best_model]
            print(f"\n🎯 {variable.upper()}:")
            print(f"   • Melhor modelo: {best_model}")
            print(f"   • RMSE: {best_metrics['RMSE']:.4f}")
            print(f"   • R²: {best_metrics['R2']:.4f}")
        else:
            print(f"\n❌ {variable.upper()}: FALHOU")
    
    return results


# Template HTML básico para a interface web
HTML_TEMPLATE_ADVANCED = '''
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistema de Downscaling Climático</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5; 
        }
        .container { 
            max-width: 900px; 
            margin: 0 auto; 
            background: white; 
            padding: 20px; 
            border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { 
            color: #2c3e50; 
            text-align: center; 
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .form-group { 
            margin-bottom: 15px; 
        }
        label { 
            display: block; 
            margin-bottom: 5px; 
            font-weight: bold; 
            color: #555;
        }
        input, select { 
            width: 100%; 
            padding: 8px; 
            border: 1px solid #ddd; 
            border-radius: 4px; 
            box-sizing: border-box;
        }
        input[type="checkbox"] {
            width: auto;
            margin-right: 10px;
        }
        .checkbox-group {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin: 10px 0;
        }
        .checkbox-item {
            flex: 0 0 calc(33.333% - 20px);
            display: flex;
            align-items: center;
        }
        button { 
            background: #3498db; 
            color: white; 
            padding: 12px 30px; 
            border: none; 
            border-radius: 4px; 
            cursor: pointer; 
            font-size: 16px;
            font-weight: bold;
            margin-top: 20px;
            width: 100%;
        }
        button:hover { 
            background: #2980b9; 
        }
        button:disabled {
            background: #95a5a6;
            cursor: not-allowed;
        }
        .results { 
            margin-top: 20px; 
            padding: 15px; 
            background: #ecf0f1; 
            border-radius: 5px; 
        }
        .error { 
            color: #e74c3c; 
            font-weight: bold;
        }
        .success { 
            color: #27ae60; 
            font-weight: bold;
        }
        .warning {
            color: #f39c12;
            font-weight: bold;
        }
        .config-section {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }
        .config-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 10px;
        }
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
            color: #3498db;
            margin-left: 5px;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 250px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -125px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 12px;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        .progress {
            display: none;
            margin-top: 20px;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #ddd;
            border-radius: 10px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background-color: #3498db;
            width: 0%;
            transition: width 0.3s;
            text-align: center;
            color: white;
            line-height: 20px;
        }
        .time-estimate {
            margin-top: 10px;
            font-style: italic;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌦️ Sistema de Downscaling Climático com IA</h1>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <h2>📁 Arquivos de Entrada</h2>
            
            <div class="form-group">
                <label>Dados ERA5/ERA5-Land (CSV):</label>
                <input type="file" name="era5" accept=".csv" required>
            </div>
            
            <div class="form-group">
                <label>Dados da Estação (CSV):</label>
                <input type="file" name="station" accept=".csv" required>
            </div>
            
            <div class="form-group">
                <label>Modelo de Elevação (NetCDF):</label>
                <input type="file" name="dem" accept=".nc" required>
            </div>
            
            <h2>📍 Localização</h2>
            
            <div class="config-grid">
                <div class="form-group">
                    <label>Latitude:</label>
                    <input type="number" name="latitude" step="0.001" value="-9.41" required>
                </div>
                
                <div class="form-group">
                    <label>Longitude:</label>
                    <input type="number" name="longitude" step="0.001" value="-77.35" required>
                </div>
            </div>
            
            <div class="form-group">
                <label>Data de Divisão (Treino/Teste):</label>
                <input type="date" name="split_date" value="2018-01-01" required>
            </div>
            
            <h2>🌡️ Variáveis Climáticas</h2>
            
            <div class="checkbox-group">
                <div class="checkbox-item">
                    <input type="checkbox" id="var_temp" name="variables" value="temperature" checked>
                    <label for="var_temp">Temperatura</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="var_prec" name="variables" value="precipitation" checked>
                    <label for="var_prec">Precipitação</label>
                </div>
            </div>
            
            <h2>🤖 Modelos de Machine Learning</h2>
            
            <div class="checkbox-group">
                <div class="checkbox-item">
                    <input type="checkbox" id="model_lr" name="models" value="LinearRegression" checked>
                    <label for="model_lr">Linear Regression</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="model_ridge" name="models" value="Ridge" checked>
                    <label for="model_ridge">Ridge</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="model_rf" name="models" value="RandomForest" checked>
                    <label for="model_rf">Random Forest</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="model_et" name="models" value="ExtraTrees" checked>
                    <label for="model_et">Extra Trees</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="model_gb" name="models" value="GradientBoosting" checked>
                    <label for="model_gb">Gradient Boosting</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="model_xgb" name="models" value="XGBoost" checked>
                    <label for="model_xgb">XGBoost</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="model_svr" name="models" value="SVR">
                    <label for="model_svr">SVR</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="model_mlp" name="models" value="MLP" checked>
                    <label for="model_mlp">MLP Neural Network</label>
                </div>
            </div>
            
            <h2>⚙️ Configurações de Otimização</h2>
            
            <div class="config-section">
                <div class="form-group">
                    <label>
                        Modo de Otimização:
                        <span class="tooltip">ℹ️
                            <span class="tooltiptext">
                                Sem otimização: Usa parâmetros padrão (rápido)<br>
                                Otimização rápida: Testa poucos parâmetros<br>
                                Otimização completa: Testa muitos parâmetros<br>
                                Random Search: Amostragem aleatória eficiente
                            </span>
                        </span>
                    </label>
                    <select id="optimization_mode" name="optimization_mode">
                        <option value="none">Sem otimização (mais rápido)</option>
                        <option value="fast" selected>Otimização rápida</option>
                        <option value="full">Otimização completa</option>
                        <option value="random">Random Search</option>
                    </select>
                </div>
                
                <div class="config-grid">
                    <div class="form-group">
                        <input type="checkbox" id="optimize_all" name="optimize_all_models" checked>
                        <label for="optimize_all" style="display: inline;">
                            Otimizar todos os modelos
                            <span class="tooltip">ℹ️
                                <span class="tooltiptext">
                                    Se desativado, otimiza apenas RandomForest, XGBoost e Ridge
                                </span>
                            </span>
                        </label>
                    </div>
                    
                    <div class="form-group">
                        <input type="checkbox" id="use_ensemble" name="use_ensemble" checked>
                        <label for="use_ensemble" style="display: inline;">
                            Criar modelo ensemble
                            <span class="tooltip">ℹ️
                                <span class="tooltiptext">
                                    Combina os melhores modelos para melhor performance
                                </span>
                            </span>
                        </label>
                    </div>
                    
                    <div class="form-group">
                        <input type="checkbox" id="feature_selection" name="feature_selection" checked>
                        <label for="feature_selection" style="display: inline;">
                            Seleção automática de features
                        </label>
                    </div>
                    
                    <div class="form-group">
                        <input type="checkbox" id="generate_plots" name="generate_individual_plots" checked>
                        <label for="generate_plots" style="display: inline;">
                            Gerar gráficos individuais
                            <span class="tooltip">ℹ️
                                <span class="tooltiptext">
                                    Gera gráficos detalhados para cada modelo (mais tempo)
                                </span>
                            </span>
                        </label>
                    </div>
                </div>
                
                <div class="form-group" id="cv_folds_group">
                    <label>Cross-validation folds:</label>
                    <input type="number" id="cv_folds" name="cv_folds" min="2" max="10" value="3">
                </div>
                
                <div class="form-group" id="random_iter_group" style="display: none;">
                    <label>Iterações do Random Search:</label>
                    <input type="number" id="random_iter" name="random_search_iter" min="10" max="100" value="30">
                </div>
            </div>
            
            <button type="submit" id="submitBtn">🚀 Processar Dados</button>
        </form>
        
        <div class="progress" id="progressDiv">
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill">0%</div>
            </div>
            <div class="time-estimate" id="timeEstimate"></div>
        </div>
        
        <div id="results" class="results" style="display: none;"></div>
    </div>

    <script>
        // Mostrar/ocultar opções baseado no modo de otimização
        document.getElementById('optimization_mode').addEventListener('change', function(e) {
            const mode = e.target.value;
            const randomIterGroup = document.getElementById('random_iter_group');
            const optimizeAllCheckbox = document.getElementById('optimize_all');
            
            if (mode === 'random') {
                randomIterGroup.style.display = 'block';
            } else {
                randomIterGroup.style.display = 'none';
            }
            
            if (mode === 'none') {
                optimizeAllCheckbox.checked = false;
                optimizeAllCheckbox.disabled = true;
            } else {
                optimizeAllCheckbox.disabled = false;
            }
            
            updateTimeEstimate();
        });
        
        // Atualizar estimativa de tempo
        function updateTimeEstimate() {
            const mode = document.getElementById('optimization_mode').value;
         const selectedModels = Array.from(document.querySelectorAll('input[name="models"]:checked'))
                               .map(cb => cb.value);
            const nVars = document.querySelectorAll('input[name="variables"]:checked').length;
            const optimizeAll = document.getElementById('optimize_all').checked;
            const individualPlots = document.getElementById('generate_plots').checked;
            
            // Tempo REAL por modelo em minutos
            const modelTimes = {
                'none': {
                    'LinearRegression': 0.5,
                    'Ridge': 0.5,
                    'RandomForest': 2,
                    'ExtraTrees': 2,
                    'GradientBoosting': 3,
                    'XGBoost': 2,
                    'SVR': 1,
                    'MLP': 1
                },
                'fast': {
                    'LinearRegression': 1,
                    'Ridge': 2,
                    'RandomForest': 15,
                    'ExtraTrees': 15,
                    'GradientBoosting': 10,
                    'XGBoost': 8,
                    'SVR': 5,
                    'MLP': 8
                },
                'full': {
                    'LinearRegression': 1,
                    'Ridge': 3,
                    'RandomForest': 180,  // 3 horas!
                    'ExtraTrees': 180,    // 3 horas!
                    'GradientBoosting': 120, // 2 horas!
                    'XGBoost': 90,        // 1.5 horas!
                    'SVR': 30,
                    'MLP': 45
                },
                'random': {
                    'LinearRegression': 1,
                    'Ridge': 2,
                    'RandomForest': 30,
                    'ExtraTrees': 30,
                    'GradientBoosting': 20,
                    'XGBoost': 15,
                    'SVR': 10,
                    'MLP': 15
                }
            };
            
            // Calcular tempo total
            let totalMinutes = 0;
            
            selectedModels.forEach(model => {
                let timePerModel = modelTimes[mode][model] || 5;
                
                // Se não otimizar todos, alguns modelos são mais rápidos
                if (!optimizeAll && mode !== 'none') {
                    const quickModels = ['LinearRegression', 'Lasso', 'DecisionTree'];
                    if (quickModels.includes(model)) {
                        timePerModel = modelTimes['none'][model] || 0.5;
                    }
                }
                
                // Adicionar tempo para gráficos individuais
                if (individualPlots) {
                    timePerModel += 2; // 2 minutos extras por modelo para gráficos
                }
                
                // Multiplicar pelo número de variáveis
                totalMinutes += timePerModel * nVars;
            });
            
            // Adicionar tempo base para processamento de dados
            totalMinutes += 5 * nVars; // 5 minutos por variável para features
            
            // Converter para formato legível
            let timeString = '';
            if (totalMinutes < 60) {
                timeString = `${Math.round(totalMinutes)} minutos`;
            } else {
                const hours = Math.floor(totalMinutes / 60);
                const minutes = Math.round(totalMinutes % 60);
                timeString = `${hours}h ${minutes}min`;
            }
            
            // Adicionar margem de erro de 20%
            const minTime = Math.round(totalMinutes * 0.8);
            const maxTime = Math.round(totalMinutes * 1.2);
            
            let finalEstimate = '';
            if (minTime < 60 && maxTime < 60) {
                finalEstimate = `${minTime}-${maxTime} minutos`;
            } else if (minTime >= 60 && maxTime >= 60) {
                const minHours = Math.floor(minTime / 60);
                const minMinutes = minTime % 60;
                const maxHours = Math.floor(maxTime / 60);
                const maxMinutes = maxTime % 60;
                finalEstimate = `${minHours}h${minMinutes}min - ${maxHours}h${maxMinutes}min`;
            } else {
                finalEstimate = `${minTime} minutos - ${Math.floor(maxTime/60)}h${maxTime%60}min`;
            }
            
            // Adicionar aviso para otimização completa
            if (mode === 'full' && selectedModels.includes('RandomForest')) {
                finalEstimate += ' ⚠️ (RandomForest pode demorar muito!)';
            }
            
            const estimate = document.getElementById('timeEstimate');
            if (estimate && selectedModels.length > 0 && nVars > 0) {
                estimate.innerHTML = `<strong>Tempo estimado:</strong> ${finalEstimate}`;
                
                // Adicionar classe de aviso se for muito demorado
                if (totalMinutes > 180) {
                    estimate.className = 'time-estimate warning';
                    estimate.innerHTML += '<br><small>💡 Considere usar "Otimização rápida" ou "Random Search" para economizar tempo.</small>';
                } else {
                    estimate.className = 'time-estimate';
                }
            }
        }
        
        // Atualizar estimativa quando mudar seleções
        document.querySelectorAll('input[type="checkbox"]').forEach(cb => {
            cb.addEventListener('change', updateTimeEstimate);
        });
        
        updateTimeEstimate();
        
        // Processar formulário
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const submitBtn = document.getElementById('submitBtn');
            const progressDiv = document.getElementById('progressDiv');
            const progressFill = document.getElementById('progressFill');
            const resultsDiv = document.getElementById('results');
            
            // Validar seleções
            const selectedModels = Array.from(document.querySelectorAll('input[name="models"]:checked'))
                                      .map(cb => cb.value);
            const selectedVars = Array.from(document.querySelectorAll('input[name="variables"]:checked'))
                                     .map(cb => cb.value);
            
            if (selectedModels.length === 0) {
                alert('Selecione pelo menos um modelo!');
                return;
            }
            
            if (selectedVars.length === 0) {
                alert('Selecione pelo menos uma variável!');
                return;
            }
            
            // Preparar dados do formulário
            const formData = new FormData(this);
            
            // Desabilitar botão e mostrar progresso
            submitBtn.disabled = true;
            progressDiv.style.display = 'block';
            resultsDiv.style.display = 'none';
            
            try {
                // 1. Upload dos arquivos
                updateProgress(10, 'Fazendo upload dos arquivos...');
                
                const uploadResponse = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const uploadResult = await uploadResponse.json();
                
                if (uploadResult.status !== 'success') {
                    throw new Error(uploadResult.message || 'Erro no upload');
                }
                
                // 2. Preparar configuração
                const mode = document.getElementById('optimization_mode').value;
                const config = {
                    models: selectedModels,
                    variables: selectedVars,
                    use_ensemble: document.getElementById('use_ensemble').checked,
                    optimize_params: mode !== 'none',
                    optimization_config: {
                        optimize_all_models: document.getElementById('optimize_all').checked,
                        fast_optimization: mode === 'fast',
                        use_random_search: mode === 'random',
                        random_search_iter: parseInt(document.getElementById('random_iter').value),
                        cv_folds: parseInt(document.getElementById('cv_folds').value),
                        feature_selection: document.getElementById('feature_selection').checked,
                        generate_individual_plots: document.getElementById('generate_plots').checked
                    }
                };
                
                // 3. Processar modelos
                updateProgress(20, 'Processando dados e criando features...');
                
                const processResponse = await fetch('/process', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                
                const processResult = await processResponse.json();
                
                if (processResult.status === 'success') {
                    updateProgress(100, 'Processamento concluído!');
                    showResults(processResult);
                } else {
                    throw new Error(processResult.message || 'Erro no processamento');
                }
                
            } catch (error) {
                resultsDiv.innerHTML = `<p class="error">❌ Erro: ${error.message}</p>`;
                resultsDiv.style.display = 'block';
            } finally {
                submitBtn.disabled = false;
                setTimeout(() => {
                    progressDiv.style.display = 'none';
                }, 2000);
            }
        });
        
        function updateProgress(percent, message) {
            const progressFill = document.getElementById('progressFill');
            progressFill.style.width = percent + '%';
            progressFill.textContent = percent + '%';
            
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `<p>${message}</p>`;
            resultsDiv.style.display = 'block';
        }
        
        function showResults(result) {
            const resultsDiv = document.getElementById('results');
            let html = '<h2>✅ Processamento Concluído!</h2>';
            
            if (result.total_plots_generated) {
                html += `<p>📊 ${result.total_plots_generated} gráficos gerados</p>`;
            }
            
            // Mostrar resultados por variável
            for (const [variable, data] of Object.entries(result.results)) {
                if (data.error) {
                    html += `<p class="error">${variable}: ${data.error}</p>`;
                } else {
                    html += `<h3>${variable.charAt(0).toUpperCase() + variable.slice(1)}</h3>`;
                    html += '<ul>';
                    
                    if (data.best_model) {
                        const metrics = data.models[data.best_model].metrics;
                        html += `<li><strong>Melhor modelo:</strong> ${data.best_model}</li>`;
                        html += `<li>RMSE: ${metrics.RMSE.toFixed(4)}</li>`;
                        html += `<li>R²: ${metrics.R2.toFixed(4)}</li>`;
                    }
                    
                    html += `<li>Dados: ${data.data_info.total_records} registros</li>`;
                    html += `<li>Features: ${data.data_info.features_count}</li>`;
                    html += '</ul>';
                }
            }
            
            html += '<p class="success"><a href="/download_results" class="button">📥 Download dos Resultados</a></p>';
            
            resultsDiv.innerHTML = html;
        }
    </script>
</body>
</html>
'''

# Salvar template HTML
def create_advanced_html_template():
    """Cria o template HTML avançado"""
    templates_dir = 'templates'
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE_ADVANCED)
    
    print("✅ Template HTML avançado criado em templates/index.html")

# Exemplo de uso
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'web':
        # Modo web
        create_advanced_html_template()
        print("🌍 Iniciando servidor web...")
        print("Acesse: http://localhost:5000")
        app.run(debug=False, port=5000)
        
    else:
        # Modo standalone - exemplo
        print("🚀 Executando exemplo de downscaling climático...")
        
        # Caminhos dos arquivos (ajustar conforme necessário)
        era5_path = "dados_era5_era5land.csv"
        station_path = "dados_estacao_cuchillacocha.csv"
        dem_path = "Quilcayhuanca_static.nc"
        
        # Verificar se arquivos existem
        files_exist = all(os.path.exists(f) for f in [era5_path, station_path, dem_path])
        
        if not files_exist:
            print("❌ Arquivos de exemplo não encontrados.")
            print("💡 Para usar o modo web: python script.py web")
            print("💡 Para standalone: ajuste os caminhos dos arquivos no código")
            
        else:
            # Executar downscaling
            results = run_weather_downscaling(
                era5_path=era5_path,
                station_path=station_path,
                dem_path=dem_path,
                variables=['temperature', 'precipitation'],
                lat=-9.41, lon=-77.35,
                split_date='2018-01-01',
                optimize_models=True,
                use_ensemble=True,
                save_results=True,
                generate_plots=True
            )
            
            print("\n🎉 Downscaling concluído!")
            print("📁 Verifique os arquivos gerados na pasta atual")
            