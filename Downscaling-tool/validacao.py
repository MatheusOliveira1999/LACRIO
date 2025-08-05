"""
Sistema de Validação de Modelos Climáticos em Outras Montanhas - CORRIGIDO
Permite testar modelos treinados em novas localidades montanhosas
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Métricas
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads_validation'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.secret_key = 'your-secret-key-here'

# Criar diretórios necessários
for folder in ['uploads_validation', 'static', 'templates', 'results_validation', 'models']:
    os.makedirs(folder, exist_ok=True)

class ModelValidator:
    """
    Classe para validar modelos de downscaling em novas localizações
    """
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.model_info = {}
        self.generic_model_info = {}
        self.validation_results = {}
        
    def load_models(self):
        """Carrega todos os modelos disponíveis e informações genéricas de features."""
        print("Carregando modelos e informações de features...")
        
        # Tentar carregar arquivos de informação genéricos primeiro
        generic_info_files = {
            'temperature': os.path.join(self.models_dir, 'model_info_temperature.json'),
            'precipitation': os.path.join(self.models_dir, 'model_info_precipitation.json')
        }

        for var_type, path in generic_info_files.items():
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        self.generic_model_info[var_type] = json.load(f).get('features', [])
                    print(f"✓ Informações genéricas de features para {var_type} carregadas de {os.path.basename(path)}")
                except Exception as e:
                    print(f"✗ Erro ao carregar informações genéricas de {os.path.basename(path)}: {str(e)}")
            else:
                print(f"Aviso: Arquivo de informações genéricas para {var_type} não encontrado: {os.path.basename(path)}")

        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        
        for model_file in model_files:
            try:
                model_path = os.path.join(self.models_dir, model_file)
                model_name = model_file.replace('.pkl', '')
                
                # Carregar modelo
                self.loaded_models[model_name] = joblib.load(model_path)
                print(f"✓ Modelo carregado: {model_name}")
                
                # Carregar informações do modelo se disponível
                info_file = model_file.replace('.pkl', '.json')
                info_path = os.path.join(self.models_dir, info_file)
                
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        self.model_info[model_name] = json.load(f)
                        print(f"  (Info específica para {model_name} carregada)")
                
            except Exception as e:
                print(f"✗ Erro ao carregar {model_file}: {str(e)}")
        
        return self.loaded_models
    
    def prepare_validation_data(self, era5_path, station_path, dem_path, lat, lon):
        """Prepara dados para validação."""
        try:
            # Carregar dados ERA5
            era5_df = pd.read_csv(era5_path)
            era5_df['date'] = pd.to_datetime(era5_df['date'], errors='coerce')
            era5_df = era5_df.dropna(subset=['date'])
            if era5_df.empty:
                raise ValueError("Nenhuma data válida encontrada no arquivo ERA5 após conversão.")
            print(f"ERA5 carregado: {era5_df.shape}")
            
            # Carregar dados da estação
            station_df = pd.read_csv(station_path)
            station_df = self._standardize_station_columns(station_df)
            print(f"Estação carregada: {station_df.shape}")
            
            # Mesclar dados por data
            merged_data = pd.merge(era5_df, station_df[['date', 'temp_obs', 'prec_obs']], on='date', how='inner')
            print(f"Dados mesclados: {merged_data.shape}")
            
            if merged_data.empty:
                raise ValueError("Nenhum dado após mesclagem. Verifique as datas e o formato dos arquivos.")
            
            # Processar elevação
            merged_data = self._process_elevation(merged_data, dem_path, lat, lon)
            
            # Criar features
            merged_data = self._create_features(merged_data)
            
            return merged_data
            
        except Exception as e:
            print(f"Erro ao preparar dados: {str(e)}")
            raise
    
    def _standardize_station_columns(self, df):
        """Padroniza nomes das colunas e garante o tipo datetime para a coluna 'date'."""
        # Renomear coluna de data se necessário
        date_columns = ['data', 'Data', 'DATE', 'datetime', 'Datetime']
        found_date_col = None
        for col in date_columns:
            if col in df.columns:
                df = df.rename(columns={col: 'date'})
                found_date_col = 'date'
                break
        
        if found_date_col:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            if df.empty:
                raise ValueError("Nenhuma data válida encontrada após padronização e conversão.")

        # Mapear colunas de temperatura
        temp_columns = {
            'Temperatura': 'temp_obs',
            'temperatura': 'temp_obs',
            'Temperature': 'temp_obs',
            'TEMP': 'temp_obs',
            'temp': 'temp_obs',
            'T2M': 'temp_obs'
        }
        
        # Mapear colunas de precipitação
        prec_columns = {
            'Precipitação': 'prec_obs',
            'precipitacao': 'prec_obs',
            'Precipitation': 'prec_obs',
            'PREC': 'prec_obs',
            'prec': 'prec_obs',
            'Rain': 'prec_obs',
            'TP': 'prec_obs'
        }
        
        # Aplicar mapeamentos
        for old_name, new_name in {**temp_columns, **prec_columns}.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Garantir que temos as colunas necessárias
        if 'temp_obs' not in df.columns:
            print("Aviso: Coluna de temperatura observada ('temp_obs') não encontrada. Será preenchida com NaN.")
            df['temp_obs'] = np.nan
        
        if 'prec_obs' not in df.columns:
            print("Aviso: Coluna de precipitação observada ('prec_obs') não encontrada. Será preenchida com 0.0.")
            df['prec_obs'] = 0.0
        
        return df
    
    def _process_elevation(self, df, dem_path, lat, lon):
        """Processa dados de elevação"""
        elev_local = 1500  # Valor padrão inicial

        try:
            if dem_path and os.path.exists(dem_path):
                dem = xr.open_dataset(dem_path)
                
                # Buscar variável de elevação
                elevation_vars = ['HGT', 'elevation', 'dem', 'z', 'height', 'altitude']
                elev_var = None
                
                for var in elevation_vars:
                    if var in dem.variables:
                        elev_var = var
                        break
                
                if elev_var:
                    elev_local = dem[elev_var].sel(lat=lat, lon=lon, method='nearest').item()
                    if np.isnan(elev_local) or np.isinf(elev_local):
                        print(f"Aviso: Elevação obtida do DEM é inválida ({elev_local}). Usando valor padrão de 1500m.")
                        elev_local = 1500
                else:
                    print("Aviso: Variável de elevação não encontrada no DEM. Usando valor padrão de 1500m.")
            
            df['elevation'] = elev_local
            
            # Calcular diferença de altitude se possível
            if 'z' in df.columns:
                df['z'] = pd.to_numeric(df['z'], errors='coerce')
                df['alt_ERA5'] = df['z'] / 9.80665
                df['alt_diff'] = elev_local - df['alt_ERA5']
            else:
                df['alt_diff'] = 0
            
            print(f"Elevação da estação processada: {elev_local:.1f}m")
            
        except Exception as e:
            print(f"Erro ao processar elevação: {str(e)}")
            df['elevation'] = elev_local
            df['alt_diff'] = 0
        
        return df
    
    def _create_features(self, df):
        """Cria features necessárias para os modelos"""
        # Features temporais básicas
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['season'] = df['month'].apply(lambda x: (x % 12) // 3 + 1)
        
        # Features cíclicas
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Tendência temporal
        start_date = df['date'].min()
        df['days_since_start'] = (df['date'] - start_date).dt.days
        
        # Features meteorológicas básicas
        if 'u10' in df.columns and 'v10' in df.columns:
            df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
            df['wind_direction'] = np.arctan2(df['v10'], df['u10'])
        else:
            df['wind_speed'] = 0
            df['wind_direction'] = 0
        
        # Criar lags básicos
        meteo_cols = ['t2m', 'tp', 'sp', 'u10', 'v10', 'd2m', 'ssr', 'str', 'e', 'hcc', 'lcc', 'mcc', 'tcc']
        meteo_cols = [col for col in meteo_cols if col in df.columns]
        
        for col in meteo_cols:
            for lag in [1, 3, 7]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Médias móveis
        for col in meteo_cols:
            for window in [3, 7, 14]:
                df[f'{col}_ma_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
        
        # Remover NaN introduzidos por lags/médias móveis
        df = df.dropna()
        
        return df
    
    def validate_model(self, model_name, data, variable='temperature'):
        """Valida um modelo específico"""
        try:
            if model_name not in self.loaded_models:
                raise ValueError(f"Modelo {model_name} não encontrado")
            
            model = self.loaded_models[model_name]
            
            # Obter features
            features = []
            if model_name in self.model_info and 'features' in self.model_info[model_name]:
                features = self.model_info[model_name]['features']
            elif variable.lower() == 'temperature' and 'temperature' in self.generic_model_info:
                features = self.generic_model_info['temperature']
            elif variable.lower() == 'precipitation' and 'precipitation' in self.generic_model_info:
                features = self.generic_model_info['precipitation']
            else:
                features = [col for col in data.columns if col not in ['date', 'temp_obs', 'prec_obs']]
            
            if not features:
                raise ValueError(f"Não foi possível determinar as features para o modelo {model_name}")

            # Preparar features
            missing_features = []
            for feat in features:
                if feat not in data.columns:
                    missing_features.append(feat)
                    data[feat] = 0
            
            if missing_features:
                print(f"Aviso: Features faltantes para {model_name} preenchidas com 0: {missing_features}")
            
            # Preparar dados para predição
            X = data[features]
            
            # Definir variável alvo
            y_col = 'temp_obs' if 'temperature' in variable.lower() else 'prec_obs'
            y_true = data[y_col]
            
            # Fazer predições
            y_pred = model.predict(X)
            
            # Reverter transformação para precipitação se necessário
            if 'precipitation' in variable.lower() and ('log' in model_name.lower() or 'log1p' in model_name.lower()):
                y_pred = np.expm1(y_pred)
                y_pred[y_pred < 0] = 0
            
            # Garantir arrays numpy para manipulação
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Filtrar NaNs para métricas
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]

            if len(y_true_filtered) == 0:
                print(f"Aviso: Após filtrar NaNs, não há dados suficientes para {model_name}.")
                return None
            
            # Calcular métricas
            metrics = self._calculate_metrics(y_true_filtered, y_pred_filtered, variable)
            
            # Armazenar resultados
            self.validation_results[model_name] = {
                'predictions': y_pred,
                'observations': y_true,
                'dates': data['date'].values,  # Converter para numpy array
                'metrics': metrics,
                'features_used': len(features) - len(missing_features),
                'features_missing': len(missing_features)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Erro ao validar {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_metrics(self, y_true, y_pred, variable):
        """Calcula métricas de validação"""
        if len(y_true) == 0:
            return None
        
        # Métricas básicas
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        
        # Correlação
        if np.var(y_true) == 0:
            correlation = np.nan
            skill_score = np.nan
        else:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            variance = np.var(y_true)
            skill_score = 1 - (rmse**2 / variance) if variance > 0 else 0
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Bias': bias,
            'Correlation': correlation,
            'Skill_Score': skill_score,
            'N_samples': len(y_true)
        }
        
        # Métricas específicas para precipitação
        if variable == 'precipitation':
            rain_threshold = 0.1
            rain_true = y_true > rain_threshold
            rain_pred = y_pred > rain_threshold
            
            if len(rain_true) > 0:
                rain_accuracy = np.mean(rain_true == rain_pred) * 100
            else:
                rain_accuracy = np.nan

            true_positives = np.sum(rain_true & rain_pred)
            false_negatives = np.sum(rain_true & ~rain_pred)
            false_positives = np.sum(~rain_true & rain_pred)

            pod = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else np.nan
            far = false_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else np.nan
            
            metrics.update({
                'Rain_Detection_Accuracy': rain_accuracy,
                'POD': pod,
                'FAR': far
            })
        
        return metrics
    
    def create_validation_plots(self):
        """Cria visualizações interativas dos resultados"""
        plots = {}
        
        # 1. Comparação de modelos
        if self.validation_results:
            try:
                fig_comparison = self._plot_model_comparison()
                plots['comparison'] = fig_comparison.to_json()
            except Exception as e:
                print(f"Erro ao gerar gráfico de comparação: {e}")
                plots['comparison'] = None
        else:
            plots['comparison'] = None
        
        # 2. Série temporal para melhor modelo
        best_model = self._get_best_model()
        if best_model and best_model in self.validation_results:
            try:
                fig_timeseries = self._plot_timeseries(best_model)
                plots['timeseries'] = fig_timeseries.to_json()
            except Exception as e:
                print(f"Erro ao gerar gráfico de série temporal para {best_model}: {e}")
                plots['timeseries'] = None
        else:
            plots['timeseries'] = None
        
        # 3. Scatter plot
        if best_model and best_model in self.validation_results:
            try:
                fig_scatter = self._plot_scatter(best_model)
                plots['scatter'] = fig_scatter.to_json()
            except Exception as e:
                print(f"Erro ao gerar gráfico de dispersão para {best_model}: {e}")
                plots['scatter'] = None
        else:
            plots['scatter'] = None
        
        # 4. Análise de resíduos
        if best_model and best_model in self.validation_results:
            try:
                fig_residuals = self._plot_residuals(best_model)
                plots['residuals'] = fig_residuals.to_json()
            except Exception as e:
                print(f"Erro ao gerar gráfico de resíduos para {best_model}: {e}")
                plots['residuals'] = None
        else:
            plots['residuals'] = None
        
        return plots
    
    def _plot_model_comparison(self):
        """Compara performance de todos os modelos"""
        models = []
        rmse_values = []
        r2_values = []
        
        for model_name, results in self.validation_results.items():
            if results and 'metrics' in results and results['metrics']:
                models.append(model_name)
                rmse_values.append(results['metrics'].get('RMSE', np.nan))
                r2_values.append(results['metrics'].get('R2', np.nan))
        
        if not models:
            fig = go.Figure()
            fig.update_layout(title="Nenhum dado para comparação de modelos.")
            return fig

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('RMSE (menor é melhor)', 'R² (maior é melhor)')
        )
        
        # RMSE
        fig.add_trace(
            go.Bar(x=models, y=rmse_values, name='RMSE', marker_color='lightcoral'),
            row=1, col=1
        )
        
        # R²
        fig.add_trace(
            go.Bar(x=models, y=r2_values, name='R²', marker_color='lightblue'),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Comparação de Performance dos Modelos",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def _plot_timeseries(self, model_name):
        """Plota série temporal de observado vs predito"""
        results = self.validation_results[model_name]
        
        # Converter para arrays numpy se necessário
        observations = np.array(results['observations'])
        predictions = np.array(results['predictions'])
        dates = pd.to_datetime(results['dates'])
        
        # Filtrar NaNs
        valid_indices = ~(np.isnan(observations) | np.isnan(predictions))
        
        if not np.any(valid_indices):
            fig = go.Figure()
            fig.update_layout(title=f"Nenhum dado válido para série temporal de {model_name}.")
            return fig
        
        dates_filtered = dates[valid_indices]
        observations_filtered = observations[valid_indices]
        predictions_filtered = predictions[valid_indices]

        # Limitar pontos para melhor performance
        n_points = min(730, len(dates_filtered))
        if len(dates_filtered) > n_points:
            indices = np.linspace(0, len(dates_filtered)-1, n_points, dtype=int)
            dates_filtered = dates_filtered[indices]
            observations_filtered = observations_filtered[indices]
            predictions_filtered = predictions_filtered[indices]

        fig = go.Figure()
        
        # Observado
        fig.add_trace(go.Scatter(
            x=dates_filtered,
            y=observations_filtered,
            mode='lines',
            name='Observado',
            line=dict(color='blue', width=2)
        ))
        
        # Predito
        fig.add_trace(go.Scatter(
            x=dates_filtered,
            y=predictions_filtered,
            mode='lines',
            name='Predito',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        variable = 'Temperatura (°C)' if 'temp' in model_name.lower() else 'Precipitação (mm)'
        
        fig.update_layout(
            title=f"Série Temporal - {model_name}",
            xaxis_title="Data",
            yaxis_title=variable,
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def _plot_scatter(self, model_name):
        """Plota gráfico de dispersão observado vs predito"""
        results = self.validation_results[model_name]
        
        # Converter para arrays numpy
        observations = np.array(results['observations'])
        predictions = np.array(results['predictions'])
        
        # Filtrar NaNs
        valid_indices = ~(np.isnan(observations) | np.isnan(predictions))
        
        if not np.any(valid_indices):
            fig = go.Figure()
            fig.update_layout(title=f"Nenhum dado válido para gráfico de dispersão de {model_name}.")
            return fig
        
        observations_filtered = observations[valid_indices]
        predictions_filtered = predictions[valid_indices]

        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=observations_filtered,
            y=predictions_filtered,
            mode='markers',
            name='Dados',
            marker=dict(size=5, color='blue', opacity=0.5)
        ))
        
        # Linha 1:1
        min_val = min(observations_filtered.min(), predictions_filtered.min())
        max_val = max(observations_filtered.max(), predictions_filtered.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Linha 1:1',
            line=dict(color='red', dash='dash')
        ))
        
        # Adicionar métricas
        metrics = results['metrics']
        if metrics:
            r2_val = metrics.get('R2', np.nan)
            rmse_val = metrics.get('RMSE', np.nan)
            annotation_text = f"R² = {r2_val:.3f}<br>RMSE = {rmse_val:.3f}"
        else:
            annotation_text = "Métricas não disponíveis"

        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            text=annotation_text,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        variable = 'Temperatura (°C)' if 'temp' in model_name.lower() else 'Precipitação (mm)'
        
        fig.update_layout(
            title=f"Observado vs Predito - {model_name}",
            xaxis_title=f"{variable} Observado",
            yaxis_title=f"{variable} Predito",
            height=400
        )
        
        return fig
    
    def _plot_residuals(self, model_name):
        """Plota análise de resíduos"""
        results = self.validation_results[model_name]
        
        # Converter para arrays numpy
        observations = np.array(results['observations'])
        predictions = np.array(results['predictions'])
        dates = pd.to_datetime(results['dates'])
        
        # Filtrar NaNs
        valid_indices = ~(np.isnan(observations) | np.isnan(predictions))
        
        if not np.any(valid_indices):
            fig = go.Figure()
            fig.update_layout(title=f"Nenhum dado válido para análise de resíduos de {model_name}.")
            return fig
        
        dates_filtered = dates[valid_indices]
        observations_filtered = observations[valid_indices]
        predictions_filtered = predictions[valid_indices]
        residuals = observations_filtered - predictions_filtered

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Distribuição dos Resíduos', 'Resíduos ao Longo do Tempo')
        )
        
        # Histograma
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=30, name='Resíduos', marker_color='lightgreen'),
            row=1, col=1
        )
        
        # Série temporal dos resíduos
        fig.add_trace(
            go.Scatter(x=dates_filtered, y=residuals, mode='markers', name='Resíduos',
                      marker=dict(size=4, color='orange')),
            row=1, col=2
        )
        
        # Linha zero
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        fig.update_layout(
            title_text=f"Análise de Resíduos - {model_name}",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def _get_best_model(self):
        """Retorna o melhor modelo baseado no RMSE"""
        best_model = None
        best_rmse = float('inf')
        
        for model_name, results in self.validation_results.items():
            if results and 'metrics' in results and results['metrics']:
                rmse = results['metrics'].get('RMSE')
                if rmse is not None and rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model_name
        
        return best_model
    
    def generate_validation_report(self):
        """Gera relatório de validação"""
        report = []
        report.append("="*80)
        report.append("RELATÓRIO DE VALIDAÇÃO DE MODELOS CLIMÁTICOS")
        report.append("="*80)
        report.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Informações gerais
        report.append(f"Total de modelos validados: {len(self.validation_results)}")
        
        # Melhor modelo
        best_model = self._get_best_model()
        if best_model:
            best_metrics = self.validation_results[best_model]['metrics']
            report.append(f"\n🏆 MELHOR MODELO: {best_model}")
            report.append(f"   RMSE: {best_metrics.get('RMSE', np.nan):.4f}")
            report.append(f"   R²: {best_metrics.get('R2', np.nan):.4f}")
            report.append(f"   Correlação: {best_metrics.get('Correlation', np.nan):.4f}")
        else:
            report.append("\nNenhum modelo pôde ser determinado como o melhor.")
        
        # Resultados por modelo
        report.append("\n📊 RESULTADOS DETALHADOS:\n")
        
        if not self.validation_results:
            report.append("Nenhum resultado de validação disponível.")
        else:
            for model_name, results in sorted(self.validation_results.items()):
                if results and 'metrics' in results and results['metrics']:
                    metrics = results['metrics']
                    report.append(f"{model_name}:")
                    report.append(f"  ├─ RMSE: {metrics.get('RMSE', np.nan):.4f}")
                    report.append(f"  ├─ MAE: {metrics.get('MAE', np.nan):.4f}")
                    report.append(f"  ├─ R²: {metrics.get('R2', np.nan):.4f}")
                    report.append(f"  ├─ Bias: {metrics.get('Bias', np.nan):.4f}")
                    report.append(f"  ├─ Correlação: {metrics.get('Correlation', np.nan):.4f}")
                    report.append(f"  ├─ Skill Score: {metrics.get('Skill_Score', np.nan):.4f}")
                    report.append(f"  ├─ Amostras: {metrics.get('N_samples', 0)}")
                    report.append(f"  └─ Features: {results.get('features_used', 0)} usadas, {results.get('features_missing', 0)} faltantes\n")
                else:
                    report.append(f"{model_name}: Não foi possível calcular métricas ou resultados incompletos.\n")
        
        # Salvar relatório
        report_text = "\n".join(report)
        report_path = os.path.join('results_validation', 'relatorio_validacao.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text


# Instância global do validador
validator = ModelValidator()

@app.route('/')
def index():
    """Página inicial"""
    return render_template('validation.html')

@app.route('/load_models', methods=['GET'])
def load_models():
    """Carrega modelos disponíveis"""
    try:
        models = validator.load_models()
        model_list = list(models.keys())
        
        return jsonify({
            'status': 'success',
            'models': model_list,
            'count': len(model_list)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/validate', methods=['POST'])
def validate():
    """Executa validação dos modelos"""
    try:
        # Verificar arquivos
        if 'era5' not in request.files or 'station' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Arquivos ERA5 e estação são obrigatórios'
            }), 400
        
        # Salvar arquivos
        era5_file = request.files['era5']
        station_file = request.files['station']
        
        era5_filename = secure_filename(era5_file.filename)
        station_filename = secure_filename(station_file.filename)

        era5_path = os.path.join(app.config['UPLOAD_FOLDER'], era5_filename)
        station_path = os.path.join(app.config['UPLOAD_FOLDER'], station_filename)
        
        era5_file.save(era5_path)
        station_file.save(station_path)
        
        # DEM é opcional
        dem_path = None
        if 'dem' in request.files and request.files['dem'].filename:
            dem_file = request.files['dem']
            dem_filename = secure_filename(dem_file.filename)
            dem_path = os.path.join(app.config['UPLOAD_FOLDER'], dem_filename)
            dem_file.save(dem_path)
        
        # Parâmetros
        lat_str = request.form.get('latitude', '-9.41')
        lon_str = request.form.get('longitude', '-77.35')
        
        try:
            lat = float(lat_str)
            lon = float(lon_str)
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': 'Latitude e Longitude devem ser números válidos.'
            }), 400

        selected_models = request.form.getlist('models[]')
        
        if not selected_models:
            return jsonify({
                'status': 'error',
                'message': 'Nenhum modelo selecionado para validação.'
            }), 400

        # Preparar dados
        print("Preparando dados para validação...")
        data = validator.prepare_validation_data(era5_path, station_path, dem_path, lat, lon)
        
        if data.empty:
            return jsonify({
                'status': 'error',
                'message': 'Nenhum dado válido foi preparado para validação.'
            }), 400

        # Validar cada modelo selecionado
        results_summary = {}
        
        for model_name in selected_models:
            print(f"\nValidando modelo: {model_name}")
            
            # Determinar variável
            variable = 'temperature'
            if 'temp' in model_name.lower():
                variable = 'temperature'
            elif 'prec' in model_name.lower():
                variable = 'precipitation'
            
            metrics = validator.validate_model(model_name, data.copy(), variable) 
            
            if metrics:
                results_summary[model_name] = metrics
                print(f"✓ {model_name} validado com sucesso")
            else:
                print(f"✗ Erro ou dados insuficientes para validar {model_name}")
        
        if not results_summary:
            return jsonify({
                'status': 'error',
                'message': 'Nenhum modelo pôde ser validado com os dados fornecidos.'
            }), 400

        # Gerar visualizações
        plots = validator.create_validation_plots()
        
        # Gerar relatório
        report = validator.generate_validation_report()
        
        return jsonify({
            'status': 'success',
            'results': results_summary,
            'plots': plots,
            'report': report,
            'data_info': {
                'total_records': len(data),
                'latitude': lat,
                'longitude': lon
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f"Ocorreu um erro interno: {str(e)}"
        }), 500

@app.route('/download_validation_results')
def download_validation_results():
    """Download dos resultados de validação"""
    try:
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Adicionar relatório
            report_path = os.path.join('results_validation', 'relatorio_validacao.txt')
            if os.path.exists(report_path):
                zip_file.write(report_path, 'relatorio_validacao.txt')
            
            # Adicionar CSV com predições
            best_model = validator._get_best_model()
            if best_model and best_model in validator.validation_results:
                results = validator.validation_results[best_model]
                
                # Criar DataFrame com resultados
                results_df = pd.DataFrame({
                    'date': results['dates'],
                    'observed': results['observations'],
                    'predicted': results['predictions'],
                    'residual': results['observations'] - results['predictions']
                })
                
                # Salvar temporariamente
                csv_filename = f'predicoes_{best_model}.csv'
                csv_path = os.path.join('results_validation', csv_filename)
                results_df.to_csv(csv_path, index=False)
                zip_file.write(csv_path, csv_filename)
        
        zip_buffer.seek(0)
        return send_file(zip_buffer,
                         mimetype='application/zip',
                         as_attachment=True,
                         download_name='resultados_validacao.zip')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f"Erro ao baixar resultados: {str(e)}"
        }), 400

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    
    # Features fornecidas pelo usuário
    generic_temp_features = [
        "z_300", "z_400", "z_500", "q_400", "t_300", "t_400", "t_500", "v_400", "v_500", "t2m",
        "tp", "ssrd", "sf", "year", "month", "day_of_year", "week_of_year", "month_sin",
        "day_sin", "days_since_start", "t2m_lag_1", "t2m_lag_2", "t2m_lag_3", "t2m_lag_7",
        "t2m_lag_14", "tp_lag_1", "tp_lag_2", "sp_lag_1", "sp_lag_2", "sp_lag_3", "d2m_lag_14",
        "ssrd_lag_1", "ssrd_lag_2", "ssrd_lag_3", "t2m_ma_3", "t2m_min_3", "t2m_max_3",
        "t2m_ma_7", "t2m_min_7", "t2m_max_7", "t2m_ma_14", "t2m_min_14", "t2m_max_14",
        "t2m_ma_30", "t2m_min_30", "t2m_max_30", "tp_ma_3", "tp_min_3", "tp_max_3",
        "tp_min_7", "tp_std_30", "tp_max_30", "sp_ma_3", "sp_min_3", "sp_max_3", "sp_ma_7",
        "sp_min_7", "sp_max_7", "sp_ma_14", "sp_min_14", "sp_max_14", "sp_ma_30", "sp_min_30",
        "sp_max_30", "v10_min_14", "d2m_min_14", "d2m_max_14", "d2m_ma_30", "d2m_min_30",
        "d2m_max_30", "sp_t2m_interaction", "dewpoint_spread", "t2m_p25_30", "t2m_p75_30",
        "sp_p25_30", "sp_p75_30", "d2m_p25_30", "t2m_corrected", "orographic_effect",
        "thermal_amplitude"
    ]

    generic_prec_features = [
        "q_300", "q_400", "q_500", "t_500", "u10", "d2m", "tp", "strd", "sf", "month_cos",
        "day_cos", "tp_lag_1", "tp_lag_2", "tp_lag_3", "sp_lag_1", "sp_lag_2", "sp_lag_3",
        "sp_lag_14", "sp_lag_30", "tp_ma_3", "tp_std_3", "tp_min_3", "tp_max_3", "tp_sum_3",
        "tp_ma_7", "tp_std_7", "tp_min_7", "tp_max_7", "tp_sum_7", "tp_ma_14", "tp_std_14",
        "tp_min_14", "tp_max_14", "tp_sum_14", "tp_ma_30", "tp_std_30", "tp_min_30",
        "tp_max_30", "tp_sum_30", "tp_ma_60", "tp_std_60", "tp_max_60", "tp_sum_60",
        "sp_ma_3", "sp_min_3", "sp_max_3", "sp_ma_7", "sp_min_7", "sp_max_7", "sp_ma_14",
        "sp_min_14", "sp_max_14", "sp_ma_30", "sp_min_30", "sp_max_30", "sp_ma_60",
        "sp_min_60", "sp_max_60", "u10_ma_3", "u10_min_3", "u10_max_3", "u10_ma_7",
        "u10_min_7", "u10_max_7", "u10_ma_14", "u10_max_14", "u10_ma_30", "u10_max_30",
        "u10_ma_60", "u10_std_60", "u10_max_60", "v10_ma_30", "v10_ma_60", "v10_std_60",
        "wind_speed", "tp_p25_30", "tp_p75_30", "sp_p25_30", "sp_p75_30", "orographic_effect"
    ]

    # Criar arquivos de informação de features (apenas se não existirem)
    temp_info_path = 'models/model_info_temperature.json'
    if not os.path.exists(temp_info_path):
        with open(temp_info_path, 'w') as f:
            json.dump({'features': generic_temp_features}, f, indent=2)
        print("Arquivo 'model_info_temperature.json' criado.")

    prec_info_path = 'models/model_info_precipitation.json'
    if not os.path.exists(prec_info_path):
        with open(prec_info_path, 'w') as f:
            json.dump({'features': generic_prec_features}, f, indent=2)
        print("Arquivo 'model_info_precipitation.json' criado.")

    validator.load_models()
    app.run(debug=True)