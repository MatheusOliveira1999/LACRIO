"""
Modelo principal de downscaling climático
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression

# Imports locais
from ..data.loader import DataLoader
from ..data.feature_engineer import FeatureEngineer
from ..ml.model_factory import ModelFactory
from ..ml.metrics import MetricsCalculator
from ..ml.optimizer import HyperparameterOptimizer
from ..ml.ensemble import EnsembleCreator
from ..utils.validation import DataValidator
from config.settings import DEFAULT_MODEL_CONFIG


class WeatherDownscalingModel:
    """
    Classe principal para downscaling de dados climáticos usando Machine Learning
    """
    
    def __init__(self, variable: str = 'temperature', config: Dict[str, Any] = None):
        """
        Inicializa o modelo de downscaling
        
        Parameters:
        -----------
        variable : str
            'temperature' ou 'precipitation'
        config : dict
            Configurações do modelo
        """
        self.variable = variable
        self.config = {**DEFAULT_MODEL_CONFIG, **(config or {})}
        
        # Componentes principais
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer(variable)
        self.model_factory = ModelFactory(variable, self.config['random_state'])
        self.metrics_calculator = MetricsCalculator(variable)
        self.optimizer = HyperparameterOptimizer(variable, self.config)
        self.ensemble_creator = EnsembleCreator()
        self.validator = DataValidator()
        
        # Estado do modelo
        self.data = None
        self.features = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_selector = None
        
        # Dados de treino/teste
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_dates = None
        self.test_dates = None
    
    def load_and_merge_data(self, era5_path: str, station_path: str, 
                           dem_path: str, lat: float, lon: float) -> pd.DataFrame:
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
            
        Returns:
        --------
        pd.DataFrame : Dados mesclados
        """
        print(f"🔄 Carregando dados para {self.variable}...")
        
        try:
            # Carregar todos os dados
            self.data = self.data_loader.load_all_data(
                era5_path, station_path, dem_path, lat, lon
            )
            
            # Validar dados
            self._validate_loaded_data()
            
            print(f"✅ Dados carregados com sucesso: {self.data.shape}")
            return self.data
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {str(e)}")
            raise
    
    def create_features(self) -> pd.DataFrame:
        """
        Cria todas as features para o modelo
        
        Returns:
        --------
        pd.DataFrame : Dados com features criadas
        """
        if self.data is None:
            raise ValueError("❌ Dados não foram carregados. Execute load_and_merge_data() primeiro.")
        
        print(f"🔧 Criando features para {self.variable}...")
        
        try:
            # Criar features usando o FeatureEngineer
            self.data = self.feature_engineer.create_all_features(self.data)
            
            # Mostrar resumo das features
            self.feature_engineer.print_feature_summary()
            
            return self.data
            
        except Exception as e:
            print(f"❌ Erro ao criar features: {str(e)}")
            raise
    
    def prepare_data(self, split_date: str = '2018-01-01') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepara dados para modelagem
        
        Parameters:
        -----------
        split_date : str
            Data para divisão treino/teste
            
        Returns:
        --------
        tuple : X_train, X_test, y_train, y_test
        """
        if self.data is None:
            raise ValueError("❌ Dados não foram preparados. Execute create_features() primeiro.")
        
        print("📊 Preparando dados para modelagem...")
        
        try:
            # Definir variável alvo
            target_col = 'temp_obs' if self.variable == 'temperature' else 'prec_obs'
            
            # Selecionar features (excluir colunas não numéricas e alvo)
            exclude_cols = ['date', 'temp_obs', 'prec_obs']
            potential_features = [c for c in self.data.columns if c not in exclude_cols]
            
            # Filtrar apenas features numéricas
            self.features = []
            for col in potential_features:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    self.features.append(col)
            
            print(f"📈 Features selecionadas: {len(self.features)}")
            
            # Preparar X e y
            X = self.data[self.features].copy()
            y = self.data[target_col].copy()
            
            # Limpar dados
            X, y = self._clean_data(X, y)
            
            # Tratamento especial para precipitação
            if self.variable == 'precipitation':
                y = np.log1p(y)  # Log transform
            
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
            
            print(f"✅ Dados preparados - Treino: {self.X_train.shape}, Teste: {self.X_test.shape}")
            
            return self.X_train, self.X_test, self.y_train, self.y_test
            
        except Exception as e:
            print(f"❌ Erro ao preparar dados: {str(e)}")
            raise
    
    def train_models(self, selected_models: List[str] = None, 
                    optimize: bool = None) -> Dict[str, Any]:
        """
        Treina todos os modelos selecionados
        
        Parameters:
        -----------
        selected_models : list, optional
            Lista de modelos para treinar
        optimize : bool, optional
            Se deve otimizar hiperparâmetros
            
        Returns:
        --------
        dict : Resultados dos modelos treinados
        """
        if self.X_train is None:
            raise ValueError("❌ Dados não foram preparados. Execute prepare_data() primeiro.")
        
        if optimize is None:
            optimize = self.config['optimize_hyperparams']
        
        print(f"\n🤖 Treinando modelos para {self.variable}...")
        
        try:
            # Criar modelos
            self.models = self.model_factory.create_all_models(selected_models)
            print(f"📋 Modelos a treinar: {list(self.models.keys())}")
            
            # Treinar cada modelo
            for name, pipeline in self.models.items():
                print(f"\n🔄 Treinando {name}...")
                
                try:
                    # Otimizar hiperparâmetros se solicitado
                    if optimize and self._should_optimize_model(name):
                        print(f"  ⚙️ Otimizando hiperparâmetros...")
                        pipeline = self.optimizer.optimize_model(name, pipeline, 
                                                               self.X_train, self.y_train)
                    
                    # Treinar modelo final
                    print(f"  🏋️ Treinando modelo final...")
                    pipeline.fit(self.X_train, self.y_train)
                    
                    # Fazer predições
                    y_pred = pipeline.predict(self.X_test)
                    
                    # Reverter transformação se necessário
                    y_pred_original, y_test_original = self._revert_transformations(y_pred)
                    
                    # Calcular métricas
                    metrics = self.metrics_calculator.calculate_all_metrics(
                        y_test_original, y_pred_original
                    )
                    
                    # Armazenar resultados
                    self.results[name] = {
                        'model': pipeline,
                        'predictions': y_pred_original,
                        'metrics': metrics
                    }
                    
                    print(f"  ✅ RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f}")
                    
                except Exception as e:
                    print(f"  ❌ Erro ao treinar {name}: {str(e)}")
                    continue
            
            # Determinar melhor modelo
            if self.results:
                self.best_model = min(self.results.items(), 
                                    key=lambda x: x[1]['metrics']['RMSE'])[0]
                print(f"\n🏆 Melhor modelo: {self.best_model}")
            
            return self.results
            
        except Exception as e:
            print(f"❌ Erro no treinamento: {str(e)}")
            raise
    
    def create_ensemble(self, ensemble_size: int = None) -> Any:
        """
        Cria modelo ensemble dos melhores modelos
        
        Parameters:
        -----------
        ensemble_size : int, optional
            Número de modelos no ensemble
            
        Returns:
        --------
        Ensemble model
        """
        if not self.results:
            print("❌ Nenhum modelo treinado para criar ensemble")
            return None
        
        if ensemble_size is None:
            ensemble_size = self.config['ensemble_size']
        
        print(f"\n🔗 Criando ensemble com {ensemble_size} modelos...")
        
        try:
            # Criar ensemble
            ensemble = self.ensemble_creator.create_voting_ensemble(
                self.results, ensemble_size
            )
            
            # Treinar ensemble
            ensemble.fit(self.X_train, self.y_train)
            
            # Avaliar ensemble
            y_pred = ensemble.predict(self.X_test)
            y_pred_original, y_test_original = self._revert_transformations(y_pred)
            
            # Calcular métricas
            metrics = self.metrics_calculator.calculate_all_metrics(
                y_test_original, y_pred_original
            )
            
            # Armazenar resultado
            self.results['Ensemble'] = {
                'model': ensemble,
                'predictions': y_pred_original,
                'metrics': metrics
            }
            
            print(f"✅ Ensemble criado - RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f}")
            
            return ensemble
            
        except Exception as e:
            print(f"❌ Erro ao criar ensemble: {str(e)}")
            return None
    
    def save_models(self, directory: str = 'models') -> List[str]:
        """
        Salva todos os modelos treinados
        
        Parameters:
        -----------
        directory : str
            Diretório para salvar modelos
            
        Returns:
        --------
        list : Lista de arquivos salvos
        """
        if not self.results:
            print("❌ Nenhum modelo para salvar")
            return []
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        saved_files = []
        
        print(f"💾 Salvando modelos em {directory}...")
        
        # Salvar cada modelo
        for name, result in self.results.items():
            try:
                model_path = os.path.join(directory, f'{name.lower()}_{self.variable}.pkl')
                joblib.dump(result['model'], model_path)
                saved_files.append(model_path)
                print(f"  ✅ {name} salvo")
            except Exception as e:
                print(f"  ❌ Erro ao salvar {name}: {str(e)}")
        
        # Salvar informações do modelo
        try:
            model_info = self._create_model_info()
            info_path = os.path.join(directory, f'model_info_{self.variable}.json')
            
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
            
            saved_files.append(info_path)
            print(f"  ✅ Informações salvas")
            
        except Exception as e:
            print(f"  ❌ Erro ao salvar informações: {str(e)}")
        
        print(f"💾 {len(saved_files)} arquivos salvos")
        return saved_files
    
    def load_model(self, model_path: str, info_path: str = None) -> Any:
        """
        Carrega modelo salvo
        
        Parameters:
        -----------
        model_path : str
            Caminho para arquivo do modelo
        info_path : str, optional
            Caminho para arquivo de informações
            
        Returns:
        --------
        Modelo carregado
        """
        try:
            model = joblib.load(model_path)
            print(f"✅ Modelo carregado: {model_path}")
            
            if info_path and os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    info = json.load(f)
                self.features = info.get('features', [])
                self.config.update(info.get('config', {}))
                print(f"✅ Informações carregadas")
            
            return model
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {str(e)}")
            return None
    
    def predict_new_data(self, new_data_path: str, model_name: str = None) -> pd.DataFrame:
        """
        Faz predições em novos dados
        
        Parameters:
        -----------
        new_data_path : str
            Caminho para novos dados
        model_name : str, optional
            Nome do modelo a usar
            
        Returns:
        --------
        pd.DataFrame : Predições
        """
        if not self.results:
            print("❌ Nenhum modelo treinado disponível")
            return None
        
        # Selecionar modelo
        if model_name is None or model_name not in self.results:
            model_name = self.best_model or list(self.results.keys())[0]
        
        print(f"🔮 Fazendo predições com {model_name}...")
        
        try:
            # Carregar novos dados
            new_data = pd.read_csv(new_data_path, parse_dates=['date'])
            
            # Verificar features
            missing_features = [f for f in self.features if f not in new_data.columns]
            if missing_features:
                print(f"❌ Features faltantes: {missing_features}")
                return None
            
            # Preparar dados
            X_new = new_data[self.features]
            
            # Fazer predições
            model = self.results[model_name]['model']
            predictions = model.predict(X_new)
            
            # Reverter transformações
            if self.variable == 'precipitation':
                predictions = np.expm1(predictions)
            
            # Criar DataFrame com resultados
            results_df = pd.DataFrame({
                'date': new_data['date'],
                f'{self.variable}_predicted': predictions
            })
            
            print(f"✅ {len(predictions)} predições realizadas")
            return results_df
            
        except Exception as e:
            print(f"❌ Erro nas predições: {str(e)}")
            return None
    
    # Métodos auxiliares privados
    def _validate_loaded_data(self):
        """Valida os dados carregados"""
        if self.data is None or self.data.empty:
            raise ValueError("Dados estão vazios")
        
        # Verificar número mínimo de registros
        min_records = 365
        if len(self.data) < min_records:
            raise ValueError(f"Dados insuficientes: {len(self.data)} registros (mínimo: {min_records})")
        
        # Verificar variável alvo
        target_col = 'temp_obs' if self.variable == 'temperature' else 'prec_obs'
        if target_col not in self.data.columns:
            raise ValueError(f"Coluna alvo '{target_col}' não encontrada")
        
        valid_data = self.data[target_col].notna().sum()
        if valid_data < min_records:
            raise ValueError(f"Dados válidos insuficientes: {valid_data} registros")
        
        print(f"✅ Validação concluída: {valid_data} registros válidos")
    
    def _clean_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Limpa dados removendo valores inválidos"""
        # Remover valores infinitos e NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)
        
        # Máscara para valores válidos
        nan_mask = X.isna().any(axis=1) | y.isna()
        if nan_mask.sum() > 0:
            print(f"⚠️ Removendo {nan_mask.sum()} linhas com valores inválidos")
            X = X[~nan_mask]
            y = y[~nan_mask]
            self.data = self.data[~nan_mask]
        
        return X, y
    
    def _select_features(self):
        """Seleciona as melhores features"""
        max_features = min(self.config['max_features'], len(self.features))
        
        print(f"🔍 Selecionando {max_features} melhores features...")
        
        # Usar SelectKBest
        self.feature_selector = SelectKBest(score_func=f_regression, k=max_features)
        
        # Ajustar e transformar
        X_train_selected = self.feature_selector.fit_transform(self.X_train, self.y_train)
        X_test_selected = self.feature_selector.transform(self.X_test)
        
        # Obter features selecionadas
        selected_features = [self.features[i] for i in self.feature_selector.get_support(indices=True)]
        
        # Atualizar dados
        self.X_train = pd.DataFrame(X_train_selected, columns=selected_features, index=self.X_train.index)
        self.X_test = pd.DataFrame(X_test_selected, columns=selected_features, index=self.X_test.index)
        self.features = selected_features
        
        print(f"✅ {len(self.features)} features selecionadas")
    
    def _should_optimize_model(self, model_name: str) -> bool:
        """Determina se deve otimizar um modelo específico"""
        if not self.config['optimize_hyperparams']:
            return False
        
        if self.config['optimize_all_models']:
            return True
        
        # Lista de modelos que vale a pena otimizar
        optimize_list = ['RandomForest', 'XGBoost', 'Ridge', 'GradientBoosting', 
                        'ExtraTrees', 'SVR', 'MLP']
        
        return model_name in optimize_list
    
    def _revert_transformations(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reverte transformações aplicadas aos dados"""
        if self.variable == 'precipitation':
            y_pred_original = np.expm1(y_pred)
            y_test_original = np.expm1(self.y_test)
        else:
            y_pred_original = y_pred
            y_test_original = self.y_test
        
        return y_pred_original, y_test_original
    
    def _create_model_info(self) -> Dict[str, Any]:
        """Cria dicionário com informações do modelo"""
        return {
            'variable': self.variable,
            'features': self.features,
            'config': self.config,
            'results': {name: result['metrics'] for name, result in self.results.items()},
            'best_model': self.best_model,
            'data_info': {
                'train_size': len(self.X_train) if self.X_train is not None else 0,
                'test_size': len(self.X_test) if self.X_test is not None else 0,
                'n_features': len(self.features) if self.features else 0
            },
            'created_at': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Teste básico
    print("🧪 Testando WeatherDownscalingModel...")
    
    model = WeatherDownscalingModel('temperature')
    print("✅ Modelo inicializado com sucesso!")