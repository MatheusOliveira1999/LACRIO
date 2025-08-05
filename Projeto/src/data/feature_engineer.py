"""
Módulo para engenharia de features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


class FeatureEngineer:
    """Classe para criação de features meteorológicas"""
    
    def __init__(self, variable: str = 'temperature'):
        """
        Inicializa o engenheiro de features
        
        Parameters:
        -----------
        variable : str
            Variável climática ('temperature' ou 'precipitation')
        """
        self.variable = variable
        self.created_features = []
    
    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cria todas as features para o modelo
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame com dados originais
            
        Returns:
        --------
        pd.DataFrame : DataFrame com features criadas
        """
        print(f"🔧 Criando features para {self.variable}...")
        
        # Fazer uma cópia para não modificar original
        df = data.copy()
        
        # Normalizar variáveis de escala muito grande
        df = self._normalize_large_scale_variables(df)
        
        # Features temporais
        df = self._create_temporal_features(df)
        
        # Features de lag
        df = self._create_lag_features(df)
        
        # Features de média móvel
        df = self._create_rolling_features(df)
        
        # Features de interação
        df = self._create_interaction_features(df)
        
        # Features estatísticas
        df = self._create_statistical_features(df)
        
        # Features topográficas
        df = self._create_topographic_features(df)
        
        # Remover NaN
        initial_shape = df.shape
        df = df.dropna()
        final_shape = df.shape
        
        print(f"Features criadas. Shape: {initial_shape} -> {final_shape}")
        print(f"Registros removidos por NaN: {initial_shape[0] - final_shape[0]}")
        
        return df
    
    def _normalize_large_scale_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza variáveis de escala muito grande"""
        df = df.copy()
        
        # Variáveis de grande escala
        large_scale_vars = ['sp', 'z', 'z_300.0', 'z_400.0', 'z_500.0', 'ssrd', 'strd']
        
        for var in large_scale_vars:
            if var in df.columns:
                # Converter pressão de Pa para hPa
                if var in ['sp', 'z', 'z_300.0', 'z_400.0', 'z_500.0']:
                    df[var] = df[var] / 100.0
                # Converter radiação acumulada para média
                elif var in ['ssrd', 'strd']:
                    df[var] = df[var] / 86400.0  # J/m² para W/m²
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
        
        # Registrar features criadas
        temporal_features = ['year', 'month', 'day_of_year', 'week_of_year', 'season',
                           'month_sin', 'month_cos', 'day_sin', 'day_cos', 'days_since_start']
        self.created_features.extend(temporal_features)
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
                feature_name = f'{col}_lag_{lag}'
                df[feature_name] = df[col].shift(lag)
                self.created_features.append(feature_name)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
                feature_name = f'{col}_ma_{window}'
                df[feature_name] = df[col].rolling(window=window, min_periods=1).mean()
                self.created_features.append(feature_name)
                
                # Desvio padrão móvel
                feature_name = f'{col}_std_{window}'
                df[feature_name] = df[col].rolling(window=window, min_periods=1).std()
                self.created_features.append(feature_name)
                
                # Mínimo e máximo móveis
                feature_name = f'{col}_min_{window}'
                df[feature_name] = df[col].rolling(window=window, min_periods=1).min()
                self.created_features.append(feature_name)
                
                feature_name = f'{col}_max_{window}'
                df[feature_name] = df[col].rolling(window=window, min_periods=1).max()
                self.created_features.append(feature_name)
                
                # Para precipitação, adicionar soma acumulada
                if self.variable == 'precipitation' and col == 'tp':
                    feature_name = f'{col}_sum_{window}'
                    df[feature_name] = df[col].rolling(window=window, min_periods=1).sum()
                    self.created_features.append(feature_name)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de interação entre variáveis"""
        df = df.copy()
        
        # Velocidade do vento (para todas as variáveis)
        if 'u10' in df.columns and 'v10' in df.columns:
            df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
            df['wind_direction'] = np.arctan2(df['v10'], df['u10'])
            self.created_features.extend(['wind_speed', 'wind_direction'])
        
        # Interações específicas por variável
        if self.variable == 'temperature':
            if 't2m' in df.columns and 'r' in df.columns:
                df['t2m_r_interaction'] = df['t2m'] * df['r']
                self.created_features.append('t2m_r_interaction')
                
            if 'sp' in df.columns and 't2m' in df.columns:
                df['sp_t2m_interaction'] = df['sp'] * df['t2m']
                self.created_features.append('sp_t2m_interaction')
                
            if 'd2m' in df.columns and 't2m' in df.columns:
                df['dewpoint_spread'] = df['t2m'] - df['d2m']
                self.created_features.append('dewpoint_spread')
                
        else:  # precipitation
            if 'tcwv' in df.columns and 'cape' in df.columns:
                df['tcwv_cape_interaction'] = df['tcwv'] * df['cape']
                self.created_features.append('tcwv_cape_interaction')
                
            if 'q' in df.columns and 'sp' in df.columns:
                df['q_sp_interaction'] = df['q'] * df['sp']
                self.created_features.append('q_sp_interaction')
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
            feature_name = f'{col}_p25_{window}'
            df[feature_name] = rolling.quantile(0.25)
            self.created_features.append(feature_name)
            
            feature_name = f'{col}_p75_{window}'
            df[feature_name] = rolling.quantile(0.75)
            self.created_features.append(feature_name)
            
            # Skewness e kurtosis (quando possível)
            try:
                feature_name = f'{col}_skew_{window}'
                df[feature_name] = rolling.skew()
                self.created_features.append(feature_name)
            except:
                pass
        
        return df
    
    def _create_topographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features específicas para regiões montanhosas"""
        df = df.copy()
        
        # Features baseadas na diferença de altitude
        if 'alt_diff' in df.columns:
            # Correção de temperatura baseada na altitude (lapse rate)
            if 't2m' in df.columns:
                lapse_rate = 0.0065  # K/m (taxa de diminuição padrão)
                df['t2m_corrected'] = df['t2m'] - (lapse_rate * df['alt_diff'])
                self.created_features.append('t2m_corrected')
            
            # Efeito orográfico na precipitação
            if 'tp' in df.columns and 'wind_speed' in df.columns:
                df['orographic_effect'] = df['alt_diff'] * df['wind_speed'] * df['tp']
                self.created_features.append('orographic_effect')
        
        # Features sazonais específicas para montanhas
        if self.variable == 'temperature':
            # Amplitude térmica diária estimada (maior em altitudes elevadas)
            if 'ssrd' in df.columns:  # Radiação solar
                df['thermal_amplitude'] = df['ssrd'] * 0.1  # Simplificado
                self.created_features.append('thermal_amplitude')
        
        return df
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """
        Retorna categorias das features criadas
        
        Returns:
        --------
        dict : Dicionário com categorias de features
        """
        categories = {
            'temporal': [],
            'lag': [],
            'rolling': [],
            'interaction': [],
            'statistical': [],
            'topographic': []
        }
        
        for feature in self.created_features:
            if any(x in feature for x in ['year', 'month', 'day', 'season', 'sin', 'cos']):
                categories['temporal'].append(feature)
            elif 'lag_' in feature:
                categories['lag'].append(feature)
            elif any(x in feature for x in ['_ma_', '_std_', '_min_', '_max_', '_sum_']):
                categories['rolling'].append(feature)
            elif 'interaction' in feature or feature in ['wind_speed', 'wind_direction', 'dewpoint_spread']:
                categories['interaction'].append(feature)
            elif any(x in feature for x in ['_p25_', '_p75_', '_skew_']):
                categories['statistical'].append(feature)
            elif any(x in feature for x in ['corrected', 'orographic', 'thermal']):
                categories['topographic'].append(feature)
        
        return categories
    
    def print_feature_summary(self):
        """Imprime resumo das features criadas"""
        categories = self.get_feature_categories()
        
        print(f"\n📊 RESUMO DAS FEATURES CRIADAS ({self.variable.upper()})")
        print("=" * 50)
        
        total_features = 0
        for category, features in categories.items():
            if features:
                print(f"{category.upper()}: {len(features)} features")
                total_features += len(features)
        
        print(f"\nTOTAL: {total_features} features criadas")
        print("=" * 50)


def create_features_for_variable(data: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Função utilitária para criar features para uma variável específica
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dados originais
    variable : str
        Variável climática
        
    Returns:
    --------
    pd.DataFrame : Dados com features criadas
    """
    engineer = FeatureEngineer(variable=variable)
    result = engineer.create_all_features(data)
    engineer.print_feature_summary()
    
    return result


if __name__ == "__main__":
    # Teste básico
    print("FeatureEngineer inicializado com sucesso!")
    
    # Criar um DataFrame de exemplo
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        't2m': np.random.normal(15, 5, len(dates)),
        'tp': np.random.exponential(2, len(dates)),
        'sp': np.random.normal(101325, 1000, len(dates)),
        'u10': np.random.normal(0, 3, len(dates)),
        'v10': np.random.normal(0, 3, len(dates)),
        'alt_diff': 500  # Diferença de altitude constante
    })
    
    # Testar engenharia de features
    engineer = FeatureEngineer('temperature')
    result = engineer.create_all_features(sample_data)
    engineer.print_feature_summary()
    
    print(f"Shape original: {sample_data.shape}")
    print(f"Shape com features: {result.shape}")