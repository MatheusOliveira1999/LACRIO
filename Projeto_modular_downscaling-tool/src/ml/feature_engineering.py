"""
feature_engineering.py
Módulo para criação de features
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Cria features para modelos de downscaling"""
    
    def __init__(self, variable='temperature'):
        self.variable = variable
    
    def create_all_features(self, data):
        """Cria todas as features para o modelo"""
        print(f"Criando features para {self.variable}...")
        
        # Normalizar variáveis de escala muito grande
        data = self._normalize_large_scale_vars(data)
        
        # Features temporais
        data = self._create_temporal_features(data)
        
        # Features de lag
        data = self._create_lag_features(data)
        
        # Features de média móvel
        data = self._create_rolling_features(data)
        
        # Features de interação
        data = self._create_interaction_features(data)
        
        # Features estatísticas
        data = self._create_statistical_features(data)
        
        # Features específicas para regiões montanhosas
        data = self._create_topographic_features(data)
        
        # Remover NaN
        initial_shape = data.shape
        data = data.dropna()
        final_shape = data.shape
        
        print(f"Features criadas. Shape: {initial_shape} -> {final_shape}")
        print(f"Registros removidos por NaN: {initial_shape[0] - final_shape[0]}")
        
        return data
    
    def _normalize_large_scale_vars(self, data):
        """Normaliza variáveis de escala muito grande"""
        large_scale_vars = ['sp', 'z', 'z_300.0', 'z_400.0', 'z_500.0', 'ssrd', 'strd']
        for var in large_scale_vars:
            if var in data.columns:
                if var in ['sp', 'z', 'z_300.0', 'z_400.0', 'z_500.0']:
                    data[var] = data[var] / 100.0
                elif var in ['ssrd', 'strd']:
                    data[var] = data[var] / 86400.0
        return data
    
    def _create_temporal_features(self, df):
        """Cria features temporais"""
        df = df.copy()
        
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
        
        start_date = df['date'].min()
        df['days_since_start'] = (df['date'] - start_date).dt.days
        
        return df
    
    def _create_lag_features(self, df):
        """Cria features de lag temporal"""
        df = df.copy()
        
        if self.variable == 'temperature':
            meteo_cols = ['t2m', 'tp', 'sp', 'u10', 'v10', 'd2m', 'r', 'ssrd', 'strd']
            lags = [1, 2, 3, 7, 14]
        else:
            meteo_cols = ['tp', 'tcwv', 'cape', 'cin', 'sp', 'u10', 'v10', 'q', 'pev']
            lags = [1, 2, 3, 7, 14, 30]
        
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
        else:
            meteo_cols = ['tp', 'tcwv', 'cape', 'sp', 'u10', 'v10']
            windows = [3, 7, 14, 30, 60]
        
        meteo_cols = [col for col in meteo_cols if col in df.columns]
        
        for col in meteo_cols:
            for window in windows:
                df[f'{col}_ma_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                df[f'{col}_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                
                if self.variable == 'precipitation' and col == 'tp':
                    df[f'{col}_sum_{window}'] = df[col].rolling(window=window, min_periods=1).sum()
        
        return df
    
    def _create_interaction_features(self, df):
        """Cria features de interação entre variáveis"""
        df = df.copy()
        
        if 'u10' in df.columns and 'v10' in df.columns:
            df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
            df['wind_direction'] = np.arctan2(df['v10'], df['u10'])
        
        if self.variable == 'temperature':
            if 't2m' in df.columns and 'r' in df.columns:
                df['t2m_r_interaction'] = df['t2m'] * df['r']
            if 'sp' in df.columns and 't2m' in df.columns:
                df['sp_t2m_interaction'] = df['sp'] * df['t2m']
            if 'd2m' in df.columns and 't2m' in df.columns:
                df['dewpoint_spread'] = df['t2m'] - df['d2m']
        else:
            if 'tcwv' in df.columns and 'cape' in df.columns:
                df['tcwv_cape_interaction'] = df['tcwv'] * df['cape']
            if 'q' in df.columns and 'sp' in df.columns:
                df['q_sp_interaction'] = df['q'] * df['sp']
        
        return df
    
    def _create_statistical_features(self, df):
        """Cria features estatísticas avançadas"""
        df = df.copy()
        
        if self.variable == 'temperature':
            main_vars = ['t2m', 'sp', 'r', 'd2m']
        else:
            main_vars = ['tp', 'tcwv', 'cape', 'sp']
        
        main_vars = [col for col in main_vars if col in df.columns]
        
        for col in main_vars:
            window = 30
            rolling = df[col].rolling(window=window, min_periods=1)
            
            df[f'{col}_p25_{window}'] = rolling.quantile(0.25)
            df[f'{col}_p75_{window}'] = rolling.quantile(0.75)
            
            try:
                df[f'{col}_skew_{window}'] = rolling.skew()
            except:
                pass
        
        return df
    
    def _create_topographic_features(self, df):
        """Cria features específicas para regiões montanhosas"""
        df = df.copy()
        
        if 'alt_diff' in df.columns:
            if 't2m' in df.columns:
                lapse_rate = 0.0065
                df['t2m_corrected'] = df['t2m'] - (lapse_rate * df['alt_diff'])
            
            if 'tp' in df.columns and 'wind_speed' in df.columns:
                df['orographic_effect'] = df['alt_diff'] * df['wind_speed'] * df['tp']
        
        if self.variable == 'temperature':
            if 'ssrd' in df.columns:
                df['thermal_amplitude'] = df['ssrd'] * 0.1
        
        return df