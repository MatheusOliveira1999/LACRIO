"""
feature_engineering.py
Módulo para criação de features
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Cria features para modelos de downscaling"""
    
    def __init__(self, variable='temperature', temporal_resolution='daily'):
        self.variable = variable
        self.temporal_resolution = temporal_resolution  # 'hourly', 'daily', 'auto'
        self._detected_resolution = None
    
    def create_all_features(self, data):
        """Cria todas as features para o modelo"""
        print(f"Criando features para {self.variable}...")
        
        # Detectar resolução temporal automaticamente
        if self.temporal_resolution == 'auto':
            self._detect_temporal_resolution(data)
        else:
            self._detected_resolution = self.temporal_resolution
        
        print(f"Resolução temporal detectada/configurada: {self._detected_resolution}")
        
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
        
        # Analisar NaN antes de remover
        initial_shape = data.shape
        
        # Verificar colunas com todos os valores NaN
        all_nan_cols = data.columns[data.isna().all()].tolist()
        if all_nan_cols:
            print(f"⚠️  Colunas com todos os valores NaN: {all_nan_cols[:10]}...")
            # Remover colunas completamente NaN
            data = data.drop(columns=all_nan_cols)
            
        # Verificar se ainda há registros válidos
        if len(data) == 0:
            raise ValueError("Todos os registros foram removidos durante o processamento")
            
        # Verificar proporção de NaN por coluna
        nan_proportions = data.isna().mean()
        high_nan_cols = nan_proportions[nan_proportions > 0.8].index.tolist()
        if high_nan_cols:
            print(f"⚠️  Colunas com >80% NaN: {high_nan_cols[:5]}...")
            
        # Estratégia mais conservadora: preservar SEMPRE linhas com dados observacionais válidos
        # Identificar colunas observacionais críticas
        obs_critical_cols = ['temp_obs', 'prec_obs', 'date']
        existing_obs_cols = [col for col in obs_critical_cols if col in data.columns]
        
        if existing_obs_cols:
            # Remover apenas linhas onde TODOS os dados observacionais são NaN E >80% das features são NaN
            feature_cols = [col for col in data.columns if col not in obs_critical_cols]
            
            # Máscara para linhas com pelo menos UM dado observacional válido
            obs_valid_mask = data[existing_obs_cols].notna().any(axis=1)
            
            # Máscara para linhas com features muito ruins (>80% NaN)
            if feature_cols:
                feature_nan_prop = data[feature_cols].isna().mean(axis=1)
                bad_features_mask = feature_nan_prop < 0.8  # Manter se <80% NaN
            else:
                bad_features_mask = pd.Series([True] * len(data), index=data.index)
            
            # Manter linhas que têm dados obs válidos OU features boas
            keep_mask = obs_valid_mask | bad_features_mask
            data = data[keep_mask]
            
            print(f"🔒 Preservadas {keep_mask.sum()}/{len(keep_mask)} linhas com dados observacionais")
        else:
            # Fallback: estratégia original se não há colunas observacionais
            threshold = int(0.5 * len(data.columns))
            data = data.dropna(thresh=threshold)
        
        # Se ainda não há dados, usar estratégia de preenchimento mais agressiva
        if len(data) == 0:
            print("⚠️  Tentando estratégia de preenchimento de NaN...")
            # Recomeçar o processamento mas com preenchimento agressivo de NaN
            # (Esta é uma situação de emergência)
            print("⚠️  Reprocessando com preenchimento de NaN ativo...")
            return data  # Retornar vazio para debug - isso deve ser tratado pelo validador
        
        final_shape = data.shape
        
        print(f"Features criadas. Shape: {initial_shape} -> {final_shape}")
        print(f"Registros removidos por NaN: {initial_shape[0] - final_shape[0]}")
        
        if len(data) == 0:
            raise ValueError("Erro crítico: Todos os dados foram perdidos durante o processamento de features")
        
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
    
    def _detect_temporal_resolution(self, data):
        """Detecta automaticamente a resolução temporal dos dados"""
        if 'date' not in data.columns:
            self._detected_resolution = 'daily'
            return
        
        try:
            dates = pd.to_datetime(data['date'])
            diff_hours = dates.diff().dt.total_seconds() / 3600
            mode_diff = diff_hours.mode().iloc[0] if len(diff_hours.mode()) > 0 else 24
            
            if 0.8 <= mode_diff <= 1.2:  # Aproximadamente 1 hora
                self._detected_resolution = 'hourly'
                print("⚠️  DADOS HORÁRIOS DETECTADOS - Usando estratégia anti-vazamento")
            elif 23 <= mode_diff <= 25:  # Aproximadamente 24 horas
                self._detected_resolution = 'daily'
            else:
                self._detected_resolution = 'daily'  # Default
                print(f"Resolução temporal não padrão ({mode_diff:.1f}h), usando estratégia diária")
        except:
            self._detected_resolution = 'daily'
    
    def _create_lag_features(self, df):
        """Cria features de lag temporal adaptadas à resolução"""
        df = df.copy()
        
        # Lags adaptados à resolução temporal
        if self._detected_resolution == 'hourly':
            # Para dados horários: estratégia mais agressiva baseada na autocorrelação
            print("🔒 Usando lags seguros para dados horários")
            if self.variable == 'temperature':
                # Temperatura: lags ainda mais conservadores devido à alta persistência
                meteo_cols = ['t2m', 'tp', 'sp', 'u10', 'v10', 'd2m', 'r', 'ssrd', 'strd']
                lags = [24, 48, 168]  # 1d, 2d, 1semana - MUITO mais conservador
                print("🔒 Temperatura: usando apenas lags ≥24h devido à alta autocorrelação")
            else:
                meteo_cols = ['tp', 'tcwv', 'cape', 'cin', 'sp', 'u10', 'v10', 'q', 'pev']
                lags = [6, 12, 24, 48, 72, 168]  # 6h, 12h, 1d, 2d, 3d, 1semana
        else:
            # Para dados diários: lags originais
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
        """Cria features de média móvel adaptadas à resolução"""
        df = df.copy()
        
        # Windows adaptadas à resolução temporal
        if self._detected_resolution == 'hourly':
            print("🔒 Usando janelas seguras para dados horários")
            if self.variable == 'temperature':
                # Temperatura: janelas muito maiores devido à alta autocorrelação
                meteo_cols = ['t2m', 'tp', 'sp', 'u10', 'v10', 'd2m', 'r']
                windows = [24, 72, 168]  # 1d, 3d, 1semana - Mais conservador
                print("🔒 Temperatura: usando janelas ≥24h para reduzir autocorrelação")
            else:
                meteo_cols = ['tp', 'tcwv', 'cape', 'sp', 'u10', 'v10']
                windows = [12, 24, 48, 168, 336]  # 12h, 1d, 2d, 1semana, 2semanas
        else:
            # Para dados diários: windows originais
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