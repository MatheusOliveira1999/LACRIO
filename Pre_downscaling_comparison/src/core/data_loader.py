"""
data_loader.py
Carregamento e sincronização de dados ERA5 e estações meteorológicas
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Carrega e sincroniza dados ERA5 e estações meteorológicas"""
    
    def __init__(self):
        self.era5_data = None
        self.station_data = None
        
    def load_era5_data(self, filepath: str) -> pd.DataFrame:
        """
        Carrega dados ERA5
        
        Parameters:
        -----------
        filepath : str
            Caminho para arquivo CSV do ERA5
            
        Returns:
        --------
        pd.DataFrame
            Dados ERA5 carregados
        """
        print(f"📁 Carregando dados ERA5: {filepath}")
        
        try:
            # Carregar CSV
            df = pd.read_csv(filepath)
            
            # Detectar coluna de data
            date_columns = ['date', 'time', 'Date', 'Time', 'datetime', 'timestamp']
            date_col = None
            
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
                    
            if date_col is None:
                raise ValueError("Coluna de data não encontrada no arquivo ERA5")
                
            # Converter para datetime
            df[date_col] = pd.to_datetime(df[date_col])
            if date_col != 'date':
                df['date'] = df[date_col]
                df.drop(columns=[date_col], inplace=True)
                
            # Ordenar por data
            df = df.sort_values('date').reset_index(drop=True)
            
            # Remover duplicatas de data
            df = df.drop_duplicates(subset=['date']).reset_index(drop=True)
            
            # Padronizar nomes das colunas ERA5
            df = self._standardize_era5_columns(df)
            
            # Tratar valores faltantes/inválidos ERA5
            print("🔍 Tratando valores faltantes do ERA5...")
            missing_values = [-9999, -999, 999, 9999, -99, 99, 'NULL', 'null', 'NA', 'na', '', ' ']
            df = df.replace(missing_values, pd.NA)
            
            # Debug: mostrar quantos NaN temos por variável no ERA5
            for col in df.columns:
                if col != 'date' and df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    nan_count = df[col].isna().sum()
                    total = len(df)
                    if nan_count > 0:
                        print(f"    ERA5 {col}: {nan_count}/{total} ({nan_count/total*100:.1f}%) valores faltantes")
            
            self.era5_data = df
            print(f"✅ ERA5 carregado: {df.shape[0]} registros, {df.shape[1]} variáveis")
            print(f"📅 Período: {df['date'].min()} a {df['date'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Erro ao carregar ERA5: {str(e)}")
            raise
            
    def load_station_data(self, filepath: str) -> pd.DataFrame:
        """
        Carrega dados da estação meteorológica
        
        Parameters:
        -----------
        filepath : str
            Caminho para arquivo CSV da estação
            
        Returns:
        --------
        pd.DataFrame
            Dados da estação carregados
        """
        print(f"📁 Carregando dados da estação: {filepath}")
        
        try:
            # Carregar CSV
            df = pd.read_csv(filepath)
            
            # Detectar coluna de data
            date_columns = ['date', 'time', 'Date', 'Time', 'datetime', 'timestamp', 'data']
            date_col = None
            
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
                    
            if date_col is None:
                raise ValueError("Coluna de data não encontrada no arquivo da estação")
                
            # Converter para datetime
            df[date_col] = pd.to_datetime(df[date_col])
            if date_col != 'date':
                df['date'] = df[date_col]
                df.drop(columns=[date_col], inplace=True)
                
            # Padronizar nomes das colunas
            df = self._standardize_station_columns(df)
            
            # Tratar valores faltantes/inválidos ANTES de qualquer processamento
            print("🔍 Tratando valores faltantes da estação...")
            
            # Substituir valores comuns de dados faltantes por NaN
            missing_values = [-9999, -999, 999, 9999, -99, 99, 'NULL', 'null', 'NA', 'na', '', ' ']
            df = df.replace(missing_values, pd.NA)
            
            # Debug: mostrar quantos NaN temos por variável
            for col in df.columns:
                if col != 'date' and df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    nan_count = df[col].isna().sum()
                    total = len(df)
                    if nan_count > 0:
                        print(f"    {col}: {nan_count}/{total} ({nan_count/total*100:.1f}%) valores faltantes")
            
            # Ordenar por data
            df = df.sort_values('date').reset_index(drop=True)
            
            # Remover duplicatas de data
            df = df.drop_duplicates(subset=['date']).reset_index(drop=True)
            
            self.station_data = df
            print(f"✅ Estação carregada: {df.shape[0]} registros, {df.shape[1]} variáveis")
            print(f"📅 Período: {df['date'].min()} a {df['date'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados da estação: {str(e)}")
            raise
            
    def _standardize_station_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Padroniza nomes das colunas da estação
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame da estação
            
        Returns:
        --------
        pd.DataFrame
            DataFrame com colunas padronizadas
        """
        # Mapeamento de colunas comuns
        column_mapping = {
            # Temperatura
            'Temperatura': 'temperature',
            'temperatura': 'temperature',
            'temp': 'temperature',
            'T': 'temperature',
            'TEMP': 'temperature',
            't2m': 'temperature',
            'temp_air': 'temperature',
            
            # Precipitação
            'Precipitação': 'precipitation',
            'precipitacao': 'precipitation',
            'precip': 'precipitation',
            'chuva': 'precipitation',
            'P': 'precipitation',
            'PRECIP': 'precipitation',
            'tp': 'precipitation',
            'rain': 'precipitation',
            
            # Umidade
            'umidade': 'humidity',
            'humid': 'humidity',
            'RH': 'humidity',
            'rh': 'humidity',
            'HUMID': 'humidity',
            
            # Pressão
            'pressao': 'pressure',
            'press': 'pressure',
            'sp': 'pressure',
            'PRESS': 'pressure',
            'msl': 'pressure',
            
            # Vento
            'vento': 'wind_speed',
            'wind': 'wind_speed',
            'u10': 'wind_u',
            'v10': 'wind_v',
        }
        
        # Aplicar mapeamento
        df = df.rename(columns=column_mapping)
        
        # Converter colunas para lowercase (exceto 'date')
        new_columns = {}
        for col in df.columns:
            if col != 'date':
                new_columns[col] = col.lower()
        df = df.rename(columns=new_columns)
        
        return df
    
    def _standardize_era5_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Padroniza nomes das colunas do ERA5
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame do ERA5
            
        Returns:
        --------
        pd.DataFrame
            DataFrame com colunas padronizadas
        """
        # Mapeamento de colunas ERA5
        era5_mapping = {
            't2m': 'temperature',  # Temperatura a 2m
            'tp': 'precipitation',  # Precipitação total
            'd2m': 'dewpoint',     # Ponto de orvalho a 2m
            'sp': 'pressure',      # Pressão superficial
            'u10': 'wind_u',       # Componente U do vento a 10m
            'v10': 'wind_v',       # Componente V do vento a 10m
            'ssrd': 'solar_radiation',  # Radiação solar
            'strd': 'thermal_radiation', # Radiação térmica
            'sf': 'snowfall',      # Neve
        }
        
        # Aplicar mapeamento
        df = df.rename(columns=era5_mapping)
        
        return df
        
    def synchronize_datasets(self, era5_df: pd.DataFrame = None, 
                           station_df: pd.DataFrame = None,
                           time_offset_hours: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Sincroniza datasets por período comum
        
        Parameters:
        -----------
        era5_df : pd.DataFrame, optional
            Dados ERA5 (usa self.era5_data se None)
        station_df : pd.DataFrame, optional
            Dados da estação (usa self.station_data se None)
        time_offset_hours : int, optional
            Offset temporal em horas para aplicar ao ERA5 (padrão: 0)
            Valores positivos fazem o ERA5 ficar "atrasado" em relação à estação
            
        Returns:
        --------
        tuple
            (era5_sync, station_sync) - DataFrames sincronizados
        """
        print("🔄 Sincronizando datasets...")
        
        if era5_df is None:
            era5_df = self.era5_data
        if station_df is None:
            station_df = self.station_data
            
        if era5_df is None or station_df is None:
            raise ValueError("Dados não carregados. Execute load_era5_data() e load_station_data() primeiro.")
        
        # Aplicar offset temporal ao ERA5 se especificado
        if time_offset_hours != 0:
            print(f"⏰ Aplicando offset temporal de {time_offset_hours} horas ao ERA5...")
            era5_df = era5_df.copy()
            era5_df['date'] = era5_df['date'] + pd.Timedelta(hours=time_offset_hours)
            
        # Encontrar período comum
        era5_start = era5_df['date'].min()
        era5_end = era5_df['date'].max()
        station_start = station_df['date'].min()
        station_end = station_df['date'].max()
        
        common_start = max(era5_start, station_start)
        common_end = min(era5_end, station_end)
        
        print(f"📅 Período comum: {common_start} a {common_end}")
        
        # Filtrar por período comum
        era5_sync = era5_df[
            (era5_df['date'] >= common_start) & 
            (era5_df['date'] <= common_end)
        ].copy().reset_index(drop=True)
        
        station_sync = station_df[
            (station_df['date'] >= common_start) & 
            (station_df['date'] <= common_end)
        ].copy().reset_index(drop=True)
        
        # Merge por data para garantir sincronização perfeita
        merged = pd.merge(era5_sync, station_sync, on='date', how='inner', suffixes=('_era5', '_station'))
        
        # Separar novamente mantendo a ordem do merge
        era5_cols = ['date'] + [col for col in merged.columns if col.endswith('_era5')]
        station_cols = ['date'] + [col for col in merged.columns if col.endswith('_station')]
        
        era5_final = merged[era5_cols].copy().reset_index(drop=True)
        station_final = merged[station_cols].copy().reset_index(drop=True)
        
        # Remover sufixos
        era5_final.columns = [col.replace('_era5', '') for col in era5_final.columns]
        station_final.columns = [col.replace('_station', '') for col in station_final.columns]
        
        print(f"✅ Sincronização concluída: {len(merged)} registros comuns")
        
        return era5_final, station_final
        
    def validate_data_quality(self, df: pd.DataFrame, data_type: str = 'unknown') -> Dict:
        """
        Valida qualidade dos dados
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame para validar
        data_type : str
            Tipo de dados ('era5', 'station', 'unknown')
            
        Returns:
        --------
        dict
            Relatório de qualidade dos dados
        """
        print(f"🔍 Validando qualidade dos dados ({data_type})...")
        
        report = {
            'data_type': data_type,
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max(),
                'duration_days': (df['date'].max() - df['date'].min()).days
            },
            'variables': [],
            'missing_data': {},
            'outliers': {},
            'summary_stats': {}
        }
        
        # Analisar cada variável (exceto date)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'date':
                continue
                
            var_info = {
                'name': col,
                'type': str(df[col].dtype),
                'count': df[col].count(),
                'missing': df[col].isnull().sum(),
                'missing_percent': (df[col].isnull().sum() / len(df)) * 100
            }
            
            # Estatísticas básicas
            if var_info['count'] > 0:
                var_info.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q25': df[col].quantile(0.25),
                    'q50': df[col].quantile(0.50),
                    'q75': df[col].quantile(0.75)
                })
                
                # Detectar outliers (IQR method)
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                var_info['outliers_count'] = len(outliers)
                var_info['outliers_percent'] = (len(outliers) / len(df)) * 100
                
            report['variables'].append(var_info)
            report['missing_data'][col] = var_info['missing_percent']
            report['outliers'][col] = var_info.get('outliers_percent', 0)
            report['summary_stats'][col] = {
                k: v for k, v in var_info.items() 
                if k in ['mean', 'std', 'min', 'max', 'q50']
            }
        
        # Resumo geral
        report['overall_quality'] = {
            'total_variables': len(report['variables']),
            'avg_missing_percent': np.mean(list(report['missing_data'].values())),
            'avg_outliers_percent': np.mean(list(report['outliers'].values())),
            'data_completeness': 100 - np.mean(list(report['missing_data'].values()))
        }
        
        print(f"✅ Validação concluída:")
        print(f"   • {report['total_records']} registros")
        print(f"   • {report['overall_quality']['total_variables']} variáveis")
        print(f"   • {report['overall_quality']['data_completeness']:.1f}% completude")
        
        return report
        
    def get_common_variables(self, era5_df: pd.DataFrame = None, 
                           station_df: pd.DataFrame = None) -> Dict:
        """
        Identifica variáveis comuns entre datasets
        
        Parameters:
        -----------
        era5_df : pd.DataFrame, optional
            Dados ERA5
        station_df : pd.DataFrame, optional
            Dados da estação
            
        Returns:
        --------
        dict
            Variáveis comuns e mapeamento
        """
        if era5_df is None:
            era5_df = self.era5_data
        if station_df is None:
            station_df = self.station_data
            
        era5_vars = set(era5_df.columns) - {'date'}
        station_vars = set(station_df.columns) - {'date'}
        
        # Variáveis com nomes idênticos
        exact_matches = era5_vars.intersection(station_vars)
        
        # Mapeamento baseado em palavras-chave
        variable_keywords = {
            'temperature': ['temp', 't2m', 'temperatura', 'temperature'],
            'precipitation': ['precip', 'tp', 'precipitacao', 'precipitation', 'chuva', 'rain'],
            'humidity': ['humid', 'rh', 'umidade', 'humidity'],
            'pressure': ['press', 'sp', 'msl', 'pressao', 'pressure'],
            'wind': ['wind', 'vento', 'u10', 'v10']
        }
        
        mapped_vars = {}
        for standard_name, keywords in variable_keywords.items():
            era5_matches = [var for var in era5_vars if any(kw in var.lower() for kw in keywords)]
            station_matches = [var for var in station_vars if any(kw in var.lower() for kw in keywords)]
            
            if era5_matches and station_matches:
                mapped_vars[standard_name] = {
                    'era5': era5_matches[0],  # Pega o primeiro match
                    'station': station_matches[0]
                }
        
        return {
            'exact_matches': list(exact_matches),
            'mapped_variables': mapped_vars,
            'era5_only': list(era5_vars - station_vars),
            'station_only': list(station_vars - era5_vars)
        }