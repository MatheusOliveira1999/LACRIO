"""
temporal_analysis.py
Análises temporais específicas dos dados climáticos
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class TemporalAnalysis:
    """Análises temporais específicas"""
    
    def __init__(self):
        self.temporal_results = {}
        
    def daily_cycle_analysis(self, era5_df: pd.DataFrame, 
                           station_df: pd.DataFrame,
                           variables: List[str] = None) -> Dict:
        """
        Análise do ciclo diário (para dados horários)
        
        Parameters:
        -----------
        era5_df : pd.DataFrame
            Dados ERA5 com coluna 'date'
        station_df : pd.DataFrame
            Dados da estação com coluna 'date'
        variables : list, optional
            Variáveis para analisar
            
        Returns:
        --------
        dict
            Resultados da análise do ciclo diário
        """
        print("🌅 Analisando ciclo diário...")
        
        # Verificar se dados são horários
        era5_df = era5_df.copy()
        station_df = station_df.copy()
        
        era5_df['date'] = pd.to_datetime(era5_df['date'])
        station_df['date'] = pd.to_datetime(station_df['date'])
        
        # Verificar se há informação horária
        if era5_df['date'].dt.hour.nunique() <= 1:
            print("⚠️ Dados não contêm informação horária")
            return {}
            
        if variables is None:
            era5_vars = set(era5_df.columns) - {'date'}
            station_vars = set(station_df.columns) - {'date'}
            variables = list(era5_vars.intersection(station_vars))
            
        results = {}
        
        # Adicionar hora
        era5_df['hour'] = era5_df['date'].dt.hour
        station_df['hour'] = station_df['date'].dt.hour
        
        for var in variables:
            if var in era5_df.columns and var in station_df.columns:
                var_results = {
                    'variable': var,
                    'hourly_stats': {},
                    'cycle_characteristics': {},
                    'era5_vs_station': {}
                }
                
                # Estatísticas por hora
                hourly_era5 = era5_df.groupby('hour')[var].agg(['mean', 'std', 'count'])
                hourly_station = station_df.groupby('hour')[var].agg(['mean', 'std', 'count'])
                
                var_results['hourly_stats'] = {
                    'era5': hourly_era5.to_dict('index'),
                    'station': hourly_station.to_dict('index')
                }
                
                # Características do ciclo
                era5_hourly_mean = hourly_era5['mean']
                station_hourly_mean = hourly_station['mean']
                
                # Horário de máximo e mínimo
                era5_max_hour = era5_hourly_mean.idxmax()
                era5_min_hour = era5_hourly_mean.idxmin()
                station_max_hour = station_hourly_mean.idxmax()
                station_min_hour = station_hourly_mean.idxmin()
                
                # Amplitude diária
                era5_amplitude = era5_hourly_mean.max() - era5_hourly_mean.min()
                station_amplitude = station_hourly_mean.max() - station_hourly_mean.min()
                
                var_results['cycle_characteristics'] = {
                    'era5': {
                        'max_hour': int(era5_max_hour),
                        'min_hour': int(era5_min_hour),
                        'amplitude': float(era5_amplitude),
                        'mean_daily_range': float(era5_amplitude)
                    },
                    'station': {
                        'max_hour': int(station_max_hour),
                        'min_hour': int(station_min_hour),
                        'amplitude': float(station_amplitude),
                        'mean_daily_range': float(station_amplitude)
                    }
                }
                
                # Comparação ERA5 vs Estação
                hour_correlation = {}
                hour_bias = {}
                
                for hour in range(24):
                    era5_hour_data = era5_df[era5_df['hour'] == hour][var].dropna()
                    station_hour_data = station_df[station_df['hour'] == hour][var].dropna()
                    
                    # Sincronizar por data
                    era5_hour_df = era5_df[era5_df['hour'] == hour][['date', var]].dropna()
                    station_hour_df = station_df[station_df['hour'] == hour][['date', var]].dropna()
                    
                    merged = pd.merge(era5_hour_df, station_hour_df, on='date', suffixes=('_era5', '_station'))
                    
                    if len(merged) > 5:
                        corr, p_val = stats.pearsonr(merged[f'{var}_era5'], merged[f'{var}_station'])
                        bias = (merged[f'{var}_era5'] - merged[f'{var}_station']).mean()
                        
                        hour_correlation[hour] = {'correlation': float(corr), 'p_value': float(p_val)}
                        hour_bias[hour] = float(bias)
                        
                var_results['era5_vs_station'] = {
                    'hourly_correlations': hour_correlation,
                    'hourly_bias': hour_bias,
                    'max_hour_difference': int(abs(era5_max_hour - station_max_hour)),
                    'min_hour_difference': int(abs(era5_min_hour - station_min_hour)),
                    'amplitude_ratio': float(era5_amplitude / station_amplitude) if station_amplitude != 0 else np.nan
                }
                
                results[var] = var_results
                
        return results
        
    def seasonal_cycle_analysis(self, era5_df: pd.DataFrame, 
                              station_df: pd.DataFrame,
                              variables: List[str] = None) -> Dict:
        """
        Análise do ciclo sazonal
        
        Parameters:
        -----------
        era5_df : pd.DataFrame
            Dados ERA5 com coluna 'date'
        station_df : pd.DataFrame
            Dados da estação com coluna 'date'
        variables : list, optional
            Variáveis para analisar
            
        Returns:
        --------
        dict
            Resultados da análise sazonal
        """
        print("🌿 Analisando ciclo sazonal...")
        
        era5_df = era5_df.copy()
        station_df = station_df.copy()
        
        era5_df['date'] = pd.to_datetime(era5_df['date'])
        station_df['date'] = pd.to_datetime(station_df['date'])
        
        era5_df['month'] = era5_df['date'].dt.month
        station_df['month'] = station_df['date'].dt.month
        
        # Definir estações (Hemisfério Sul - Peru)
        def get_season(month):
            if month in [12, 1, 2]:
                return 'DJF'  # Verão
            elif month in [3, 4, 5]:
                return 'MAM'  # Outono
            elif month in [6, 7, 8]:
                return 'JJA'  # Inverno
            else:
                return 'SON'  # Primavera
                
        era5_df['season'] = era5_df['month'].apply(get_season)
        station_df['season'] = station_df['date'].dt.month.apply(get_season)
        
        if variables is None:
            era5_vars = set(era5_df.columns) - {'date', 'month', 'season'}
            station_vars = set(station_df.columns) - {'date', 'month', 'season'}
            variables = list(era5_vars.intersection(station_vars))
            
        results = {}
        
        for var in variables:
            if var in era5_df.columns and var in station_df.columns:
                var_results = {
                    'variable': var,
                    'monthly_climatology': {},
                    'seasonal_climatology': {},
                    'seasonal_comparison': {},
                    'annual_cycle_metrics': {}
                }
                
                # Climatologia mensal
                monthly_era5 = era5_df.groupby('month')[var].agg(['mean', 'std', 'count'])
                monthly_station = station_df.groupby('month')[var].agg(['mean', 'std', 'count'])
                
                var_results['monthly_climatology'] = {
                    'era5': monthly_era5.to_dict('index'),
                    'station': monthly_station.to_dict('index')
                }
                
                # Climatologia sazonal
                seasonal_era5 = era5_df.groupby('season')[var].agg(['mean', 'std', 'count'])
                seasonal_station = station_df.groupby('season')[var].agg(['mean', 'std', 'count'])
                
                var_results['seasonal_climatology'] = {
                    'era5': seasonal_era5.to_dict('index'),
                    'station': seasonal_station.to_dict('index')
                }
                
                # Comparação sazonal detalhada
                seasonal_comparison = {}
                for season in ['DJF', 'MAM', 'JJA', 'SON']:
                    era5_season = era5_df[era5_df['season'] == season]
                    station_season = station_df[station_df['season'] == season]
                    
                    # Merge por data para comparação direta
                    merged = pd.merge(
                        era5_season[['date', var]], 
                        station_season[['date', var]], 
                        on='date', 
                        suffixes=('_era5', '_station')
                    )
                    
                    if len(merged) > 10:
                        era5_vals = merged[f'{var}_era5']
                        station_vals = merged[f'{var}_station']
                        
                        corr, p_val = stats.pearsonr(era5_vals, station_vals)
                        bias = (era5_vals - station_vals).mean()
                        rmse = np.sqrt(((era5_vals - station_vals)**2).mean())
                        
                        seasonal_comparison[season] = {
                            'correlation': float(corr),
                            'p_value': float(p_val),
                            'bias': float(bias),
                            'rmse': float(rmse),
                            'era5_mean': float(era5_vals.mean()),
                            'station_mean': float(station_vals.mean()),
                            'n_points': len(merged)
                        }
                        
                var_results['seasonal_comparison'] = seasonal_comparison
                
                # Métricas do ciclo anual
                era5_monthly_means = monthly_era5['mean']
                station_monthly_means = monthly_station['mean']
                
                # Amplitude anual
                era5_annual_amplitude = era5_monthly_means.max() - era5_monthly_means.min()
                station_annual_amplitude = station_monthly_means.max() - station_monthly_means.min()
                
                # Mês de máximo e mínimo
                era5_max_month = era5_monthly_means.idxmax()
                era5_min_month = era5_monthly_means.idxmin()
                station_max_month = station_monthly_means.idxmax()
                station_min_month = station_monthly_means.idxmin()
                
                # Correlação do ciclo anual
                if len(era5_monthly_means) == len(station_monthly_means):
                    annual_cycle_corr, annual_cycle_p = stats.pearsonr(era5_monthly_means, station_monthly_means)
                else:
                    annual_cycle_corr = annual_cycle_p = np.nan
                
                var_results['annual_cycle_metrics'] = {
                    'era5_amplitude': float(era5_annual_amplitude),
                    'station_amplitude': float(station_annual_amplitude),
                    'amplitude_ratio': float(era5_annual_amplitude / station_annual_amplitude) if station_annual_amplitude != 0 else np.nan,
                    'era5_max_month': int(era5_max_month),
                    'era5_min_month': int(era5_min_month),
                    'station_max_month': int(station_max_month),
                    'station_min_month': int(station_min_month),
                    'max_month_difference': int(abs(era5_max_month - station_max_month)),
                    'min_month_difference': int(abs(era5_min_month - station_min_month)),
                    'annual_cycle_correlation': float(annual_cycle_corr),
                    'annual_cycle_p_value': float(annual_cycle_p)
                }
                
                results[var] = var_results
                
        return results
        
    def interannual_variability(self, era5_df: pd.DataFrame, 
                              station_df: pd.DataFrame,
                              variables: List[str] = None) -> Dict:
        """
        Análise de variabilidade interanual
        
        Parameters:
        -----------
        era5_df : pd.DataFrame
            Dados ERA5 com coluna 'date'
        station_df : pd.DataFrame
            Dados da estação com coluna 'date'
        variables : list, optional
            Variáveis para analisar
            
        Returns:
        --------
        dict
            Resultados da análise interanual
        """
        print("📅 Analisando variabilidade interanual...")
        
        era5_df = era5_df.copy()
        station_df = station_df.copy()
        
        era5_df['date'] = pd.to_datetime(era5_df['date'])
        station_df['date'] = pd.to_datetime(station_df['date'])
        
        era5_df['year'] = era5_df['date'].dt.year
        station_df['year'] = station_df['date'].dt.year
        
        if variables is None:
            era5_vars = set(era5_df.columns) - {'date', 'year'}
            station_vars = set(station_df.columns) - {'date', 'year'}
            variables = list(era5_vars.intersection(station_vars))
            
        results = {}
        
        for var in variables:
            if var in era5_df.columns and var in station_df.columns:
                var_results = {
                    'variable': var,
                    'annual_statistics': {},
                    'variability_metrics': {},
                    'trend_analysis': {},
                    'extreme_years': {}
                }
                
                # Estatísticas anuais
                annual_era5 = era5_df.groupby('year')[var].agg(['mean', 'std', 'min', 'max', 'count'])
                annual_station = station_df.groupby('year')[var].agg(['mean', 'std', 'min', 'max', 'count'])
                
                # Filtrar anos com dados suficientes (pelo menos 300 dias)
                annual_era5 = annual_era5[annual_era5['count'] >= 300]
                annual_station = annual_station[annual_station['count'] >= 300]
                
                var_results['annual_statistics'] = {
                    'era5': annual_era5.to_dict('index'),
                    'station': annual_station.to_dict('index')
                }
                
                # Anos comuns
                common_years = set(annual_era5.index).intersection(set(annual_station.index))
                
                if len(common_years) >= 3:
                    common_years = sorted(list(common_years))
                    
                    era5_annual = annual_era5.loc[common_years]
                    station_annual = annual_station.loc[common_years]
                    
                    # Métricas de variabilidade
                    era5_cv = era5_annual['mean'].std() / era5_annual['mean'].mean()
                    station_cv = era5_annual['mean'].std() / station_annual['mean'].mean()
                    
                    # Correlação interanual
                    annual_corr, annual_p = stats.pearsonr(era5_annual['mean'], station_annual['mean'])
                    
                    var_results['variability_metrics'] = {
                        'years_analyzed': common_years,
                        'n_years': len(common_years),
                        'era5_cv': float(era5_cv),
                        'station_cv': float(station_cv),
                        'annual_correlation': float(annual_corr),
                        'annual_correlation_p': float(annual_p),
                        'era5_mean_annual_std': float(era5_annual['mean'].std()),
                        'station_mean_annual_std': float(station_annual['mean'].std())
                    }
                    
                    # Análise de tendência interanual
                    years_array = np.array(common_years)
                    
                    # Tendência ERA5
                    era5_slope, era5_intercept, era5_r, era5_p, era5_se = stats.linregress(years_array, era5_annual['mean'])
                    
                    # Tendência Estação
                    station_slope, station_intercept, station_r, station_p, station_se = stats.linregress(years_array, station_annual['mean'])
                    
                    var_results['trend_analysis'] = {
                        'era5_trend': {
                            'slope': float(era5_slope),
                            'p_value': float(era5_p),
                            'r_value': float(era5_r),
                            'significant': era5_p < 0.05
                        },
                        'station_trend': {
                            'slope': float(station_slope),
                            'p_value': float(station_p),
                            'r_value': float(station_r),
                            'significant': station_p < 0.05
                        },
                        'trend_agreement': (era5_slope * station_slope) > 0
                    }
                    
                    # Anos extremos
                    era5_mean_annual = era5_annual['mean']
                    station_mean_annual = station_annual['mean']
                    
                    # Anos mais quentes/frios ou mais secos/chuvosos
                    era5_max_year = era5_mean_annual.idxmax()
                    era5_min_year = era5_mean_annual.idxmin()
                    station_max_year = station_mean_annual.idxmax()
                    station_min_year = station_mean_annual.idxmin()
                    
                    var_results['extreme_years'] = {
                        'era5_maximum': {
                            'year': int(era5_max_year),
                            'value': float(era5_mean_annual[era5_max_year])
                        },
                        'era5_minimum': {
                            'year': int(era5_min_year),
                            'value': float(era5_mean_annual[era5_min_year])
                        },
                        'station_maximum': {
                            'year': int(station_max_year),
                            'value': float(station_mean_annual[station_max_year])
                        },
                        'station_minimum': {
                            'year': int(station_min_year),
                            'value': float(station_mean_annual[station_min_year])
                        },
                        'extreme_years_match': {
                            'max_year_same': era5_max_year == station_max_year,
                            'min_year_same': era5_min_year == station_min_year
                        }
                    }
                    
                results[var] = var_results
                
        return results
        
    def missing_data_analysis(self, era5_df: pd.DataFrame, 
                            station_df: pd.DataFrame) -> Dict:
        """
        Análise de dados faltantes
        
        Parameters:
        -----------
        era5_df : pd.DataFrame
            Dados ERA5
        station_df : pd.DataFrame
            Dados da estação
            
        Returns:
        --------
        dict
            Análise de dados faltantes
        """
        print("🔍 Analisando dados faltantes...")
        
        results = {
            'era5_missing': {},
            'station_missing': {},
            'temporal_patterns': {},
            'data_availability': {}
        }
        
        # Análise geral de dados faltantes
        for df, name in [(era5_df, 'era5'), (station_df, 'station')]:
            missing_info = {}
            
            for col in df.columns:
                if col != 'date':
                    total = len(df)
                    missing = df[col].isnull().sum()
                    missing_pct = (missing / total) * 100
                    
                    missing_info[col] = {
                        'total_records': total,
                        'missing_count': int(missing),
                        'missing_percent': float(missing_pct),
                        'available_count': int(total - missing),
                        'availability_percent': float(100 - missing_pct)
                    }
                    
            results[f'{name}_missing'] = missing_info
            
        # Padrões temporais de dados faltantes
        if 'date' in era5_df.columns:
            era5_temporal = era5_df.copy()
            station_temporal = station_df.copy()
            
            era5_temporal['date'] = pd.to_datetime(era5_temporal['date'])
            station_temporal['date'] = pd.to_datetime(station_temporal['date'])
            
            # Análise mensal de disponibilidade
            era5_temporal['month'] = era5_temporal['date'].dt.month
            station_temporal['month'] = station_temporal['date'].dt.month
            
            # Análise anual de disponibilidade
            era5_temporal['year'] = era5_temporal['date'].dt.year
            station_temporal['year'] = station_temporal['date'].dt.year
            
            temporal_patterns = {}
            
            # Verificar se existem variáveis numéricas além de date
            era5_numeric = era5_temporal.select_dtypes(include=[np.number]).columns
            station_numeric = station_temporal.select_dtypes(include=[np.number]).columns
            
            if len(era5_numeric) > 0:
                # Disponibilidade por mês (ERA5)
                monthly_availability_era5 = {}
                for month in range(1, 13):
                    month_data = era5_temporal[era5_temporal['month'] == month]
                    if len(month_data) > 0:
                        availability = {}
                        for col in era5_numeric:
                            if col in ['month', 'year']:
                                continue
                            avail_pct = ((len(month_data) - month_data[col].isnull().sum()) / len(month_data)) * 100
                            availability[col] = float(avail_pct)
                        monthly_availability_era5[month] = availability
                        
                temporal_patterns['era5_monthly_availability'] = monthly_availability_era5
                
            if len(station_numeric) > 0:
                # Disponibilidade por mês (Estação)
                monthly_availability_station = {}
                for month in range(1, 13):
                    month_data = station_temporal[station_temporal['month'] == month]
                    if len(month_data) > 0:
                        availability = {}
                        for col in station_numeric:
                            if col in ['month', 'year']:
                                continue
                            avail_pct = ((len(month_data) - month_data[col].isnull().sum()) / len(month_data)) * 100
                            availability[col] = float(avail_pct)
                        monthly_availability_station[month] = availability
                        
                temporal_patterns['station_monthly_availability'] = monthly_availability_station
                
            results['temporal_patterns'] = temporal_patterns
            
        # Análise de disponibilidade de dados
        era5_start = era5_df['date'].min() if 'date' in era5_df.columns else None
        era5_end = era5_df['date'].max() if 'date' in era5_df.columns else None
        station_start = station_df['date'].min() if 'date' in station_df.columns else None
        station_end = station_df['date'].max() if 'date' in station_df.columns else None
        
        if era5_start and station_start:
            common_start = max(era5_start, station_start)
            common_end = min(era5_end, station_end)
            
            total_days = (common_end - common_start).days + 1
            
            # Contar dias com dados em ambos os datasets
            era5_common = era5_df[(era5_df['date'] >= common_start) & (era5_df['date'] <= common_end)]
            station_common = station_df[(station_df['date'] >= common_start) & (station_df['date'] <= common_end)]
            
            # Merge para encontrar dias comuns
            merged = pd.merge(era5_common[['date']], station_common[['date']], on='date')
            common_days = len(merged)
            
            results['data_availability'] = {
                'common_period': {
                    'start_date': str(common_start),
                    'end_date': str(common_end),
                    'total_days': total_days,
                    'common_days': common_days,
                    'availability_percent': float((common_days / total_days) * 100) if total_days > 0 else 0
                },
                'era5_period': {
                    'start_date': str(era5_start),
                    'end_date': str(era5_end),
                    'total_days': (era5_end - era5_start).days + 1
                },
                'station_period': {
                    'start_date': str(station_start),
                    'end_date': str(station_end),
                    'total_days': (station_end - station_start).days + 1
                }
            }
            
        return results