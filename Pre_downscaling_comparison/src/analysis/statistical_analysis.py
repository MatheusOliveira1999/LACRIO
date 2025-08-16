"""
statistical_analysis.py
Análises estatísticas avançadas dos dados climáticos
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnalysis:
    """Análises estatísticas dos dados climáticos"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def correlation_analysis(self, era5_data: pd.Series, 
                           station_data: pd.Series,
                           variable_name: str = 'unknown') -> Dict:
        """
        Análise completa de correlação
        
        Parameters:
        -----------
        era5_data : pd.Series
            Dados ERA5
        station_data : pd.Series
            Dados da estação
        variable_name : str
            Nome da variável
            
        Returns:
        --------
        dict
            Resultados da análise de correlação
        """
        print(f"📈 Análise de correlação para {variable_name}...")
        
        results = {
            'variable': variable_name,
            'n_points': len(era5_data),
            'correlations': {},
            'significance_tests': {},
            'confidence_intervals': {},
            'correlation_by_season': {},
            'correlation_by_quantile': {}
        }
        
        # Correlações básicas
        pearson_r, pearson_p = stats.pearsonr(era5_data, station_data)
        spearman_r, spearman_p = stats.spearmanr(era5_data, station_data)
        kendall_tau, kendall_p = stats.kendalltau(era5_data, station_data)
        
        results['correlations'] = {
            'pearson': float(pearson_r),
            'spearman': float(spearman_r),
            'kendall': float(kendall_tau),
            'r_squared': float(pearson_r**2)
        }
        
        results['significance_tests'] = {
            'pearson_p_value': float(pearson_p),
            'spearman_p_value': float(spearman_p),
            'kendall_p_value': float(kendall_p),
            'pearson_significant': pearson_p < 0.05,
            'spearman_significant': spearman_p < 0.05,
            'kendall_significant': kendall_p < 0.05
        }
        
        # Intervalos de confiança para correlação de Pearson
        r = pearson_r
        n = len(era5_data)
        if n > 3:
            # Transformação de Fisher
            z = 0.5 * np.log((1 + r) / (1 - r))
            se = 1 / np.sqrt(n - 3)
            ci_z = [z - 1.96 * se, z + 1.96 * se]
            
            # Transformação inversa
            ci_r = [(np.exp(2 * ci_z[0]) - 1) / (np.exp(2 * ci_z[0]) + 1),
                    (np.exp(2 * ci_z[1]) - 1) / (np.exp(2 * ci_z[1]) + 1)]
            
            results['confidence_intervals'] = {
                'pearson_95_ci': [float(ci_r[0]), float(ci_r[1])],
                'margin_of_error': float(abs(ci_r[1] - r))
            }
            
        # Correlação por quantis
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        quantile_correlations = {}
        
        for q in quantiles:
            threshold = station_data.quantile(q)
            mask = station_data >= threshold
            if mask.sum() > 10:
                q_corr, q_p = stats.pearsonr(era5_data[mask], station_data[mask])
                quantile_correlations[f'q{int(q*100)}'] = {
                    'correlation': float(q_corr),
                    'p_value': float(q_p),
                    'n_points': int(mask.sum())
                }
                
        results['correlation_by_quantile'] = quantile_correlations
        
        return results
        
    def bias_analysis(self, era5_data: pd.Series, 
                     station_data: pd.Series,
                     variable_name: str = 'unknown') -> Dict:
        """
        Análise detalhada de bias
        
        Parameters:
        -----------
        era5_data : pd.Series
            Dados ERA5
        station_data : pd.Series
            Dados da estação
        variable_name : str
            Nome da variável
            
        Returns:
        --------
        dict
            Resultados da análise de bias
        """
        print(f"⚖️ Análise de bias para {variable_name}...")
        
        bias_series = era5_data - station_data
        
        results = {
            'variable': variable_name,
            'overall_bias': {},
            'relative_bias': {},
            'bias_distribution': {},
            'bias_by_magnitude': {},
            'bias_trends': {},
            'multiplicative_bias': {}
        }
        
        # Bias geral
        results['overall_bias'] = {
            'mean_bias': float(bias_series.mean()),
            'median_bias': float(bias_series.median()),
            'std_bias': float(bias_series.std()),
            'min_bias': float(bias_series.min()),
            'max_bias': float(bias_series.max()),
            'bias_range': float(bias_series.max() - bias_series.min())
        }
        
        # Bias relativo
        relative_bias = (era5_data - station_data) / station_data * 100
        relative_bias = relative_bias[np.isfinite(relative_bias)]  # Remove inf/nan
        
        if len(relative_bias) > 0:
            results['relative_bias'] = {
                'mean_relative_bias': float(relative_bias.mean()),
                'median_relative_bias': float(relative_bias.median()),
                'std_relative_bias': float(relative_bias.std())
            }
            
        # Distribuição do bias
        results['bias_distribution'] = {
            'bias_skewness': float(stats.skew(bias_series)),
            'bias_kurtosis': float(stats.kurtosis(bias_series)),
            'is_bias_normal': float(stats.shapiro(bias_series.sample(min(5000, len(bias_series))))[1]) > 0.05,
            'bias_percentiles': {
                'p5': float(bias_series.quantile(0.05)),
                'p25': float(bias_series.quantile(0.25)),
                'p75': float(bias_series.quantile(0.75)),
                'p95': float(bias_series.quantile(0.95))
            }
        }
        
        # Bias por magnitude da observação
        magnitude_bins = station_data.quantile([0, 0.33, 0.67, 1.0])
        bias_by_magnitude = {}
        
        for i in range(len(magnitude_bins) - 1):
            lower = magnitude_bins.iloc[i]
            upper = magnitude_bins.iloc[i + 1]
            mask = (station_data >= lower) & (station_data <= upper)
            
            if mask.sum() > 5:
                bias_subset = bias_series[mask]
                bias_by_magnitude[f'magnitude_bin_{i+1}'] = {
                    'range': [float(lower), float(upper)],
                    'mean_bias': float(bias_subset.mean()),
                    'std_bias': float(bias_subset.std()),
                    'n_points': int(mask.sum())
                }
                
        results['bias_by_magnitude'] = bias_by_magnitude
        
        # Bias multiplicativo
        if (station_data > 0).all():
            multiplicative_bias = era5_data / station_data
            results['multiplicative_bias'] = {
                'mean_factor': float(multiplicative_bias.mean()),
                'median_factor': float(multiplicative_bias.median()),
                'geometric_mean_factor': float(stats.gmean(multiplicative_bias)),
                'std_factor': float(multiplicative_bias.std())
            }
            
        return results
        
    def distribution_analysis(self, era5_data: pd.Series, 
                            station_data: pd.Series,
                            variable_name: str = 'unknown') -> Dict:
        """
        Análise de distribuições
        
        Parameters:
        -----------
        era5_data : pd.Series
            Dados ERA5
        station_data : pd.Series
            Dados da estação
        variable_name : str
            Nome da variável
            
        Returns:
        --------
        dict
            Resultados da análise de distribuições
        """
        print(f"📊 Análise de distribuições para {variable_name}...")
        
        results = {
            'variable': variable_name,
            'era5_distribution': {},
            'station_distribution': {},
            'distribution_comparison': {},
            'moments_comparison': {}
        }
        
        # Análise individual das distribuições
        for data, name in [(era5_data, 'era5'), (station_data, 'station')]:
            dist_stats = {
                'mean': float(data.mean()),
                'std': float(data.std()),
                'variance': float(data.var()),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data)),
                'min': float(data.min()),
                'max': float(data.max()),
                'range': float(data.max() - data.min()),
                'iqr': float(data.quantile(0.75) - data.quantile(0.25)),
                'cv': float(data.std() / data.mean()) if data.mean() != 0 else np.nan
            }
            
            # Testes de normalidade
            sample_size = min(5000, len(data))
            sample_data = data.sample(sample_size) if len(data) > sample_size else data
            
            shapiro_stat, shapiro_p = stats.shapiro(sample_data)
            dist_stats['normality_test'] = {
                'shapiro_stat': float(shapiro_stat),
                'shapiro_p': float(shapiro_p),
                'is_normal': shapiro_p > 0.05
            }
            
            # Teste de Jarque-Bera para normalidade
            jb_stat, jb_p = stats.jarque_bera(data)
            dist_stats['jarque_bera'] = {
                'statistic': float(jb_stat),
                'p_value': float(jb_p),
                'is_normal': jb_p > 0.05
            }
            
            results[f'{name}_distribution'] = dist_stats
            
        # Comparação entre distribuições
        
        # Teste de Kolmogorov-Smirnov
        ks_stat, ks_p = stats.ks_2samp(era5_data, station_data)
        
        # Teste de Mann-Whitney U
        mw_stat, mw_p = stats.mannwhitneyu(era5_data, station_data, alternative='two-sided')
        
        # Teste de Levene para igualdade de variâncias
        levene_stat, levene_p = stats.levene(era5_data, station_data)
        
        # Teste t para igualdade de médias (se normal) ou Wilcoxon (se não normal)
        era5_normal = results['era5_distribution']['normality_test']['is_normal']
        station_normal = results['station_distribution']['normality_test']['is_normal']
        
        if era5_normal and station_normal:
            # Teste t
            t_stat, t_p = stats.ttest_ind(era5_data, station_data)
            mean_test = {
                'test_type': 'ttest',
                'statistic': float(t_stat),
                'p_value': float(t_p),
                'means_equal': t_p > 0.05
            }
        else:
            # Teste de Wilcoxon-Mann-Whitney
            mean_test = {
                'test_type': 'mannwhitney',
                'statistic': float(mw_stat),
                'p_value': float(mw_p),
                'means_equal': mw_p > 0.05
            }
            
        results['distribution_comparison'] = {
            'ks_test': {
                'statistic': float(ks_stat),
                'p_value': float(ks_p),
                'distributions_similar': ks_p > 0.05
            },
            'levene_test': {
                'statistic': float(levene_stat),
                'p_value': float(levene_p),
                'variances_equal': levene_p > 0.05
            },
            'mean_test': mean_test
        }
        
        # Comparação de momentos
        results['moments_comparison'] = {
            'mean_difference': float(era5_data.mean() - station_data.mean()),
            'std_ratio': float(era5_data.std() / station_data.std()) if station_data.std() != 0 else np.nan,
            'variance_ratio': float(era5_data.var() / station_data.var()) if station_data.var() != 0 else np.nan,
            'skewness_difference': float(stats.skew(era5_data) - stats.skew(station_data)),
            'kurtosis_difference': float(stats.kurtosis(era5_data) - stats.kurtosis(station_data))
        }
        
        return results
        
    def trend_analysis(self, era5_data: pd.Series, 
                      station_data: pd.Series,
                      dates: pd.Series = None,
                      variable_name: str = 'unknown') -> Dict:
        """
        Análise de tendências temporais
        
        Parameters:
        -----------
        era5_data : pd.Series
            Dados ERA5
        station_data : pd.Series
            Dados da estação
        dates : pd.Series, optional
            Série temporal das datas
        variable_name : str
            Nome da variável
            
        Returns:
        --------
        dict
            Resultados da análise de tendências
        """
        print(f"📈 Análise de tendências para {variable_name}...")
        
        results = {
            'variable': variable_name,
            'era5_trend': {},
            'station_trend': {},
            'trend_comparison': {}
        }
        
        # Se não há datas, criar índice sequencial
        if dates is None:
            time_index = np.arange(len(era5_data))
        else:
            # Converter datas para valores numéricos (dias desde início)
            dates = pd.to_datetime(dates)
            time_index = (dates - dates.min()).dt.days
            
        # Análise de tendência para cada série
        for data, name in [(era5_data, 'era5'), (station_data, 'station')]:
            # Regressão linear
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, data)
            
            # Mann-Kendall trend test
            mk_trend, mk_p = self._mann_kendall_test(data)
            
            trend_stats = {
                'linear_slope': float(slope),
                'linear_intercept': float(intercept),
                'linear_r_value': float(r_value),
                'linear_p_value': float(p_value),
                'linear_std_err': float(std_err),
                'trend_significant': p_value < 0.05,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                'mann_kendall_trend': mk_trend,
                'mann_kendall_p': float(mk_p),
                'mk_significant': mk_p < 0.05
            }
            
            # Cálculo da mudança total no período
            total_change = slope * (time_index.max() - time_index.min())
            trend_stats['total_change'] = float(total_change)
            
            # Mudança relativa (%)
            if data.mean() != 0:
                relative_change = (total_change / data.mean()) * 100
                trend_stats['relative_change_percent'] = float(relative_change)
                
            results[f'{name}_trend'] = trend_stats
            
        # Comparação de tendências
        slope_diff = results['era5_trend']['linear_slope'] - results['station_trend']['linear_slope']
        
        results['trend_comparison'] = {
            'slope_difference': float(slope_diff),
            'era5_trend_stronger': abs(results['era5_trend']['linear_slope']) > abs(results['station_trend']['linear_slope']),
            'trends_same_direction': (results['era5_trend']['linear_slope'] * results['station_trend']['linear_slope']) > 0,
            'both_significant': results['era5_trend']['trend_significant'] and results['station_trend']['trend_significant']
        }
        
        return results
        
    def _mann_kendall_test(self, data: pd.Series) -> Tuple[str, float]:
        """
        Teste de Mann-Kendall para tendência
        
        Parameters:
        -----------
        data : pd.Series
            Série temporal
            
        Returns:
        --------
        tuple
            (trend_direction, p_value)
        """
        n = len(data)
        
        # Calcular estatística S
        S = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if data.iloc[j] > data.iloc[i]:
                    S += 1
                elif data.iloc[j] < data.iloc[i]:
                    S -= 1
                    
        # Calcular variância
        var_S = (n * (n - 1) * (2 * n + 5)) / 18
        
        # Calcular Z
        if S > 0:
            Z = (S - 1) / np.sqrt(var_S)
        elif S < 0:
            Z = (S + 1) / np.sqrt(var_S)
        else:
            Z = 0
            
        # Calcular p-value (teste bilateral)
        p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
        
        # Determinar tendência
        if p_value < 0.05:
            if S > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
        else:
            trend = 'no trend'
            
        return trend, p_value
        
    def performance_metrics(self, era5_data: pd.Series, 
                          station_data: pd.Series,
                          variable_name: str = 'unknown') -> Dict:
        """
        Métricas de performance detalhadas
        
        Parameters:
        -----------
        era5_data : pd.Series
            Dados ERA5 (predições)
        station_data : pd.Series
            Dados da estação (observações)
        variable_name : str
            Nome da variável
            
        Returns:
        --------
        dict
            Métricas de performance
        """
        print(f"📊 Calculando métricas de performance para {variable_name}...")
        
        results = {
            'variable': variable_name,
            'basic_metrics': {},
            'normalized_metrics': {},
            'skill_scores': {},
            'efficiency_metrics': {}
        }
        
        # Métricas básicas
        bias = float(np.mean(era5_data - station_data))
        mae = float(mean_absolute_error(station_data, era5_data))
        mse = float(mean_squared_error(station_data, era5_data))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(station_data, era5_data))
        
        results['basic_metrics'] = {
            'bias': bias,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r_squared': r2
        }
        
        # Métricas normalizadas
        mean_obs = station_data.mean()
        std_obs = station_data.std()
        
        if mean_obs != 0:
            normalized_bias = bias / mean_obs
            normalized_mae = mae / mean_obs
            normalized_rmse = rmse / mean_obs
        else:
            normalized_bias = normalized_mae = normalized_rmse = np.nan
            
        if std_obs != 0:
            standardized_rmse = rmse / std_obs
        else:
            standardized_rmse = np.nan
            
        results['normalized_metrics'] = {
            'normalized_bias': float(normalized_bias),
            'normalized_mae': float(normalized_mae),
            'normalized_rmse': float(normalized_rmse),
            'standardized_rmse': float(standardized_rmse)
        }
        
        # Skill scores
        
        # Nash-Sutcliffe Efficiency
        nse = 1 - (np.sum((station_data - era5_data)**2) / 
                  np.sum((station_data - mean_obs)**2))
        
        # Kling-Gupta Efficiency
        correlation = np.corrcoef(era5_data, station_data)[0, 1]
        alpha = std_obs / era5_data.std() if era5_data.std() != 0 else np.nan
        beta = mean_obs / era5_data.mean() if era5_data.mean() != 0 else np.nan
        kge = 1 - np.sqrt((correlation - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        
        # Index of Agreement
        d = 1 - (np.sum((era5_data - station_data)**2) / 
                np.sum((np.abs(era5_data - mean_obs) + np.abs(station_data - mean_obs))**2))
        
        results['skill_scores'] = {
            'nash_sutcliffe': float(nse),
            'kling_gupta': float(kge),
            'index_of_agreement': float(d)
        }
        
        # Métricas de eficiência
        
        # Percent Bias
        pbias = 100 * np.sum(era5_data - station_data) / np.sum(station_data)
        
        # Mean Absolute Percentage Error
        mape = 100 * np.mean(np.abs((station_data - era5_data) / station_data))
        
        # Symmetric Mean Absolute Percentage Error
        smape = 100 * np.mean(2 * np.abs(era5_data - station_data) / 
                             (np.abs(era5_data) + np.abs(station_data)))
        
        results['efficiency_metrics'] = {
            'percent_bias': float(pbias),
            'mape': float(mape),
            'smape': float(smape)
        }
        
        return results