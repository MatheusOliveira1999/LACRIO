"""
data_comparator.py
Comparação e análise de dados ERA5 vs. estações meteorológicas
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class DataComparator:
    """Compara dados ERA5 vs. estações meteorológicas"""
    
    def __init__(self):
        self.comparison_results = {}
        
    def compare_variables(self, era5_df: pd.DataFrame, 
                         station_df: pd.DataFrame,
                         variables: List[str] = None) -> Dict:
        """
        Compara variáveis climáticas entre ERA5 e estação
        
        Parameters:
        -----------
        era5_df : pd.DataFrame
            Dados ERA5 sincronizados
        station_df : pd.DataFrame
            Dados da estação sincronizados
        variables : list, optional
            Lista de variáveis para comparar
            
        Returns:
        --------
        dict
            Resultados da comparação
        """
# (Início do seu método ou função)

        print("📊 Iniciando comparação de variáveis...")

        # =======================================================================
        # INÍCIO DA IMPLEMENTAÇÃO INTEGRADA
        # =======================================================================
        # Se a lista de variáveis não for fornecida, detecta automaticamente as comuns.
        if variables is None:
            print("\n--- Verificando automaticamente as colunas comuns ---")
            
            # ETAPA DE DEBUG: Imprime as colunas para inspeção visual.
            # Ajuda a identificar diferenças sutis (ex: 'temp' vs 'temperatura').
            print("Colunas do DataFrame ERA5:", list(era5_df.columns))
            print("Colunas do DataFrame da Estação:", list(station_df.columns))

            # Converte os nomes das colunas para conjuntos (sets) para usar a intersecção.
            era5_cols_set = set(era5_df.columns)
            station_cols_set = set(station_df.columns)

            # Acha a intersecção (colunas que existem em AMBOS os DataFrames).
            common_cols = era5_cols_set.intersection(station_cols_set)

            # Remove a coluna 'date' ou outras colunas que não são variáveis de dados.
            # A operação de diferença de conjuntos ('-') é ideal para isso.
            variables_set = common_cols - {'date'}

            # Converte o conjunto de volta para uma lista.
            variables = list(variables_set)
        # =======================================================================
        # FIM DA IMPLEMENTAÇÃO INTEGRADA
        # =======================================================================

        if not variables:
            print("\n⚠️ Nenhuma variável comum foi encontrada para comparação. Encerrando.")
            return {}

        print(f"\n✅ Variáveis comuns encontradas para análise: {variables}")

        # 1. SELECIONAR: Cria subconjuntos dos dados contendo apenas as variáveis de interesse.
        era5_subset = era5_df[variables]
        station_subset = station_df[variables]

        results = {}

        # 2. SINCRONIZAR E ANALISAR (uma variável por vez)
        for var in variables:
            print(f"  🔍 Analisando '{var}'...")

            # Remove valores nulos para a variável atual em cada série.
            era5_series = era5_subset[var].dropna()
            station_series = station_subset[var].dropna()

            # Encontra os índices (datas/horas) comuns onde ambos têm dados válidos.
            common_idx = era5_series.index.intersection(station_series.index)

            if len(common_idx) < 10:
                print(f"  ⚠️ Poucos dados sincronizados para '{var}' ({len(common_idx)} pontos). Análise pulada.")
                continue

            # Filtra as séries para manter apenas os dados nos tempos em comum.
            era5_sync = era5_series.loc[common_idx]
            station_sync = station_series.loc[common_idx]

            # Usar a função completa de comparação
            print(f"    -> Comparando {len(era5_sync)} pontos de dados para '{var}'.")
            var_results = self._compare_single_variable(era5_sync, station_sync, var)
            results[var] = var_results

        self.comparison_results = results # Se estiver em uma classe

        if not results:
            print("\n❌ Análise concluída, mas nenhuma variável tinha dados suficientes para comparação.")
        else:
            print(f"\n✅ Comparação concluída com sucesso para {len(results)} variáveis.")

        return results
# (Fim do seu método ou função)
        
    def _compare_single_variable(self, era5_data: pd.Series, 
                                station_data: pd.Series, 
                                var_name: str) -> Dict:
        """
        Compara uma única variável
        
        Parameters:
        -----------
        era5_data : pd.Series
            Dados ERA5 da variável
        station_data : pd.Series
            Dados da estação da variável
        var_name : str
            Nome da variável
            
        Returns:
        --------
        dict
            Resultados da comparação
        """
        results = {
            'variable': var_name,
            'n_points': len(era5_data),
            'era5_stats': {},
            'station_stats': {},
            'comparison_metrics': {},
            'bias_analysis': {},
            'distribution_analysis': {},
            'seasonal_analysis': {}
        }
        
        # Estatísticas básicas
        results['era5_stats'] = self._calculate_basic_stats(era5_data, 'ERA5')
        results['station_stats'] = self._calculate_basic_stats(station_data, 'Station')
        
        # Métricas de comparação
        results['comparison_metrics'] = self._calculate_comparison_metrics(era5_data, station_data)
        
        # Análise de bias
        results['bias_analysis'] = self._analyze_bias(era5_data, station_data)
        
        # Análise de distribuições
        results['distribution_analysis'] = self._analyze_distributions(era5_data, station_data)
        
        return results
        
    def _calculate_basic_stats(self, data: pd.Series, source: str) -> Dict:
        """Calcula estatísticas básicas"""
        return {
            'source': source,
            'count': len(data),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'median': float(data.median()),
            'q25': float(data.quantile(0.25)),
            'q75': float(data.quantile(0.75)),
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data))
        }
        
    def _calculate_comparison_metrics(self, era5_data: pd.Series, 
                                    station_data: pd.Series) -> Dict:
        """Calcula métricas de comparação"""
        
        # Correlações
        pearson_corr, pearson_p = stats.pearsonr(era5_data, station_data)
        spearman_corr, spearman_p = stats.spearmanr(era5_data, station_data)
        
        # Métricas de erro
        bias = float(np.mean(era5_data - station_data))
        mae = float(np.mean(np.abs(era5_data - station_data)))
        rmse = float(np.sqrt(np.mean((era5_data - station_data)**2)))
        
        # Coeficiente de determinação
        r2 = float(pearson_corr**2)
        
        # Nash-Sutcliffe Efficiency
        nse = 1 - (np.sum((station_data - era5_data)**2) / 
                  np.sum((station_data - np.mean(station_data))**2))
        
        # Percent Bias
        pbias = float(100 * np.sum(era5_data - station_data) / np.sum(station_data))
        
        # Mean Absolute Percentage Error
        mape = float(100 * np.mean(np.abs((station_data - era5_data) / station_data)))
        
        return {
            'pearson_correlation': float(pearson_corr),
            'pearson_p_value': float(pearson_p),
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'r_squared': r2,
            'bias': bias,
            'mae': mae,
            'rmse': rmse,
            'nse': float(nse),
            'pbias': pbias,
            'mape': mape
        }
        
    def _analyze_bias(self, era5_data: pd.Series, station_data: pd.Series) -> Dict:
        """Análise detalhada de bias"""
        
        bias_series = era5_data - station_data
        
        # Bias por percentis
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        bias_percentiles = {}
        
        for p in percentiles:
            threshold = np.percentile(station_data, p)
            mask = station_data >= threshold
            if mask.sum() > 0:
                bias_percentiles[f'p{p}'] = float(bias_series[mask].mean())
                
        # Bias sazonal (por mês)
        if hasattr(era5_data.index, 'month'):
            monthly_bias = {}
            for month in range(1, 13):
                mask = era5_data.index.month == month
                if mask.sum() > 0:
                    monthly_bias[f'month_{month}'] = float(bias_series[mask].mean())
        else:
            monthly_bias = {}
            
        return {
            'overall_bias': float(bias_series.mean()),
            'bias_std': float(bias_series.std()),
            'bias_range': [float(bias_series.min()), float(bias_series.max())],
            'systematic_bias': float(bias_series.mean()),
            'random_bias': float(bias_series.std()),
            'bias_percentiles': bias_percentiles,
            'monthly_bias': monthly_bias
        }
        
    def _analyze_distributions(self, era5_data: pd.Series, station_data: pd.Series) -> Dict:
        """Análise de distribuições"""
        
        # Testes de normalidade
        era5_shapiro = stats.shapiro(era5_data.sample(min(5000, len(era5_data))))
        station_shapiro = stats.shapiro(station_data.sample(min(5000, len(station_data))))
        
        # Teste de Kolmogorov-Smirnov
        ks_stat, ks_p = stats.ks_2samp(era5_data, station_data)
        
        # Teste de Mann-Whitney U
        mw_stat, mw_p = stats.mannwhitneyu(era5_data, station_data, alternative='two-sided')
        
        return {
            'era5_normality': {
                'shapiro_stat': float(era5_shapiro[0]),
                'shapiro_p': float(era5_shapiro[1]),
                'is_normal': era5_shapiro[1] > 0.05
            },
            'station_normality': {
                'shapiro_stat': float(station_shapiro[0]),
                'shapiro_p': float(station_shapiro[1]),
                'is_normal': station_shapiro[1] > 0.05
            },
            'ks_test': {
                'statistic': float(ks_stat),
                'p_value': float(ks_p),
                'distributions_similar': ks_p > 0.05
            },
            'mannwhitney_test': {
                'statistic': float(mw_stat),
                'p_value': float(mw_p),
                'medians_similar': mw_p > 0.05
            }
        }
        
    def seasonal_comparison(self, era5_df: pd.DataFrame, 
                          station_df: pd.DataFrame,
                          variables: List[str] = None) -> Dict:
        """
        Análise sazonal comparativa
        
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
            Análise sazonal por variável
        """
        print("🌱 Executando análise sazonal...")
        
        if variables is None:
            era5_vars = set(era5_df.columns) - {'date'}
            station_vars = set(station_df.columns) - {'date'}
            variables = list(era5_vars.intersection(station_vars))
            
        # Adicionar informações sazonais
        era5_seasonal = era5_df.copy()
        station_seasonal = station_df.copy()
        
        era5_seasonal['month'] = pd.to_datetime(era5_seasonal['date']).dt.month
        station_seasonal['month'] = pd.to_datetime(station_seasonal['date']).dt.month
        
        # Definir estações (Hemisfério Sul)
        def get_season(month):
            if month in [12, 1, 2]:
                return 'DJF'  # Verão
            elif month in [3, 4, 5]:
                return 'MAM'  # Outono
            elif month in [6, 7, 8]:
                return 'JJA'  # Inverno
            else:
                return 'SON'  # Primavera
                
        era5_seasonal['season'] = era5_seasonal['month'].apply(get_season)
        station_seasonal['season'] = station_seasonal['month'].apply(get_season)
        
        seasonal_results = {}
        
        for var in variables:
            if var in era5_df.columns and var in station_df.columns:
                var_seasonal = {}
                
                for season in ['DJF', 'MAM', 'JJA', 'SON']:
                    era5_season = era5_seasonal[era5_seasonal['season'] == season][var].dropna()
                    station_season = station_seasonal[station_seasonal['season'] == season][var].dropna()
                    
                    if len(era5_season) > 5 and len(station_season) > 5:
                        # Sincronizar índices
                        common_idx = era5_season.index.intersection(station_season.index)
                        if len(common_idx) > 5:
                            era5_sync = era5_season.loc[common_idx]
                            station_sync = station_season.loc[common_idx]
                            
                            var_seasonal[season] = self._calculate_comparison_metrics(era5_sync, station_sync)
                            var_seasonal[season]['era5_mean'] = float(era5_sync.mean())
                            var_seasonal[season]['station_mean'] = float(station_sync.mean())
                            var_seasonal[season]['n_points'] = len(common_idx)
                            
                seasonal_results[var] = var_seasonal
                
        return seasonal_results
        
    def extreme_events_analysis(self, era5_df: pd.DataFrame, 
                               station_df: pd.DataFrame,
                               variables: List[str] = None,
                               percentiles: List[float] = [95, 99, 99.9]) -> Dict:
        """
        Análise de eventos extremos
        
        Parameters:
        -----------
        era5_df : pd.DataFrame
            Dados ERA5
        station_df : pd.DataFrame
            Dados da estação
        variables : list, optional
            Variáveis para analisar
        percentiles : list
            Percentis para definir extremos
            
        Returns:
        --------
        dict
            Análise de eventos extremos
        """
        print("⚡ Analisando eventos extremos...")
        
        if variables is None:
            era5_vars = set(era5_df.columns) - {'date'}
            station_vars = set(station_df.columns) - {'date'}
            variables = list(era5_vars.intersection(station_vars))
            
        extreme_results = {}
        
        for var in variables:
            if var in era5_df.columns and var in station_df.columns:
                era5_data = era5_df[var].dropna()
                station_data = station_df[var].dropna()
                
                # Sincronizar
                common_idx = era5_data.index.intersection(station_data.index)
                if len(common_idx) < 10:
                    continue
                    
                era5_sync = era5_data.loc[common_idx]
                station_sync = station_data.loc[common_idx]
                
                var_extremes = {}
                
                for percentile in percentiles:
                    # Thresholds baseados na estação (referência)
                    high_threshold = np.percentile(station_sync, percentile)
                    low_threshold = np.percentile(station_sync, 100 - percentile)
                    
                    # Eventos extremos altos
                    high_events_station = station_sync >= high_threshold
                    high_events_era5 = era5_sync >= high_threshold
                    
                    # Eventos extremos baixos  
                    low_events_station = station_sync <= low_threshold
                    low_events_era5 = era5_sync <= low_threshold
                    
                    # Métricas de detecção
                    if high_events_station.sum() > 0:
                        high_hit_rate = (high_events_station & high_events_era5).sum() / high_events_station.sum()
                        high_false_alarm = (high_events_era5 & ~high_events_station).sum() / high_events_era5.sum() if high_events_era5.sum() > 0 else 0
                    else:
                        high_hit_rate = high_false_alarm = 0
                        
                    if low_events_station.sum() > 0:
                        low_hit_rate = (low_events_station & low_events_era5).sum() / low_events_station.sum()
                        low_false_alarm = (low_events_era5 & ~low_events_station).sum() / low_events_era5.sum() if low_events_era5.sum() > 0 else 0
                    else:
                        low_hit_rate = low_false_alarm = 0
                    
                    var_extremes[f'p{percentile}'] = {
                        'high_threshold': float(high_threshold),
                        'low_threshold': float(low_threshold),
                        'high_events_station': int(high_events_station.sum()),
                        'high_events_era5': int(high_events_era5.sum()),
                        'low_events_station': int(low_events_station.sum()),
                        'low_events_era5': int(low_events_era5.sum()),
                        'high_hit_rate': float(high_hit_rate),
                        'high_false_alarm_rate': float(high_false_alarm),
                        'low_hit_rate': float(low_hit_rate),
                        'low_false_alarm_rate': float(low_false_alarm)
                    }
                    
                # Valores máximos e mínimos absolutos
                var_extremes['absolute_extremes'] = {
                    'era5_max': float(era5_sync.max()),
                    'era5_min': float(era5_sync.min()),
                    'station_max': float(station_sync.max()),
                    'station_min': float(station_sync.min()),
                    'max_bias': float(era5_sync.max() - station_sync.max()),
                    'min_bias': float(era5_sync.min() - station_sync.min())
                }
                
                extreme_results[var] = var_extremes
                
        return extreme_results
        
    def get_comparison_summary(self) -> Dict:
        """
        Retorna resumo da comparação
        
        Returns:
        --------
        dict
            Resumo dos resultados
        """
        if not self.comparison_results:
            return {}
            
        summary = {
            'variables_analyzed': list(self.comparison_results.keys()),
            'best_correlations': {},
            'worst_correlations': {},
            'overall_performance': {}
        }
        
        correlations = {}
        biases = {}
        rmses = {}
        
        for var, results in self.comparison_results.items():
            metrics = results.get('comparison_metrics', {})
            correlations[var] = metrics.get('pearson_correlation', 0)
            biases[var] = abs(metrics.get('bias', 999))
            rmses[var] = metrics.get('rmse', 999)
            
        # Melhores e piores correlações
        if correlations:
            best_corr_var = max(correlations, key=correlations.get)
            worst_corr_var = min(correlations, key=correlations.get)
            
            summary['best_correlations'] = {
                'variable': best_corr_var,
                'correlation': correlations[best_corr_var]
            }
            summary['worst_correlations'] = {
                'variable': worst_corr_var,
                'correlation': correlations[worst_corr_var]
            }
            
        # Performance geral
        summary['overall_performance'] = {
            'avg_correlation': float(np.mean(list(correlations.values()))),
            'avg_bias': float(np.mean(list(biases.values()))),
            'avg_rmse': float(np.mean(list(rmses.values()))),
            'variables_with_good_correlation': sum(1 for c in correlations.values() if c > 0.7),
            'variables_with_poor_correlation': sum(1 for c in correlations.values() if c < 0.3)
        }
        
        return summary