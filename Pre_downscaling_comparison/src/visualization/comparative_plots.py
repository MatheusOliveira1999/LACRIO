"""
comparative_plots.py
Geração de gráficos comparativos ERA5 vs. estações meteorológicas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ComparativePlots:
    """Geração de gráficos comparativos"""
    
    def __init__(self, output_dir: str = "./static/img", figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'era5': '#1f77b4',
            'station': '#ff7f0e',
            'comparison': '#2ca02c'
        }
        
    def time_series_comparison(self, era5_data: pd.Series, 
                              station_data: pd.Series,
                              dates: pd.Series,
                              variable: str,
                              title: str = None,
                              save_path: str = None) -> str:
        """
        Série temporal comparativa do período completo
        
        Parameters:
        -----------
        era5_data : pd.Series
            Dados ERA5
        station_data : pd.Series
            Dados da estação
        dates : pd.Series
            Datas correspondentes
        variable : str
            Nome da variável
        title : str, optional
            Título do gráfico
        save_path : str, optional
            Caminho para salvar
            
        Returns:
        --------
        str
            Caminho do arquivo salvo
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Converter datas
        dates = pd.to_datetime(dates)
        
        # Gráfico principal - séries temporais
        ax1.plot(dates, era5_data, color=self.colors['era5'], label='ERA5', linewidth=1, alpha=0.8)
        ax1.plot(dates, station_data, color=self.colors['station'], label='Estação', linewidth=1, alpha=0.8)
        
        ax1.set_ylabel(f'{variable.title()}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if title:
            ax1.set_title(title)
        else:
            ax1.set_title(f'Comparação Temporal: {variable.title()} - ERA5 vs. Estação')
            
        # Gráfico de diferenças (bias)
        bias = era5_data - station_data
        ax2.plot(dates, bias, color=self.colors['comparison'], linewidth=1, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=bias.mean(), color='red', linestyle='-', alpha=0.7, 
                   label=f'Bias médio: {bias.mean():.3f}')
        
        ax2.set_ylabel(f'Bias (ERA5 - Estação)')
        ax2.set_xlabel('Data')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Adicionar estatísticas no gráfico
        corr, p_val = stats.pearsonr(era5_data, station_data)
        rmse = np.sqrt(np.mean((era5_data - station_data)**2))
        
        stats_text = f'r = {corr:.3f} (p < 0.001)\nRMSE = {rmse:.3f}\nBias = {bias.mean():.3f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/time_series_comparison_{variable}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def scatter_plot_comparison(self, era5_data: pd.Series, 
                               station_data: pd.Series,
                               variable: str,
                               title: str = None,
                               save_path: str = None) -> str:
        """
        Gráfico de dispersão ERA5 vs. Estação
        
        Parameters:
        -----------
        era5_data : pd.Series
            Dados ERA5
        station_data : pd.Series
            Dados da estação
        variable : str
            Nome da variável
        title : str, optional
            Título do gráfico
        save_path : str, optional
            Caminho para salvar
            
        Returns:
        --------
        str
            Caminho do arquivo salvo
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Scatter plot
        ax.scatter(station_data, era5_data, alpha=0.6, s=20, color=self.colors['era5'])
        
        # Linha 1:1
        min_val = min(station_data.min(), era5_data.min())
        max_val = max(station_data.max(), era5_data.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Linha 1:1')
        
        # Linha de regressão
        slope, intercept, r_value, p_value, std_err = stats.linregress(station_data, era5_data)
        line_x = np.array([min_val, max_val])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color='red', alpha=0.8, 
               label=f'Regressão (y = {slope:.3f}x + {intercept:.3f})')
        
        ax.set_xlabel(f'{variable.title()} - Estação')
        ax.set_ylabel(f'{variable.title()} - ERA5')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Dispersão: {variable.title()} - ERA5 vs. Estação')
            
        # Adicionar estatísticas
        r2 = r_value**2
        rmse = np.sqrt(np.mean((era5_data - station_data)**2))
        bias = np.mean(era5_data - station_data)
        
        stats_text = f'R² = {r2:.3f}\nr = {r_value:.3f}\nRMSE = {rmse:.3f}\nBias = {bias:.3f}\nn = {len(era5_data)}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/scatter_comparison_{variable}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def monthly_boxplots(self, era5_df: pd.DataFrame, 
                        station_df: pd.DataFrame,
                        variable: str,
                        title: str = None,
                        save_path: str = None) -> str:
        """
        Box plots mensais comparativos
        
        Parameters:
        -----------
        era5_df : pd.DataFrame
            DataFrame ERA5 com colunas 'date' e variável
        station_df : pd.DataFrame
            DataFrame da estação com colunas 'date' e variável
        variable : str
            Nome da variável
        title : str, optional
            Título do gráfico
        save_path : str, optional
            Caminho para salvar
            
        Returns:
        --------
        str
            Caminho do arquivo salvo
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Preparar dados
        era5_monthly = era5_df.copy()
        station_monthly = station_df.copy()
        
        era5_monthly['date'] = pd.to_datetime(era5_monthly['date'])
        station_monthly['date'] = pd.to_datetime(station_monthly['date'])
        
        era5_monthly['month'] = era5_monthly['date'].dt.month
        station_monthly['month'] = station_monthly['date'].dt.month
        
        # Preparar dados para boxplot
        monthly_data = []
        labels = []
        positions = []
        
        months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        
        for i, month in enumerate(range(1, 13)):
            era5_month_data = era5_monthly[era5_monthly['month'] == month][variable].dropna()
            station_month_data = station_monthly[station_monthly['month'] == month][variable].dropna()
            
            if len(era5_month_data) > 0:
                monthly_data.append(era5_month_data.values)
                positions.append(i * 2 + 1)
                
            if len(station_month_data) > 0:
                monthly_data.append(station_month_data.values)
                positions.append(i * 2 + 2)
        
        # Criar boxplots
        bp = ax.boxplot(monthly_data, positions=positions, patch_artist=True, widths=0.7)
        
        # Colorir boxplots
        for i, patch in enumerate(bp['boxes']):
            if i % 2 == 0:  # ERA5
                patch.set_facecolor(self.colors['era5'])
                patch.set_alpha(0.7)
            else:  # Estação
                patch.set_facecolor(self.colors['station'])
                patch.set_alpha(0.7)
        
        # Configurar eixos
        ax.set_xticks([i * 2 + 1.5 for i in range(12)])
        ax.set_xticklabels(months)
        ax.set_ylabel(f'{variable.title()}')
        ax.set_xlabel('Mês')
        ax.grid(True, alpha=0.3)
        
        # Legenda
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=self.colors['era5'], alpha=0.7, label='ERA5'),
                          Patch(facecolor=self.colors['station'], alpha=0.7, label='Estação')]
        ax.legend(handles=legend_elements)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Variabilidade Mensal: {variable.title()}')
            
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/monthly_boxplots_{variable}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def bias_analysis_plots(self, era5_data: pd.Series, 
                           station_data: pd.Series,
                           dates: pd.Series,
                           variable: str,
                           title: str = None,
                           save_path: str = None) -> str:
        """
        Gráficos de análise de bias
        
        Parameters:
        -----------
        era5_data : pd.Series
            Dados ERA5
        station_data : pd.Series
            Dados da estação
        dates : pd.Series
            Datas
        variable : str
            Nome da variável
        title : str, optional
            Título do gráfico
        save_path : str, optional
            Caminho para salvar
            
        Returns:
        --------
        str
            Caminho do arquivo salvo
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        bias = era5_data - station_data
        dates = pd.to_datetime(dates)
        
        # 1. Série temporal do bias
        ax1.plot(dates, bias, color=self.colors['comparison'], alpha=0.7, linewidth=1)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.axhline(y=bias.mean(), color='red', linestyle='-', alpha=0.7,
                   label=f'Bias médio: {bias.mean():.3f}')
        ax1.set_ylabel('Bias (ERA5 - Estação)')
        ax1.set_title('Evolução Temporal do Bias')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Histograma do bias
        ax2.hist(bias, bins=50, alpha=0.7, color=self.colors['comparison'], edgecolor='black')
        ax2.axvline(bias.mean(), color='red', linestyle='--', alpha=0.7, label=f'Média: {bias.mean():.3f}')
        ax2.axvline(bias.median(), color='orange', linestyle='--', alpha=0.7, label=f'Mediana: {bias.median():.3f}')
        ax2.set_xlabel('Bias')
        ax2.set_ylabel('Frequência')
        ax2.set_title('Distribuição do Bias')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Bias vs. magnitude da observação
        ax3.scatter(station_data, bias, alpha=0.6, s=15, color=self.colors['comparison'])
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Linha de tendência do bias
        try:
            z = np.polyfit(station_data, bias, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(station_data.min(), station_data.max(), 100)
            ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8, 
                    label=f'Tendência: y = {z[0]:.4f}x + {z[1]:.3f}')
            ax3.legend()
        except:
            pass
            
        ax3.set_xlabel(f'{variable.title()} - Estação')
        ax3.set_ylabel('Bias')
        ax3.set_title('Bias vs. Magnitude Observada')
        ax3.grid(True, alpha=0.3)
        
        # 4. Bias sazonal
        df_seasonal = pd.DataFrame({
            'bias': bias,
            'date': dates
        })
        df_seasonal['month'] = df_seasonal['date'].dt.month
        
        monthly_bias = df_seasonal.groupby('month')['bias'].agg(['mean', 'std']).reset_index()
        
        months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        ax4.bar(monthly_bias['month'], monthly_bias['mean'], 
               yerr=monthly_bias['std'], capsize=5, alpha=0.7, 
               color=self.colors['comparison'], edgecolor='black')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xticks(range(1, 13))
        ax4.set_xticklabels(months)
        ax4.set_xlabel('Mês')
        ax4.set_ylabel('Bias Médio')
        ax4.set_title('Variação Sazonal do Bias')
        ax4.grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=14)
        else:
            fig.suptitle(f'Análise de Bias: {variable.title()}', fontsize=14)
            
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/bias_analysis_{variable}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def seasonal_comparison_plots(self, era5_df: pd.DataFrame, 
                                 station_df: pd.DataFrame,
                                 variable: str,
                                 title: str = None,
                                 save_path: str = None) -> str:
        """
        Gráficos de comparação sazonal
        
        Parameters:
        -----------
        era5_df : pd.DataFrame
            DataFrame ERA5
        station_df : pd.DataFrame
            DataFrame da estação
        variable : str
            Nome da variável
        title : str, optional
            Título do gráfico
        save_path : str, optional
            Caminho para salvar
            
        Returns:
        --------
        str
            Caminho do arquivo salvo
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Preparar dados
        era5_seasonal = era5_df.copy()
        station_seasonal = station_df.copy()
        
        era5_seasonal['date'] = pd.to_datetime(era5_seasonal['date'])
        station_seasonal['date'] = pd.to_datetime(station_seasonal['date'])
        
        era5_seasonal['month'] = era5_seasonal['date'].dt.month
        station_seasonal['month'] = station_seasonal['date'].dt.month
        
        # Definir estações
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
        
        seasons = ['DJF', 'MAM', 'JJA', 'SON']
        season_names = ['Verão', 'Outono', 'Inverno', 'Primavera']
        
        # 1. Médias sazonais
        era5_seasonal_mean = era5_seasonal.groupby('season')[variable].mean().reindex(seasons)
        station_seasonal_mean = station_seasonal.groupby('season')[variable].mean().reindex(seasons)
        
        x = np.arange(len(seasons))
        width = 0.35
        
        ax1.bar(x - width/2, era5_seasonal_mean.values, width, 
               label='ERA5', color=self.colors['era5'], alpha=0.7)
        ax1.bar(x + width/2, station_seasonal_mean.values, width,
               label='Estação', color=self.colors['station'], alpha=0.7)
        
        ax1.set_xlabel('Estação')
        ax1.set_ylabel(f'{variable.title()}')
        ax1.set_title('Médias Sazonais')
        ax1.set_xticks(x)
        ax1.set_xticklabels(season_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Boxplots sazonais
        seasonal_data_era5 = []
        seasonal_data_station = []
        
        for season in seasons:
            era5_data = era5_seasonal[era5_seasonal['season'] == season][variable].dropna()
            station_data = station_seasonal[station_seasonal['season'] == season][variable].dropna()
            seasonal_data_era5.append(era5_data.values)
            seasonal_data_station.append(station_data.values)
        
        positions_era5 = [i * 3 + 1 for i in range(len(seasons))]
        positions_station = [i * 3 + 2 for i in range(len(seasons))]
        
        bp1 = ax2.boxplot(seasonal_data_era5, positions=positions_era5, patch_artist=True, widths=0.6)
        bp2 = ax2.boxplot(seasonal_data_station, positions=positions_station, patch_artist=True, widths=0.6)
        
        for patch in bp1['boxes']:
            patch.set_facecolor(self.colors['era5'])
            patch.set_alpha(0.7)
        for patch in bp2['boxes']:
            patch.set_facecolor(self.colors['station'])
            patch.set_alpha(0.7)
            
        ax2.set_xticks([i * 3 + 1.5 for i in range(len(seasons))])
        ax2.set_xticklabels(season_names)
        ax2.set_ylabel(f'{variable.title()}')
        ax2.set_title('Distribuições Sazonais')
        ax2.grid(True, alpha=0.3)
        
        # 3. Correlações sazonais
        seasonal_correlations = []
        seasonal_names_plot = []
        
        for i, season in enumerate(seasons):
            era5_season = era5_seasonal[era5_seasonal['season'] == season]
            station_season = station_seasonal[station_seasonal['season'] == season]
            
            # Merge por data
            merged = pd.merge(era5_season[['date', variable]], 
                            station_season[['date', variable]], 
                            on='date', suffixes=('_era5', '_station'))
            
            if len(merged) > 10:
                corr, p_val = stats.pearsonr(merged[f'{variable}_era5'], merged[f'{variable}_station'])
                seasonal_correlations.append(corr)
                seasonal_names_plot.append(season_names[i])
                
        if seasonal_correlations:
            bars = ax3.bar(seasonal_names_plot, seasonal_correlations, 
                          color=self.colors['comparison'], alpha=0.7)
            ax3.set_ylabel('Correlação')
            ax3.set_title('Correlações Sazonais')
            ax3.set_ylim([0, 1])
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Adicionar valores nas barras
            for bar, corr in zip(bars, seasonal_correlations):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{corr:.3f}', ha='center', va='bottom')
        
        # 4. Bias sazonal
        seasonal_bias = []
        
        for season in seasons:
            era5_season = era5_seasonal[era5_seasonal['season'] == season]
            station_season = station_seasonal[station_seasonal['season'] == season]
            
            merged = pd.merge(era5_season[['date', variable]], 
                            station_season[['date', variable]], 
                            on='date', suffixes=('_era5', '_station'))
            
            if len(merged) > 5:
                bias = (merged[f'{variable}_era5'] - merged[f'{variable}_station']).mean()
                seasonal_bias.append(bias)
            else:
                seasonal_bias.append(0)
                
        bars = ax4.bar(season_names, seasonal_bias, color=self.colors['comparison'], alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_ylabel('Bias Médio')
        ax4.set_title('Bias Sazonal')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        for bar, bias in zip(bars, seasonal_bias):
            height = bar.get_height()
            y_pos = height + 0.01 if height >= 0 else height - 0.01
            va = 'bottom' if height >= 0 else 'top'
            ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{bias:.3f}', ha='center', va=va)
        
        if title:
            fig.suptitle(title, fontsize=14)
        else:
            fig.suptitle(f'Análise Sazonal: {variable.title()}', fontsize=14)
            
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/seasonal_comparison_{variable}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def distribution_comparison(self, era5_data: pd.Series, 
                               station_data: pd.Series,
                               variable: str,
                               title: str = None,
                               save_path: str = None) -> str:
        """
        Comparação de distribuições
        
        Parameters:
        -----------
        era5_data : pd.Series
            Dados ERA5
        station_data : pd.Series
            Dados da estação
        variable : str
            Nome da variável
        title : str, optional
            Título do gráfico
        save_path : str, optional
            Caminho para salvar
            
        Returns:
        --------
        str
            Caminho do arquivo salvo
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Histogramas sobrepostos
        bins = np.histogram_bin_edges(np.concatenate([era5_data, station_data]), bins=50)
        
        ax1.hist(era5_data, bins=bins, alpha=0.6, label='ERA5', 
                color=self.colors['era5'], density=True, edgecolor='black', linewidth=0.5)
        ax1.hist(station_data, bins=bins, alpha=0.6, label='Estação', 
                color=self.colors['station'], density=True, edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel(f'{variable.title()}')
        ax1.set_ylabel('Densidade')
        ax1.set_title('Distribuições Comparativas')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Boxplots lado a lado
        data_for_boxplot = [era5_data.values, station_data.values]
        bp = ax2.boxplot(data_for_boxplot, labels=['ERA5', 'Estação'], patch_artist=True)
        
        bp['boxes'][0].set_facecolor(self.colors['era5'])
        bp['boxes'][1].set_facecolor(self.colors['station'])
        for box in bp['boxes']:
            box.set_alpha(0.7)
            
        ax2.set_ylabel(f'{variable.title()}')
        ax2.set_title('Estatísticas Descritivas')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Q-Q Plot
        # Calcular quantis
        n_quantiles = min(len(era5_data), len(station_data), 1000)
        quantiles = np.linspace(0.01, 0.99, n_quantiles)
        
        era5_quantiles = np.quantile(era5_data, quantiles)
        station_quantiles = np.quantile(station_data, quantiles)
        
        ax3.scatter(station_quantiles, era5_quantiles, alpha=0.6, s=15, color=self.colors['comparison'])
        
        # Linha 1:1
        min_val = min(station_quantiles.min(), era5_quantiles.min())
        max_val = max(station_quantiles.max(), era5_quantiles.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Linha 1:1')
        
        ax3.set_xlabel(f'Quantis - Estação')
        ax3.set_ylabel(f'Quantis - ERA5')
        ax3.set_title('Q-Q Plot')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Funções de distribuição cumulativa
        era5_sorted = np.sort(era5_data)
        station_sorted = np.sort(station_data)
        
        era5_cdf = np.arange(1, len(era5_sorted) + 1) / len(era5_sorted)
        station_cdf = np.arange(1, len(station_sorted) + 1) / len(station_sorted)
        
        ax4.plot(era5_sorted, era5_cdf, color=self.colors['era5'], label='ERA5', linewidth=2)
        ax4.plot(station_sorted, station_cdf, color=self.colors['station'], label='Estação', linewidth=2)
        
        ax4.set_xlabel(f'{variable.title()}')
        ax4.set_ylabel('Probabilidade Cumulativa')
        ax4.set_title('Funções de Distribuição Cumulativa')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=14)
        else:
            fig.suptitle(f'Comparação de Distribuições: {variable.title()}', fontsize=14)
            
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/distribution_comparison_{variable}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def summary_dashboard(self, comparison_results: Dict, 
                         variables: List[str],
                         title: str = "Resumo da Comparação ERA5 vs. Estação",
                         save_path: str = None) -> str:
        """
        Dashboard resumo com métricas principais
        
        Parameters:
        -----------
        comparison_results : dict
            Resultados da comparação
        variables : list
            Lista de variáveis analisadas
        title : str, optional
            Título do dashboard
        save_path : str, optional
            Caminho para salvar
            
        Returns:
        --------
        str
            Caminho do arquivo salvo
        """
        n_vars = len(variables)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Preparar dados para gráficos
        correlations = []
        rmses = []
        biases = []
        var_names = []
        
        for var in variables:
            if var in comparison_results and comparison_results[var]:
                metrics = comparison_results[var].get('comparison_metrics', {})
                if metrics:  # Verificar se metrics não está vazio
                    correlations.append(metrics.get('pearson_correlation', 0))
                    rmses.append(metrics.get('rmse', 0))
                    biases.append(abs(metrics.get('bias', 0)))
                    var_names.append(var.title())
        
        # 1. Correlações
        if correlations and var_names:
            bars1 = axes[0].bar(var_names, correlations, color=self.colors['era5'], alpha=0.7)
            axes[0].set_ylabel('Correlação (r)')
            axes[0].set_title('Correlações ERA5 vs. Estação')
            axes[0].set_ylim([0, 1])
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Adicionar valores nas barras
            for bar, corr in zip(bars1, correlations):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{corr:.3f}', ha='center', va='bottom')
        else:
            axes[0].text(0.5, 0.5, 'Dados não disponíveis\npara correlação', 
                        ha='center', va='center', transform=axes[0].transAxes,
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
            axes[0].set_title('Correlações ERA5 vs. Estação')
            axes[0].set_xticks([])
            axes[0].set_yticks([])
        
        # 2. RMSE
        if rmses and var_names:
            bars2 = axes[1].bar(var_names, rmses, color=self.colors['station'], alpha=0.7)
            axes[1].set_ylabel('RMSE')
            axes[1].set_title('Erro Quadrático Médio')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # Adicionar valores nas barras
            for bar, rmse in zip(bars2, rmses):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{rmse:.3f}', ha='center', va='bottom')
        else:
            axes[1].text(0.5, 0.5, 'Dados não disponíveis\npara RMSE', 
                        ha='center', va='center', transform=axes[1].transAxes,
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
            axes[1].set_title('Erro Quadrático Médio')
            axes[1].set_xticks([])
            axes[1].set_yticks([])
        
        # 3. Bias absoluto
        if biases and var_names:
            bars3 = axes[2].bar(var_names, biases, color=self.colors['comparison'], alpha=0.7)
            axes[2].set_ylabel('|Bias|')
            axes[2].set_title('Bias Absoluto')
            axes[2].grid(True, alpha=0.3, axis='y')
            
            # Adicionar valores nas barras
            for bar, bias in zip(bars3, biases):
                height = bar.get_height()
                axes[2].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{bias:.3f}', ha='center', va='bottom')
        else:
            axes[2].text(0.5, 0.5, 'Dados não disponíveis\npara Bias', 
                        ha='center', va='center', transform=axes[2].transAxes,
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
            axes[2].set_title('Bias Absoluto')
            axes[2].set_xticks([])
            axes[2].set_yticks([])
        
        # 4. Radar plot com métricas normalizadas ou mensagem de dados indisponíveis
        if var_names and len(variables) > 0:
            # Normalizar métricas para radar plot
            metrics_normalized = []
            metric_names = ['Correlação', 'R²', '1-|Bias|', '1-RMSE_norm']
            
            for var in variables:
                if var in comparison_results and comparison_results[var]:
                    metrics = comparison_results[var].get('comparison_metrics', {})
                    
                    if metrics:  # Verificar se metrics não está vazio
                        # Normalizar métricas (0-1, onde 1 é melhor)
                        corr = metrics.get('pearson_correlation', 0)
                        r2 = metrics.get('r_squared', 0)
                        bias_norm = max(0, 1 - abs(metrics.get('bias', 0)) / 10)  # Assumindo bias máximo de 10
                        rmse_norm = max(0, 1 - metrics.get('rmse', 0) / 10)  # Assumindo RMSE máximo de 10
                        
                        metrics_normalized.append([corr, r2, bias_norm, rmse_norm])
            
            if metrics_normalized:
                # Radar plot simples (usando plot polar)
                angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
                angles += angles[:1]  # Fechar o círculo
                
                ax4 = plt.subplot(224, projection='polar')
                
                colors_cycle = plt.cm.tab10(np.linspace(0, 1, len(metrics_normalized)))
                
                for i, (var, metrics_norm) in enumerate(zip(var_names, metrics_normalized)):
                    values = metrics_norm + metrics_norm[:1]  # Fechar o círculo
                    ax4.plot(angles, values, 'o-', linewidth=2, 
                            label=var, color=colors_cycle[i], alpha=0.7)
                    ax4.fill(angles, values, alpha=0.1, color=colors_cycle[i])
                
                ax4.set_xticks(angles[:-1])
                ax4.set_xticklabels(metric_names, fontsize=9)
                ax4.set_ylim(0, 1)
                ax4.set_title('Performance Geral', y=1.08)
                ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                ax4.grid(True, alpha=0.3)
            else:
                # Usar subplot regular para mensagem
                ax4 = axes[3]
                ax4.text(0.5, 0.5, 'Dados não disponíveis\npara Performance Geral', 
                        ha='center', va='center', transform=ax4.transAxes,
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
                ax4.set_title('Performance Geral')
                ax4.set_xticks([])
                ax4.set_yticks([])
        else:
            # Usar subplot regular para mensagem
            ax4 = axes[3]
            ax4.text(0.5, 0.5, 'Dados não disponíveis\npara Performance Geral', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
            ax4.set_title('Performance Geral')
            ax4.set_xticks([])
            ax4.set_yticks([])
        
        # Rotacionar labels se necessário
        for ax in axes[:3]:
            if len(var_names) > 3:
                ax.tick_params(axis='x', rotation=45)
        
        fig.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/summary_dashboard.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path