"""
era5_station_plots.py
Gráficos especializados para comparação ERA5 vs Estação no estilo dos exemplos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ERA5StationPlots:
    """Gráficos especializados ERA5 vs Estação"""
    
    def __init__(self, output_dir: str = "./static/img", figsize: Tuple[int, int] = (15, 6), dpi: int = 300):
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'observed': '#1f77b4',  # Azul para observado
            'era5': '#d62728',      # Vermelho para ERA5
            'line_11': '#d62728',   # Linha 1:1 vermelha
            'density': 'viridis'    # Mapa de cores para densidade
        }
    
    def scatter_with_density(self, era5_data: pd.Series, 
                           station_data: pd.Series,
                           variable: str,
                           title: str = None,
                           save_path: str = None) -> str:
        """
        Gráfico de dispersão com densidade igual ao exemplo
        
        Parameters:
        -----------
        era5_data : pd.Series
            Dados ERA5
        station_data : pd.Series
            Dados da estação (observado)
        variable : str
            Nome da variável
        title : str, optional
            Título personalizado
        save_path : str, optional
            Caminho para salvar
            
        Returns:
        --------
        str
            Caminho do arquivo salvo
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Remover valores NaN
        mask = ~(np.isnan(era5_data) | np.isnan(station_data))
        era5_clean = era5_data[mask]
        station_clean = station_data[mask]
        
        # ===== GRÁFICO 1: SCATTER COM DENSIDADE =====
        
        # Calcular estatísticas
        pearson_corr, pearson_p = stats.pearsonr(era5_clean, station_clean)
        r2 = pearson_corr**2
        bias = np.mean(era5_clean - station_clean)
        rmse = np.sqrt(np.mean((era5_clean - station_clean)**2))
        
        # Criar scatter plot com densidade
        xy = np.vstack([station_clean, era5_clean])
        z = gaussian_kde(xy)(xy)
        
        # Ordenar pontos por densidade para que os mais densos fiquem por cima
        idx = z.argsort()
        x_sorted = station_clean.iloc[idx] if hasattr(station_clean, 'iloc') else station_clean[idx]
        y_sorted = era5_clean.iloc[idx] if hasattr(era5_clean, 'iloc') else era5_clean[idx]
        z_sorted = z[idx]
        
        scatter = ax1.scatter(x_sorted, y_sorted, c=z_sorted, s=20, alpha=0.7, 
                            cmap=self.colors['density'], edgecolors='none')
        
        # Linha 1:1
        min_val = min(station_clean.min(), era5_clean.min())
        max_val = max(station_clean.max(), era5_clean.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 
                color=self.colors['line_11'], linewidth=2, alpha=0.8, label='Linha 1:1')
        
        # Configurar eixos
        ax1.set_xlabel(f'Observado (°C)' if 'temp' in variable.lower() else 'Observado (mm)')
        ax1.set_ylabel(f'ERA5 (°C)' if 'temp' in variable.lower() else 'ERA5 (mm)')
        
        # Título do subplot
        subtitle = f"Observado vs ERA5 - {variable.upper()}"
        ax1.set_title(subtitle, fontsize=12, pad=10)
        
        # Caixa de estatísticas (igual ao exemplo)
        stats_text = f'r² = {r2:.2f}\nCorrelação = {pearson_corr:.2f}\nRMSE = {rmse:.3f}\nbias = {bias:.3f}'
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # Grid
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Densidade', rotation=270, labelpad=15)
        
        # ===== GRÁFICO 2: SÉRIE TEMPORAL (ÚLTIMOS 90 DIAS) =====
        
        # Pegar dados dos últimos 90 registros disponíveis
        n_points = min(90, len(era5_clean))
        recent_era5 = era5_clean.tail(n_points) if hasattr(era5_clean, 'tail') else era5_clean[-n_points:]
        recent_station = station_clean.tail(n_points) if hasattr(station_clean, 'tail') else station_clean[-n_points:]
        
        # Criar índice temporal
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n_points, freq='D')
        
        # Plot das séries
        ax2.plot(dates, recent_station, color=self.colors['observed'], 
                linewidth=1.5, label='Observado', alpha=0.8)
        ax2.plot(dates, recent_era5, color=self.colors['era5'], 
                linewidth=1.5, label='ERA5', alpha=0.8)
        
        # Configurar eixos
        ax2.set_ylabel(f'{variable.title()} (°C)' if 'temp' in variable.lower() else f'{variable.title()} (mm)')
        ax2.set_xlabel('Data')
        ax2.set_title(f'Série Temporal (Últimos {n_points} dias)', fontsize=12, pad=10)
        
        # Legenda
        ax2.legend(loc='upper right')
        
        # Grid
        ax2.grid(True, alpha=0.3)
        
        # Rotacionar labels das datas
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Título principal
        main_title = title if title else f"Resultados da Comparação - ERA5 - {variable.title()}"
        fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/era5_vs_station_{variable}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def annual_cycle_comparison(self, era5_df: pd.DataFrame, 
                              station_df: pd.DataFrame,
                              variable: str,
                              title: str = None,
                              save_path: str = None) -> str:
        """
        Gráfico do ciclo anual médio com bandas de confiança
        
        Parameters:
        -----------
        era5_df : pd.DataFrame
            DataFrame ERA5 com colunas 'date' e variável
        station_df : pd.DataFrame
            DataFrame da estação com colunas 'date' e variável
        variable : str
            Nome da variável
        title : str, optional
            Título personalizado
        save_path : str, optional
            Caminho para salvar
            
        Returns:
        --------
        str
            Caminho do arquivo salvo
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Preparar dados
        era5_df = era5_df.copy()
        station_df = station_df.copy()
        
        era5_df['date'] = pd.to_datetime(era5_df['date'])
        station_df['date'] = pd.to_datetime(station_df['date'])
        
        era5_df['month'] = era5_df['date'].dt.month
        station_df['month'] = station_df['date'].dt.month
        
        # Calcular estatísticas mensais
        months = range(1, 13)
        month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        
        era5_monthly = []
        station_monthly = []
        era5_std = []
        station_std = []
        
        for month in months:
            era5_month = era5_df[era5_df['month'] == month][variable].dropna()
            station_month = station_df[station_df['month'] == month][variable].dropna()
            
            era5_monthly.append(era5_month.mean() if len(era5_month) > 0 else np.nan)
            station_monthly.append(station_month.mean() if len(station_month) > 0 else np.nan)
            
            era5_std.append(era5_month.std() if len(era5_month) > 0 else 0)
            station_std.append(station_month.std() if len(station_month) > 0 else 0)
        
        # Converter para arrays numpy
        era5_monthly = np.array(era5_monthly)
        station_monthly = np.array(station_monthly)
        era5_std = np.array(era5_std)
        station_std = np.array(station_std)
        
        # Plot linhas principais
        x = np.arange(1, 13)
        ax.plot(x, station_monthly, 'o-', color=self.colors['observed'], 
                linewidth=2, markersize=6, label='Observado', alpha=0.8)
        ax.plot(x, era5_monthly, 's-', color=self.colors['era5'], 
                linewidth=2, markersize=6, label='ERA5', alpha=0.8)
        
        # Bandas de confiança (desvio padrão)
        ax.fill_between(x, station_monthly - station_std, station_monthly + station_std,
                       color=self.colors['observed'], alpha=0.3)
        ax.fill_between(x, era5_monthly - era5_std, era5_monthly + era5_std,
                       color=self.colors['era5'], alpha=0.3)
        
        # Configurar eixos
        ax.set_xlabel('Mês')
        ax.set_ylabel(f'{variable.title()} (°C)' if 'temp' in variable.lower() else f'{variable.title()} (mm)')
        ax.set_xticks(x)
        ax.set_xticklabels(month_names)
        
        # Título
        title_text = title if title else f"Análise Temporal - Ciclo Anual Médio - {variable.title()}"
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=15)
        
        # Legenda
        ax.legend(loc='upper right')
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Adicionar estatísticas gerais
        valid_mask = ~(np.isnan(era5_monthly) | np.isnan(station_monthly))
        if np.sum(valid_mask) > 0:
            corr_annual = np.corrcoef(era5_monthly[valid_mask], station_monthly[valid_mask])[0, 1]
            rmse_annual = np.sqrt(np.mean((era5_monthly[valid_mask] - station_monthly[valid_mask])**2))
            
            stats_text = f'Correlação Anual: {corr_annual:.3f}\nRMSE Anual: {rmse_annual:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/annual_cycle_{variable}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def combined_analysis_plot(self, era5_df: pd.DataFrame, 
                             station_df: pd.DataFrame,
                             variable: str,
                             title: str = None,
                             save_path: str = None) -> str:
        """
        Gráfico combinado com múltiplas análises
        
        Parameters:
        -----------
        era5_df : pd.DataFrame
            DataFrame ERA5
        station_df : pd.DataFrame
            DataFrame da estação
        variable : str
            Nome da variável
        title : str, optional
            Título personalizado
        save_path : str, optional
            Caminho para salvar
            
        Returns:
        --------
        str
            Caminho do arquivo salvo
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Sincronizar dados
        merged = pd.merge(era5_df[['date', variable]], 
                         station_df[['date', variable]], 
                         on='date', suffixes=('_era5', '_station'))
        
        era5_data = merged[f'{variable}_era5'].dropna()
        station_data = merged[f'{variable}_station'].dropna()
        
        # ===== SUBPLOT 1: SCATTER COM DENSIDADE =====
        ax1 = plt.subplot(2, 3, 1)
        
        # Calcular estatísticas
        pearson_corr, _ = stats.pearsonr(era5_data, station_data)
        r2 = pearson_corr**2
        bias = np.mean(era5_data - station_data)
        rmse = np.sqrt(np.mean((era5_data - station_data)**2))
        
        # Scatter com densidade
        xy = np.vstack([station_data, era5_data])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        
        scatter = ax1.scatter(station_data.iloc[idx], era5_data.iloc[idx], 
                            c=z[idx], s=15, alpha=0.7, cmap='viridis')
        
        # Linha 1:1
        min_val = min(station_data.min(), era5_data.min())
        max_val = max(station_data.max(), era5_data.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Observado')
        ax1.set_ylabel('ERA5')
        ax1.set_title(f'Observado vs ERA5')
        ax1.grid(True, alpha=0.3)
        
        # Estatísticas
        stats_text = f'r² = {r2:.2f}\nRMSE = {rmse:.3f}\nbias = {bias:.3f}'
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # ===== SUBPLOT 2: SÉRIE TEMPORAL COMPLETA =====
        ax2 = plt.subplot(2, 3, (2, 3))
        
        dates = pd.to_datetime(merged['date'])
        ax2.plot(dates, station_data, color='blue', linewidth=1, label='Observado', alpha=0.7)
        ax2.plot(dates, era5_data, color='red', linewidth=1, label='ERA5', alpha=0.7)
        
        ax2.set_ylabel(f'{variable.title()}')
        ax2.set_title('Série Temporal Completa')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # ===== SUBPLOT 3: CICLO ANUAL =====
        ax3 = plt.subplot(2, 3, 4)
        
        merged['month'] = pd.to_datetime(merged['date']).dt.month
        monthly_stats = merged.groupby('month').agg({
            f'{variable}_era5': ['mean', 'std'],
            f'{variable}_station': ['mean', 'std']
        })
        
        months = range(1, 13)
        month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        
        era5_mean = monthly_stats[f'{variable}_era5']['mean']
        station_mean = monthly_stats[f'{variable}_station']['mean']
        era5_std = monthly_stats[f'{variable}_era5']['std']
        station_std = monthly_stats[f'{variable}_station']['std']
        
        ax3.plot(months, station_mean, 'o-', color='blue', linewidth=2, label='Observado')
        ax3.plot(months, era5_mean, 's-', color='red', linewidth=2, label='ERA5')
        
        ax3.fill_between(months, station_mean - station_std, station_mean + station_std,
                        color='blue', alpha=0.3)
        ax3.fill_between(months, era5_mean - era5_std, era5_mean + era5_std,
                        color='red', alpha=0.3)
        
        ax3.set_xlabel('Mês')
        ax3.set_ylabel(f'{variable.title()}')
        ax3.set_title('Ciclo Anual Médio')
        ax3.set_xticks(months)
        ax3.set_xticklabels(month_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # ===== SUBPLOT 4: HISTOGRAMA =====
        ax4 = plt.subplot(2, 3, 5)
        
        bins = np.histogram_bin_edges(np.concatenate([era5_data, station_data]), bins=30)
        ax4.hist(station_data, bins=bins, alpha=0.6, label='Observado', 
                color='blue', density=True, edgecolor='black', linewidth=0.5)
        ax4.hist(era5_data, bins=bins, alpha=0.6, label='ERA5', 
                color='red', density=True, edgecolor='black', linewidth=0.5)
        
        ax4.set_xlabel(f'{variable.title()}')
        ax4.set_ylabel('Densidade')
        ax4.set_title('Distribuições')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # ===== SUBPLOT 5: BOXPLOT MENSAL =====
        ax5 = plt.subplot(2, 3, 6)
        
        # Preparar dados para boxplot
        box_data = []
        positions = []
        colors = []
        
        for month in months:
            month_data = merged[merged['month'] == month]
            if len(month_data) > 0:
                box_data.append(month_data[f'{variable}_station'].dropna())
                box_data.append(month_data[f'{variable}_era5'].dropna())
                positions.extend([month*2-0.4, month*2+0.4])
                colors.extend(['blue', 'red'])
        
        bp = ax5.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax5.set_xlabel('Mês')
        ax5.set_ylabel(f'{variable.title()}')
        ax5.set_title('Variabilidade Mensal')
        ax5.set_xticks([month*2 for month in months])
        ax5.set_xticklabels(month_names)
        ax5.grid(True, alpha=0.3)
        
        # Título principal
        main_title = title if title else f"Análise Completa ERA5 vs Estação - {variable.title()}"
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/combined_analysis_{variable}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path