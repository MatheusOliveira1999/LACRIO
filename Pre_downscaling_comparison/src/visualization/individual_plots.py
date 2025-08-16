"""
individual_plots.py
Gráficos individuais separados para comparação ERA5 vs Estação
Cada gráfico em arquivo próprio, sem combinações
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


class IndividualPlots:
    """Gráficos individuais ERA5 vs Estação"""
    
    def __init__(self, output_dir: str = "./static/img", figsize: Tuple[int, int] = (10, 8), dpi: int = 300):
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'observed': '#1f77b4',  # Azul para observado
            'era5': '#d62728',      # Vermelho para ERA5
            'line_11': '#d62728',   # Linha 1:1 vermelha
            'density': 'viridis'    # Mapa de cores para densidade
        }
    
    def scatter_density_plot(self, era5_data: pd.Series, 
                           station_data: pd.Series,
                           variable: str,
                           title: str = None,
                           save_path: str = None) -> str:
        """
        Gráfico de dispersão com densidade (individual)
        """
        # Aumentar figura para acomodar métricas e colorbar
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Garantir que os dados estejam alinhados e remover valores NaN
        # Converter para arrays numpy para garantir alinhamento
        era5_array = np.array(era5_data)
        station_array = np.array(station_data)
        
        # Remover valores NaN
        mask = ~(np.isnan(era5_array) | np.isnan(station_array))
        era5_clean = era5_array[mask]
        station_clean = station_array[mask]
        
        # Debug: Verificar dados
        print(f"      🔍 DEBUG - {variable}:")
        print(f"          Total pontos originais: {len(era5_array)}")
        print(f"          Pontos válidos após limpeza: {len(era5_clean)}")
        print(f"          ERA5 - Min: {era5_clean.min():.2f}, Max: {era5_clean.max():.2f}, Média: {era5_clean.mean():.2f}")
        print(f"          Estação - Min: {station_clean.min():.2f}, Max: {station_clean.max():.2f}, Média: {station_clean.mean():.2f}")
        
        # Verificar se os dados são idênticos (correlação = 1)
        if np.array_equal(era5_clean, station_clean):
            print(f"          ⚠️  ATENÇÃO: Dados são idênticos!")
        
        # Verificar se há variabilidade suficiente
        era5_std = np.std(era5_clean)
        station_std = np.std(station_data)
        print(f"          Desvio padrão - ERA5: {era5_std:.4f}, Estação: {station_std:.4f}")
        
        if era5_std < 1e-10 or station_std < 1e-10:
            print(f"          ⚠️  ATENÇÃO: Dados com pouca variabilidade!")
        
        # Calcular estatísticas
        pearson_corr, pearson_p = stats.pearsonr(era5_clean, station_clean)
        r2 = pearson_corr**2
        bias = np.mean(era5_clean - station_clean)
        rmse = np.sqrt(np.mean((era5_clean - station_clean)**2))
        
        print(f"          Correlação calculada: {pearson_corr:.6f}")
        print(f"          R²: {r2:.6f}")
        print(f"          P-value: {pearson_p:.6f}")
        
        # Debug adicional para identificar problema
        print(f"          🔬 ANÁLISE DETALHADA:")
        
        # 1. Verificar distribuição dos dados
        print(f"          ERA5 - Q25: {np.percentile(era5_clean, 25):.3f}, Mediana: {np.percentile(era5_clean, 50):.3f}, Q75: {np.percentile(era5_clean, 75):.3f}")
        print(f"          Estação - Q25: {np.percentile(station_clean, 25):.3f}, Mediana: {np.percentile(station_clean, 50):.3f}, Q75: {np.percentile(station_clean, 75):.3f}")
        
        # 2. Verificar outliers extremos
        era5_iqr = np.percentile(era5_clean, 75) - np.percentile(era5_clean, 25)
        station_iqr = np.percentile(station_clean, 75) - np.percentile(station_clean, 25)
        era5_outliers = np.sum((era5_clean < np.percentile(era5_clean, 25) - 3*era5_iqr) | 
                               (era5_clean > np.percentile(era5_clean, 75) + 3*era5_iqr))
        station_outliers = np.sum((station_clean < np.percentile(station_clean, 25) - 3*station_iqr) | 
                                  (station_clean > np.percentile(station_clean, 75) + 3*station_iqr))
        print(f"          Outliers extremos - ERA5: {era5_outliers}, Estação: {station_outliers}")
        
        # 3. Verificar possível deslocamento sistemático
        diff = era5_clean - station_clean
        print(f"          Diferença - Min: {diff.min():.3f}, Max: {diff.max():.3f}, Std: {diff.std():.3f}")
        
        # 4. Verificar correlação sem outliers (robusta)
        from scipy.stats import spearmanr
        spearman_corr, _ = spearmanr(era5_clean, station_clean)
        print(f"          Correlação Spearman (robusta): {spearman_corr:.6f}")
        
        # 5. Mostrar exemplos de pares de dados
        n_examples = min(10, len(era5_clean))
        print(f"          Primeiros {n_examples} pares (ERA5, Estação):")
        for i in range(n_examples):
            print(f"            {i+1}: ({era5_clean[i]:.3f}, {station_clean[i]:.3f}) - diff: {era5_clean[i]-station_clean[i]:.3f}")
        
        # 6. Verificar se há padrão temporal estranho (primeiros vs últimos)
        mid_point = len(era5_clean) // 2
        first_half_corr = np.corrcoef(era5_clean[:mid_point], station_clean[:mid_point])[0,1]
        second_half_corr = np.corrcoef(era5_clean[mid_point:], station_clean[mid_point:])[0,1]
        print(f"          Correlação 1ª metade: {first_half_corr:.6f}, 2ª metade: {second_half_corr:.6f}")
        
        print("          " + "="*50)
        
        # Criar scatter plot com densidade
        xy = np.vstack([station_clean, era5_clean])
        z = gaussian_kde(xy)(xy)
        
        # Ordenar pontos por densidade
        idx = z.argsort()
        x_sorted = station_clean[idx]
        y_sorted = era5_clean[idx]
        z_sorted = z[idx]
        
        scatter = ax.scatter(x_sorted, y_sorted, c=z_sorted, s=25, alpha=0.7, 
                           cmap=self.colors['density'], edgecolors='none')
        
        # Linha 1:1
        min_val = min(station_clean.min(), era5_clean.min())
        max_val = max(station_clean.max(), era5_clean.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               color=self.colors['line_11'], linewidth=2, alpha=0.8)
        
        # Configurar eixos
        unit = '(°C)' if 'temp' in variable.lower() else '(mm)'
        ax.set_xlabel(f'Observado {unit}', fontsize=12)
        ax.set_ylabel(f'ERA5 {unit}', fontsize=12)
        
        # Título
        title_text = title if title else f"Observado vs ERA5 - {variable.title()}"
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        # Grid e aspectos
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Caixa de estatísticas (dentro do gráfico, canto superior esquerdo)
        stats_text = f'r² = {r2:.3f}\nCorrelação = {pearson_corr:.3f}\nRMSE = {rmse:.3f}\nBias = {bias:.3f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Densidade', rotation=270, labelpad=20, fontsize=11)
        
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/scatter_{variable}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def time_series_recent(self, era5_data: pd.Series, 
                          station_data: pd.Series,
                          dates: pd.Series,
                          variable: str,
                          n_days: int = 90,
                          title: str = None,
                          save_path: str = None) -> str:
        """
        Série temporal dos últimos N dias consecutivos com dados completos (individual)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Encontrar período de n_days consecutivos com dados completos
        def find_complete_period(era5, station, dates_series, n_days):
            """Encontra período de n_days consecutivos com dados válidos"""
            era5_array = np.array(era5)
            station_array = np.array(station)
            
            # Criar máscara de dados válidos (não-NaN)
            valid_mask = ~(np.isnan(era5_array) | np.isnan(station_array))
            
            # Procurar janela de n_days consecutivos válidos (começando do final)
            for start_idx in range(len(valid_mask) - n_days, -1, -1):
                window_mask = valid_mask[start_idx:start_idx + n_days]
                if np.all(window_mask):  # Todos os dados válidos na janela
                    return start_idx, start_idx + n_days
            
            # Se não encontrar período completo, pegar o maior período disponível
            max_consecutive = 0
            best_start = len(valid_mask) - n_days if len(valid_mask) >= n_days else 0
            current_consecutive = 0
            current_start = 0
            
            for i, is_valid in enumerate(valid_mask):
                if is_valid:
                    if current_consecutive == 0:
                        current_start = i
                    current_consecutive += 1
                else:
                    if current_consecutive > max_consecutive:
                        max_consecutive = current_consecutive
                        best_start = current_start
                    current_consecutive = 0
            
            # Verificar último segmento
            if current_consecutive > max_consecutive:
                max_consecutive = current_consecutive
                best_start = current_start
            
            # Usar o período encontrado, limitado a n_days
            period_length = min(max_consecutive, n_days)
            end_idx = min(best_start + period_length, len(valid_mask))
            
            return best_start, end_idx
        
        # Encontrar melhor período
        start_idx, end_idx = find_complete_period(era5_data, station_data, dates, n_days)
        
        # Extrair dados do período encontrado
        if hasattr(era5_data, 'iloc'):
            recent_era5 = era5_data.iloc[start_idx:end_idx]
            recent_station = station_data.iloc[start_idx:end_idx] 
            recent_dates = dates.iloc[start_idx:end_idx]
        else:
            recent_era5 = era5_data[start_idx:end_idx]
            recent_station = station_data[start_idx:end_idx]
            recent_dates = dates[start_idx:end_idx]
        
        actual_days = end_idx - start_idx
        print(f"      📅 Período encontrado: {actual_days} dias consecutivos com dados completos")
        
        # Plot das séries
        ax.plot(recent_dates, recent_station, color=self.colors['observed'], 
               linewidth=1.5, label='Observado', alpha=0.8)
        ax.plot(recent_dates, recent_era5, color=self.colors['era5'], 
               linewidth=1.5, label='ERA5', alpha=0.8)
        
        # Configurar eixos
        unit = '(°C)' if 'temp' in variable.lower() else '(mm)'
        ax.set_ylabel(f'{variable.title()} {unit}', fontsize=12)
        ax.set_xlabel('Data', fontsize=12)
        
        # Título
        if title:
            title_text = title + f" ({actual_days} dias consecutivos)"
        else:
            title_text = f'Série Temporal - {variable.title()} ({actual_days} dias consecutivos com dados completos)'
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        # Legenda fora da área do gráfico
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Rotacionar labels das datas
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/timeseries_{variable}_{actual_days}days_complete.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def time_series_complete(self, era5_data: pd.Series, 
                           station_data: pd.Series,
                           dates: pd.Series,
                           variable: str,
                           title: str = None,
                           save_path: str = None) -> str:
        """
        Série temporal completa (individual)
        """
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot das séries completas
        ax.plot(dates, station_data, color=self.colors['observed'], 
               linewidth=1, label='Observado', alpha=0.7)
        ax.plot(dates, era5_data, color=self.colors['era5'], 
               linewidth=1, label='ERA5', alpha=0.7)
        
        # Configurar eixos
        unit = '(°C)' if 'temp' in variable.lower() else '(mm)'
        ax.set_ylabel(f'{variable.title()} {unit}', fontsize=12)
        ax.set_xlabel('Data', fontsize=12)
        
        # Título
        title_text = title if title else f'Série Temporal Completa - {variable.title()}'
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        # Legenda fora da área do gráfico
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Rotacionar labels das datas
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/timeseries_complete_{variable}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def annual_cycle_plot(self, era5_df: pd.DataFrame, 
                         station_df: pd.DataFrame,
                         variable: str,
                         title: str = None,
                         save_path: str = None) -> str:
        """
        Ciclo anual médio com bandas de confiança (individual)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Preparar dados
        era5_df = era5_df.copy()
        station_df = station_df.copy()
        
        era5_df['date'] = pd.to_datetime(era5_df['date'])
        station_df['date'] = pd.to_datetime(station_df['date'])
        
        era5_df['month'] = era5_df['date'].dt.month
        station_df['month'] = station_df['date'].dt.month
        
        # Calcular estatísticas mensais
        months = range(1, 13)
        month_names = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                      'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        
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
               linewidth=2.5, markersize=8, label='Observado', alpha=0.9)
        ax.plot(x, era5_monthly, 's-', color=self.colors['era5'], 
               linewidth=2.5, markersize=8, label='ERA5', alpha=0.9)
        
        # Bandas de confiança (desvio padrão)
        ax.fill_between(x, station_monthly - station_std, station_monthly + station_std,
                       color=self.colors['observed'], alpha=0.3)
        ax.fill_between(x, era5_monthly - era5_std, era5_monthly + era5_std,
                       color=self.colors['era5'], alpha=0.3)
        
        # Configurar eixos
        unit = '(°C)' if 'temp' in variable.lower() else '(mm)'
        ax.set_xlabel('Mês', fontsize=12)
        ax.set_ylabel(f'{variable.title()} {unit}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(month_names)
        
        # Título
        title_text = title if title else f"Ciclo Anual Médio - {variable.title()}"
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        # Legenda fora da área do gráfico
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Adicionar estatísticas
        valid_mask = ~(np.isnan(era5_monthly) | np.isnan(station_monthly))
        if np.sum(valid_mask) > 0:
            corr_annual = np.corrcoef(era5_monthly[valid_mask], station_monthly[valid_mask])[0, 1]
            rmse_annual = np.sqrt(np.mean((era5_monthly[valid_mask] - station_monthly[valid_mask])**2))
            
            stats_text = f'Correlação Anual: {corr_annual:.3f}\nRMSE Anual: {rmse_annual:.3f}'
            ax.text(1.02, 0.15, stats_text, transform=ax.transAxes, 
                   verticalalignment='bottom', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/annual_cycle_{variable}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def distributions_plot(self, era5_data: pd.Series, 
                          station_data: pd.Series,
                          variable: str,
                          title: str = None,
                          save_path: str = None) -> str:
        """
        Histogramas de distribuições (individual)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calcular bins comuns
        bins = np.histogram_bin_edges(np.concatenate([era5_data.dropna(), station_data.dropna()]), bins=40)
        
        # Histogramas
        ax.hist(station_data.dropna(), bins=bins, alpha=0.6, label='Observado', 
               color=self.colors['observed'], density=True, edgecolor='black', linewidth=0.5)
        ax.hist(era5_data.dropna(), bins=bins, alpha=0.6, label='ERA5', 
               color=self.colors['era5'], density=True, edgecolor='black', linewidth=0.5)
        
        # Configurar eixos
        unit = '(°C)' if 'temp' in variable.lower() else '(mm)'
        ax.set_xlabel(f'{variable.title()} {unit}', fontsize=12)
        ax.set_ylabel('Densidade', fontsize=12)
        
        # Título
        title_text = title if title else f"Distribuições - {variable.title()}"
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        # Legenda fora da área do gráfico
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Estatísticas básicas
        era5_mean = era5_data.mean()
        station_mean = station_data.mean()
        era5_std = era5_data.std()
        station_std = station_data.std()
        
        stats_text = f'Observado: μ={station_mean:.2f}, σ={station_std:.2f}\nERA5: μ={era5_mean:.2f}, σ={era5_std:.2f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/distributions_{variable}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def boxplot_monthly(self, era5_df: pd.DataFrame, 
                       station_df: pd.DataFrame,
                       variable: str,
                       title: str = None,
                       save_path: str = None) -> str:
        """
        Boxplots mensais (individual)
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Preparar dados
        merged = pd.merge(era5_df[['date', variable]], 
                         station_df[['date', variable]], 
                         on='date', suffixes=('_era5', '_station'))
        merged['date'] = pd.to_datetime(merged['date'])
        merged['month'] = merged['date'].dt.month
        
        # Preparar dados para boxplot
        months = range(1, 13)
        month_names = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                      'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        
        box_data = []
        positions = []
        colors = []
        
        for month in months:
            month_data = merged[merged['month'] == month]
            if len(month_data) > 0:
                box_data.append(month_data[f'{variable}_station'].dropna())
                box_data.append(month_data[f'{variable}_era5'].dropna())
                positions.extend([month*2-0.4, month*2+0.4])
                colors.extend([self.colors['observed'], self.colors['era5']])
        
        # Criar boxplots
        bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
        
        # Colorir boxplots
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
        
        # Configurar eixos
        unit = '(°C)' if 'temp' in variable.lower() else '(mm)'
        ax.set_xlabel('Mês', fontsize=12)
        ax.set_ylabel(f'{variable.title()} {unit}', fontsize=12)
        ax.set_xticks([month*2 for month in months])
        ax.set_xticklabels(month_names)
        
        # Título
        title_text = title if title else f"Variabilidade Mensal - {variable.title()}"
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        # Legenda personalizada posicionada para evitar conflito
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['observed'], alpha=0.7, label='Observado'),
            Patch(facecolor=self.colors['era5'], alpha=0.7, label='ERA5')
        ]
        ax.legend(handles=legend_elements, fontsize=11, loc='best', framealpha=0.9)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/boxplot_monthly_{variable}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def qq_plot(self, era5_data: pd.Series, 
               station_data: pd.Series,
               variable: str,
               title: str = None,
               save_path: str = None) -> str:
        """
        Q-Q Plot para comparar distribuições (individual)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Remover NaN
        era5_clean = era5_data.dropna()
        station_clean = station_data.dropna()
        
        # Calcular quantis
        n_quantiles = min(len(era5_clean), len(station_clean), 1000)
        quantiles = np.linspace(0.01, 0.99, n_quantiles)
        
        era5_quantiles = np.quantile(era5_clean, quantiles)
        station_quantiles = np.quantile(station_clean, quantiles)
        
        # Plot dos quantis
        ax.scatter(station_quantiles, era5_quantiles, alpha=0.6, s=20, 
                  color=self.colors['era5'], edgecolors='black', linewidth=0.5)
        
        # Linha 1:1
        min_val = min(station_quantiles.min(), era5_quantiles.min())
        max_val = max(station_quantiles.max(), era5_quantiles.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               color='black', linestyle='--', linewidth=2, alpha=0.8, label='Linha 1:1')
        
        # Configurar eixos
        unit = '(°C)' if 'temp' in variable.lower() else '(mm)'
        ax.set_xlabel(f'Quantis Observado {unit}', fontsize=12)
        ax.set_ylabel(f'Quantis ERA5 {unit}', fontsize=12)
        
        # Título
        title_text = title if title else f"Q-Q Plot - {variable.title()}"
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        # Legenda posicionada para evitar sobreposição
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Calcular R² dos quantis - posicionado dentro do gráfico (canto superior esquerdo)
        r2_qq = stats.pearsonr(station_quantiles, era5_quantiles)[0]**2
        ax.text(0.05, 0.95, f'R² quantis: {r2_qq:.3f}', transform=ax.transAxes, 
               verticalalignment='top', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/qq_plot_{variable}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def bias_analysis_plot(self, era5_data: pd.Series, 
                          station_data: pd.Series,
                          dates: pd.Series,
                          variable: str,
                          title: str = None,
                          save_path: str = None) -> str:
        """
        Análise de bias temporal (individual)
        """
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Calcular bias
        bias = era5_data - station_data
        
        # Plot do bias temporal
        ax.plot(dates, bias, color=self.colors['era5'], alpha=0.7, linewidth=1)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=bias.mean(), color='red', linestyle='-', alpha=0.8, linewidth=2,
                  label=f'Bias médio: {bias.mean():.3f}')
        
        # Configurar eixos
        unit = '(°C)' if 'temp' in variable.lower() else '(mm)'
        ax.set_ylabel(f'Bias ERA5-Observado {unit}', fontsize=12)
        ax.set_xlabel('Data', fontsize=12)
        
        # Título
        title_text = title if title else f"Evolução Temporal do Bias - {variable.title()}"
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        # Legenda e grid
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Estatísticas do bias (fora do gráfico)
        bias_stats = f'Bias médio: {bias.mean():.3f}\nBias mediano: {bias.median():.3f}\nDesvio padrão: {bias.std():.3f}'
        ax.text(1.02, 0.98, bias_stats, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # Rotacionar labels das datas
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Salvar
        if save_path is None:
            save_path = f"{self.output_dir}/bias_temporal_{variable}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
