"""
Módulo para geração de visualizações
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo

import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy import stats
import os
from typing import Dict, Any, Optional, List

from config.settings import IMG_FOLDER


class PlotGenerator:
    """Classe para gerar todas as visualizações do sistema"""
    
    def __init__(self, model):
        """
        Inicializa o gerador de plots
        
        Parameters:
        -----------
        model : WeatherDownscalingModel
            Modelo treinado
        """
        self.model = model
        self.variable = model.variable
        self.results = model.results
        
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")
    
    def generate_all_plots(self, save_plots: bool = True):
        """
        Gera todas as visualizações
        
        Parameters:
        -----------
        save_plots : bool
            Se deve salvar os plots
        """
        if not self.results:
            print("❌ Nenhum resultado para plotar")
            return
        
        print("📊 Gerando visualizações...")
        
        try:
            # Plots principais
            self.plot_results(save_plots=save_plots)
            self.plot_feature_importance(save_plot=save_plots)
            self.plot_temporal_analysis(save_plot=save_plots)
            self.plot_models_comparison(save_plot=save_plots)
            
            # Plots individuais (opcional)
            self.plot_all_models_results(save_plots=save_plots)
            self.plot_temporal_analysis_all_models(save_plots=save_plots)
            
            print("✅ Todas as visualizações geradas!")
            
        except Exception as e:
            print(f"❌ Erro ao gerar visualizações: {str(e)}")
    
    def plot_results(self, save_plots: bool = True):
        """Gera visualizações dos resultados principais"""
        if not self.results:
            print("Nenhum resultado para plotar")
            return
        
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Resultados do Downscaling - {self.variable.capitalize()}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Comparação de modelos
        ax = axes[0, 0]
        models = list(self.results.keys())
        rmse_values = [self.results[m]['metrics']['RMSE'] for m in models]
        r2_values = [self.results[m]['metrics']['R2'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.8, color='skyblue')
        ax.set_ylabel('RMSE', fontweight='bold')
        ax.set_xlabel('Modelos', fontweight='bold')
        ax.set_title('Comparação de Modelos', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars1, rmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Eixo secundário para R²
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, r2_values, width, label='R²', alpha=0.8, color='lightcoral')
        ax2.set_ylabel('R²', fontweight='bold')
        ax2.set_ylim(0, 1)
        
        # Adicionar valores nas barras R²
        for bar, value in zip(bars2, r2_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Legendas
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # 2. Melhor modelo - Observado vs Predito
        ax = axes[0, 1]
        best_model_name = min(self.results.items(), 
                             key=lambda x: x[1]['metrics']['RMSE'])[0]
        y_pred = self.results[best_model_name]['predictions']
        
        if self.variable == 'precipitation':
            y_true = np.expm1(self.model.y_test) if hasattr(self.model.y_test, '__iter__') else self.model.y_test
        else:
            y_true = self.model.y_test
        
        # Scatter plot
        scatter = ax.scatter(y_true, y_pred, alpha=0.6, s=20, c='blue', edgecolors='none')
        
        # Linha 1:1
        min_val = min(y_true.min() if hasattr(y_true, 'min') else min(y_true), 
                     y_pred.min() if hasattr(y_pred, 'min') else min(y_pred))
        max_val = max(y_true.max() if hasattr(y_true, 'max') else max(y_true), 
                     y_pred.max() if hasattr(y_pred, 'max') else max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
        
        # Estatísticas
        r2 = self.results[best_model_name]['metrics']['R2']
        rmse = self.results[best_model_name]['metrics']['RMSE']
        ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'Observado ({self._get_unit()})', fontweight='bold')
        ax.set_ylabel(f'Predito ({self._get_unit()})', fontweight='bold')
        ax.set_title(f'{best_model_name} - Observado vs Predito', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. Série temporal
        ax = axes[1, 0]
        n_points = min(365, len(self.model.test_dates))
        indices = np.linspace(0, len(self.model.test_dates)-1, n_points, dtype=int)
        
        dates_subset = self.model.test_dates.iloc[indices]
        y_true_subset = y_true.iloc[indices] if hasattr(y_true, 'iloc') else np.array(y_true)[indices]
        y_pred_subset = y_pred[indices]
        
        ax.plot(dates_subset, y_true_subset, 'b-', alpha=0.7, label='Observado', linewidth=1.5)
        ax.plot(dates_subset, y_pred_subset, 'r-', alpha=0.7, label='Predito', linewidth=1.5)
        ax.set_xlabel('Data', fontweight='bold')
        ax.set_ylabel(f'{self.variable.capitalize()} ({self._get_unit()})', fontweight='bold')
        ax.set_title('Série Temporal (Subset)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotacionar datas
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Distribuição dos resíduos
        ax = axes[1, 1]
        residuals = np.array(y_true) - y_pred
        
        # Histograma
        n_bins = min(30, len(residuals)//20)
        ax.hist(residuals, bins=n_bins, edgecolor='black', alpha=0.7, color='lightgreen')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(x=np.mean(residuals), color='orange', linestyle='-', linewidth=2, label='Média')
        
        ax.set_xlabel(f'Resíduos ({self._get_unit()})', fontweight='bold')
        ax.set_ylabel('Frequência', fontweight='bold')
        ax.set_title('Distribuição dos Resíduos', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Estatísticas dos resíduos
        bias = np.mean(residuals)
        std_res = np.std(residuals)
        ax.text(0.05, 0.95, f'Bias = {bias:.3f}\nStd = {std_res:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            filename = os.path.join(IMG_FOLDER, f'resultados_downscaling_{self.variable}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico principal salvo: {filename}")
        
        plt.close()
    
    def plot_feature_importance(self, model_name: str = None, top_n: int = 20, save_plot: bool = True):
        """Plota importância das features"""
        if not self.results:
            return
        
        # Selecionar modelo com feature importance
        if model_name is None:
            tree_models = [name for name in self.results.keys() 
                          if any(x in name for x in ['RandomForest', 'ExtraTrees', 'XGBoost', 'GradientBoosting'])]
            if tree_models:
                model_name = tree_models[0]
            else:
                print("❌ Nenhum modelo baseado em árvore disponível")
                return
        
        if model_name not in self.results:
            print(f"❌ Modelo {model_name} não encontrado")
            return
        
        model = self.results[model_name]['model']
        
        # Extrair importâncias
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            importances = model.named_steps['model'].feature_importances_
        else:
            print(f"❌ Modelo {model_name} não possui feature_importances_")
            return
        
        # Criar DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.model.features,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plotar
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(feature_importance)), feature_importance['importance'], 
                       color='steelblue', alpha=0.8)
        
        # Personalizar gráfico
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Importância', fontweight='bold', fontsize=12)
        plt.title(f'Top {top_n} Features Mais Importantes - {model_name}\n{self.variable.capitalize()}', 
                 fontweight='bold', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        # Adicionar valores nas barras
        for i, (bar, value) in enumerate(zip(bars, feature_importance['importance'])):
            plt.text(value + value*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_plot:
            filename = os.path.join(IMG_FOLDER, f'feature_importance_{self.variable}_{model_name.lower()}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Importância das features salva: {filename}")
        
        plt.close()
        return feature_importance
    
    def plot_temporal_analysis(self, save_plot: bool = True):
        """Análise temporal detalhada"""
        if not self.results:
            return
        
        # Melhor modelo
        best_model_name = min(self.results.items(), 
                             key=lambda x: x[1]['metrics']['RMSE'])[0]
        y_pred = self.results[best_model_name]['predictions']
        
        if self.variable == 'precipitation':
            y_true = np.expm1(self.model.y_test) if hasattr(self.model.y_test, '__iter__') else self.model.y_test
        else:
            y_true = self.model.y_test
        
        # Criar DataFrame para análise
        temporal_df = pd.DataFrame({
            'date': self.model.test_dates.values,
            'observed': y_true.values if hasattr(y_true, 'values') else y_true,
            'predicted': y_pred,
            'residual': (y_true.values if hasattr(y_true, 'values') else y_true) - y_pred
        })
        
        temporal_df['month'] = pd.to_datetime(temporal_df['date']).dt.month
        temporal_df['year'] = pd.to_datetime(temporal_df['date']).dt.year
        temporal_df['season'] = temporal_df['month'].apply(lambda x: (x % 12) // 3 + 1)
        
        # Análise mensal
        monthly_stats = temporal_df.groupby('month').agg({
            'observed': ['mean', 'std'],
            'predicted': ['mean', 'std'],
            'residual': ['mean', 'std']
        }).round(3)
        
        # Plotar
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Análise Temporal - {best_model_name} - {self.variable.capitalize()}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Médias mensais
        ax = axes[0, 0]
        months = range(1, 13)
        month_names = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                      'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        
        obs_means = [monthly_stats.loc[m, ('observed', 'mean')] for m in months]
        pred_means = [monthly_stats.loc[m, ('predicted', 'mean')] for m in months]
        obs_stds = [monthly_stats.loc[m, ('observed', 'std')] for m in months]
        pred_stds = [monthly_stats.loc[m, ('predicted', 'std')] for m in months]
        
        ax.plot(months, obs_means, 'o-', label='Observado', linewidth=2, markersize=6, color='blue')
        ax.plot(months, pred_means, 's-', label='Predito', linewidth=2, markersize=6, color='red')
        
        # Barras de erro
        ax.fill_between(months, 
                       np.array(obs_means) - np.array(obs_stds),
                       np.array(obs_means) + np.array(obs_stds),
                       alpha=0.2, color='blue')
        ax.fill_between(months, 
                       np.array(pred_means) - np.array(pred_stds),
                       np.array(pred_means) + np.array(pred_stds),
                       alpha=0.2, color='red')
        
        ax.set_xlabel('Mês', fontweight='bold')
        ax.set_ylabel(f'{self.variable.capitalize()} ({self._get_unit()})', fontweight='bold')
        ax.set_title('Variação Mensal', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(months)
        ax.set_xticklabels(month_names)
        
        # 2. Erro médio mensal
        ax = axes[0, 1]
        residual_means = [monthly_stats.loc[m, ('residual', 'mean')] for m in months]
        residual_stds = [monthly_stats.loc[m, ('residual', 'std')] for m in months]
        
        bars = ax.bar(months, residual_means, yerr=residual_stds, 
                     capsize=5, alpha=0.7, color='lightcoral')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Mês', fontweight='bold')
        ax.set_ylabel(f'Erro Médio ({self._get_unit()})', fontweight='bold')
        ax.set_title('Erro Médio Mensal', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(months)
        ax.set_xticklabels(month_names)
        
        # 3. Análise sazonal
        ax = axes[1, 0]
        seasonal_stats = temporal_df.groupby('season').agg({
            'observed': ['mean', 'std'],
            'predicted': ['mean', 'std']
        }).round(3)
        
        seasons = [1, 2, 3, 4]
        season_names = ['Verão', 'Outono', 'Inverno', 'Primavera']
        
        obs_seasonal = [seasonal_stats.loc[s, ('observed', 'mean')] for s in seasons]
        pred_seasonal = [seasonal_stats.loc[s, ('predicted', 'mean')] for s in seasons]
        
        x = np.arange(len(seasons))
        width = 0.35
        
        ax.bar(x - width/2, obs_seasonal, width, label='Observado', alpha=0.8, color='blue')
        ax.bar(x + width/2, pred_seasonal, width, label='Predito', alpha=0.8, color='red')
        
        ax.set_xlabel('Estação', fontweight='bold')
        ax.set_ylabel(f'{self.variable.capitalize()} ({self._get_unit()})', fontweight='bold')
        ax.set_title('Variação Sazonal', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(season_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Boxplot dos resíduos por mês
        ax = axes[1, 1]
        monthly_residuals = [temporal_df[temporal_df['month'] == m]['residual'].values 
                           for m in months]
        
        bp = ax.boxplot(monthly_residuals, labels=month_names, patch_artist=True)
        
        # Colorir boxplots
        colors = plt.cm.viridis(np.linspace(0, 1, 12))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Mês', fontweight='bold')
        ax.set_ylabel(f'Resíduos ({self._get_unit()})', fontweight='bold')
        ax.set_title('Distribuição Mensal dos Resíduos', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            filename = os.path.join(IMG_FOLDER, f'analise_temporal_{self.variable}_{best_model_name.lower()}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Análise temporal salva: {filename}")
        
        plt.close()
        return monthly_stats
    
    def plot_models_comparison(self, save_plot: bool = True):
        """Gera gráfico comparativo de todos os modelos"""
        if not self.results:
            return
        
        # Configurar figura
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Comparação de Modelos - {self.variable.capitalize()}', 
                     fontsize=16, fontweight='bold')
        
        # Preparar dados
        models = list(self.results.keys())
        
        # 1. Barras de RMSE e MAE
        ax = axes[0, 0]
        x = np.arange(len(models))
        width = 0.35
        
        rmse_values = [self.results[m]['metrics']['RMSE'] for m in models]
        mae_values = [self.results[m]['metrics']['MAE'] for m in models]
        
        bars1 = ax.bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.8)
        bars2 = ax.bar(x + width/2, mae_values, width, label='MAE', alpha=0.8)
        
        ax.set_ylabel('Erro', fontweight='bold')
        ax.set_title('RMSE e MAE por Modelo', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. R² e Correlação
        ax = axes[0, 1]
        r2_values = [self.results[m]['metrics']['R2'] for m in models]
        corr_values = [self.results[m]['metrics']['Correlation'] for m in models]
        
        bars1 = ax.bar(x - width/2, r2_values, width, label='R²', alpha=0.8)
        bars2 = ax.bar(x + width/2, corr_values, width, label='Correlação', alpha=0.8)
        
        ax.set_ylabel('Valor', fontweight='bold')
        ax.set_title('R² e Correlação por Modelo', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Bias
        ax = axes[0, 2]
        bias_values = [self.results[m]['metrics']['Bias'] for m in models]
        colors = ['green' if abs(b) < 0.5 else 'orange' if abs(b) < 1 else 'red' 
                  for b in bias_values]
        
        bars = ax.bar(models, bias_values, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel(f'Bias ({self._get_unit()})', fontweight='bold')
        ax.set_title('Bias por Modelo', fontweight='bold')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 4. Skill Score
        ax = axes[1, 0]
        skill_values = [self.results[m]['metrics']['Skill_Score'] for m in models]
        colors = ['green' if s > 0.5 else 'orange' if s > 0 else 'red' for s in skill_values]
        
        bars = ax.bar(models, skill_values, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel('Skill Score', fontweight='bold')
        ax.set_title('Skill Score por Modelo', fontweight='bold')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 5. Heatmap de métricas
        ax = axes[1, 1]
        metrics_names = ['RMSE', 'MAE', 'R2', 'Correlation', 'Bias', 'Skill_Score']
        metrics_matrix = []
        
        for model in models:
            row = []
            for metric in metrics_names:
                value = self.results[model]['metrics'].get(metric, np.nan)
                row.append(value)
            metrics_matrix.append(row)
        
        metrics_matrix = np.array(metrics_matrix)
        
        # Normalizar métricas para visualização
        normalized_matrix = np.zeros_like(metrics_matrix)
        for j, metric in enumerate(metrics_names):
            col = metrics_matrix[:, j]
            if metric in ['RMSE', 'MAE', 'Bias']:  # Menor é melhor
                normalized_matrix[:, j] = 1 - (col - col.min()) / (col.max() - col.min() + 1e-10)
            else:  # Maior é melhor
                normalized_matrix[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-10)
        
        im = ax.imshow(normalized_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(metrics_names)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax.set_yticklabels(models)
        ax.set_title('Heatmap de Métricas (Normalizado)', fontweight='bold')
        
        # 6. Ranking dos modelos
        ax = axes[1, 2]
        
        # Calcular score composto
        scores = []
        for model in models:
            m = self.results[model]['metrics']
            score = (
                m['RMSE'] * 0.3 + m['MAE'] * 0.2 + 
                (1 - m['R2']) * 0.3 + abs(m['Bias']) * 0.2
            )
            scores.append(score)
        
        # Ordenar modelos por score
        model_scores = list(zip(models, scores))
        model_scores.sort(key=lambda x: x[1])
        
        ranked_models = [m[0] for m in model_scores]
        ranked_scores = [m[1] for m in model_scores]
        
        # Plotar ranking
        y_pos = np.arange(len(ranked_models))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(ranked_models)))
        
        bars = ax.barh(y_pos, ranked_scores, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{i+1}. {m}" for i, m in enumerate(ranked_models)])
        ax.set_xlabel('Score Composto (menor é melhor)', fontweight='bold')
        ax.set_title('Ranking de Modelos', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_plot:
            filename = os.path.join(IMG_FOLDER, f'comparacao_modelos_{self.variable}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Comparação de modelos salva: {filename}")
        
        plt.close()
    
    def plot_all_models_results(self, save_plots: bool = True):
        """Gera visualizações para TODOS os modelos treinados"""
        if not self.results:
            return
        
        print("📊 Gerando gráficos individuais para cada modelo...")
        
        # Para cada modelo treinado
        for model_name, model_data in self.results.items():
            self._plot_individual_model(model_name, model_data, save_plots)
    
    def plot_temporal_analysis_all_models(self, save_plots: bool = True):
        """Análise temporal para todos os modelos"""
        if not self.results:
            return
        
        print("📊 Gerando análise temporal para cada modelo...")
        
        for model_name, model_data in self.results.items():
            self._plot_individual_temporal_analysis(model_name, model_data, save_plots)
    
    def _plot_individual_model(self, model_name: str, model_data: Dict[str, Any], save_plots: bool):
        """Plota resultados de um modelo individual"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Resultados - {model_name} - {self.variable.capitalize()}', 
                     fontsize=16, fontweight='bold')
        
        y_pred = model_data['predictions']
        metrics = model_data['metrics']
        
        if self.variable == 'precipitation':
            y_true = np.expm1(self.model.y_test) if hasattr(self.model.y_test, '__iter__') else self.model.y_test
        else:
            y_true = self.model.y_test
        
        # 1. Observado vs Predito
        ax = axes[0, 0]
        ax.scatter(y_true, y_pred, alpha=0.6, s=20, c='blue', edgecolors='none')
        
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
        
        stats_text = f'R² = {metrics["R2"]:.3f}\nRMSE = {metrics["RMSE"]:.3f}\nMAE = {metrics["MAE"]:.3f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'Observado ({self._get_unit()})', fontweight='bold')
        ax.set_ylabel(f'Predito ({self._get_unit()})', fontweight='bold')
        ax.set_title('Observado vs Predito', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Série temporal (últimos 90 dias)
        ax = axes[0, 1]
        n_points = min(90, len(self.model.test_dates))
        indices = np.arange(len(self.model.test_dates) - n_points, len(self.model.test_dates))
        
        dates_subset = self.model.test_dates.iloc[indices]
        y_true_subset = y_true.iloc[indices] if hasattr(y_true, 'iloc') else np.array(y_true)[indices]
        y_pred_subset = y_pred[indices]
        
        ax.plot(dates_subset, y_true_subset, 'b-', alpha=0.7, label='Observado', linewidth=1.5)
        ax.plot(dates_subset, y_pred_subset, 'r-', alpha=0.7, label='Predito', linewidth=1.5)
        ax.set_xlabel('Data', fontweight='bold')
        ax.set_ylabel(f'{self.variable.capitalize()} ({self._get_unit()})', fontweight='bold')
        ax.set_title('Série Temporal (Últimos 90 dias)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Distribuição dos resíduos
        ax = axes[1, 0]
        residuals = np.array(y_true) - y_pred
        
        n_bins = min(50, len(residuals)//10)
        n, bins, patches = ax.hist(residuals, bins=n_bins, density=True, 
                                  alpha=0.7, color='lightgreen', edgecolor='black')
        
        # Adicionar KDE se possível
        if len(residuals) > 10:
            try:
                kde = gaussian_kde(residuals)
                x_range = np.linspace(residuals.min(), residuals.max(), 200)
                ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
            except:
                pass
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(x=np.mean(residuals), color='orange', linestyle='-', linewidth=2, label='Média')
        
        ax.set_xlabel(f'Resíduos ({self._get_unit()})', fontweight='bold')
        ax.set_ylabel('Densidade', fontweight='bold')
        ax.set_title('Distribuição dos Resíduos', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 4. Q-Q Plot
        ax = axes[1, 1]
        try:
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.set_title('Q-Q Plot dos Resíduos', fontweight='bold')
            ax.grid(True, alpha=0.3)
        except:
            ax.text(0.5, 0.5, 'Q-Q Plot não disponível', 
                   transform=ax.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        if save_plots:
            model_name_clean = model_name.lower().replace(' ', '_')
            filename = os.path.join(IMG_FOLDER, f'resultados_{self.variable}_{model_name_clean}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  📊 {model_name} salvo")
        
        plt.close()
    
    def _plot_individual_temporal_analysis(self, model_name: str, model_data: Dict[str, Any], save_plots: bool):
        """Análise temporal individual de um modelo"""
        # Implementação similar ao plot_temporal_analysis mas para modelo específico
        # (código similar ao método principal, adaptado para modelo individual)
        pass
    
    def _get_unit(self) -> str:
        """Retorna a unidade de medida"""
        return '°C' if self.variable == 'temperature' else 'mm'


if __name__ == "__main__":
    print("📊 PlotGenerator inicializado!")