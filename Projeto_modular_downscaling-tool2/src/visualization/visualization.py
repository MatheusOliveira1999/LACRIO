"""
visualization.py
Módulo para geração de gráficos e visualizações
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import gaussian_kde
from datetime import datetime
from scipy import stats


class Visualizer:
    """Classe para gerar visualizações dos resultados"""
    
    def __init__(self, model):
        self.model = model
        self.variable = model.variable
        self.results = model.results
        self.y_test = model.y_test  # ⬅️ ADICIONE ESTA LINHA
        self.test_dates = model.test_dates  # ⬅️ E ESTA LINHA
        self.station_name = self._extract_station_name()  # Extrair nome da estação
        
    def _extract_station_name(self):
        """Extrai o nome da estação do caminho do arquivo CSV"""
        try:
            # Verificar se há dados carregados com caminho do arquivo
            if hasattr(self.model, 'data_processor') and hasattr(self.model.data_processor, 'station_file'):
                file_path = self.model.data_processor.station_file
                # Extrair o primeiro nome antes do primeiro underscore
                filename = os.path.basename(file_path)
                station_name = filename.split('_')[0]
                return station_name
            else:
                return 'unknown_station'
        except Exception as e:
            print(f"Erro ao extrair nome da estação: {e}")
            return 'unknown_station'
    
    def _get_station_output_dir(self):
        """Retorna o diretório de saída específico da estação"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(project_root, 'static', 'img', self.station_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir
    
    def _is_result_valid(self, metrics):
        """
        Verifica se os resultados do modelo são válidos e não absurdos
        
        Args:
            metrics: Dicionário com métricas do modelo
            
        Returns:
            bool: True se os resultados são válidos, False caso contrário
        """
        try:
            rmse = metrics.get('RMSE', float('inf'))
            r2 = metrics.get('R2', -float('inf'))
            mae = metrics.get('MAE', float('inf'))
            bias = metrics.get('Bias', 0)
            
            # Critérios para valores absurdos
            # RMSE muito alto (> 1e6)
            if rmse > 1e6:
                return False
            
            # R² extremamente negativo (< -1e6)  
            if r2 < -1e6:
                return False
            
            # MAE muito alto (> 1e6)
            if mae > 1e6:
                return False
            
            # Bias extremo (> 1e6 em valor absoluto)
            if abs(bias) > 1e6:
                return False
            
            # Verificar se há valores NaN ou inf
            if any(np.isnan(v) or np.isinf(v) for v in [rmse, r2, mae, bias] if v is not None):
                return False
            
            return True
            
        except Exception as e:
            print(f"Erro ao validar métricas: {e}")
            return False

    # --- CORREÇÃO 1: Adicionado um método principal para gerar todos os gráficos ---
    def generate_all_plots(self, save_plots=True):
        """
        Ponto de entrada principal para gerar todas as visualizações.
        Chama os métodos de plotagem específicos.
        """
        if not self.results:
            print("Nenhum resultado para plotar")
            return

        print(f"\n📊 Gerando visualizações para estação: {self.station_name}")
        print("\n📊 Verificando resultados antes de gerar visualizações...")
        
        # Filtrar modelos com resultados válidos
        valid_models = {}
        invalid_models = []
        
        for model_name, model_data in self.results.items():
            metrics = model_data.get('metrics', {})
            if self._is_result_valid(metrics):
                valid_models[model_name] = model_data
                print(f"  ✓ {model_name}: Resultados válidos")
            else:
                invalid_models.append(model_name)
                print(f"  ❌ {model_name}: Resultados absurdos detectados - PULANDO visualizações")
                print(f"      RMSE: {metrics.get('RMSE', 'N/A')}")
                print(f"      R²: {metrics.get('R2', 'N/A')}")
        
        if not valid_models:
            print("\n⚠️ Nenhum modelo com resultados válidos encontrado. Nenhuma visualização será gerada.")
            return
        
        print(f"\n📊 Gerando visualizações para {len(valid_models)} modelo(s) válido(s)...")
        
        # Temporariamente substituir self.results apenas com modelos válidos
        original_results = self.results
        self.results = valid_models
        
        try:
            # Gerar gráfico de comparação
            self.plot_models_comparison(save_plot=save_plots)
            
            # Gerar gráficos individuais para cada modelo
            print("\n📈 Gerando gráficos individuais para cada modelo...")
            for model_name in self.results.keys():
                try:
                    self.plot_individual_model_results(model_name=model_name, save_plot=save_plots)
                    print(f"  ✓ Gráficos individuais gerados para {model_name}")
                except Exception as e:
                    print(f"  ❌ Erro ao gerar gráficos individuais para {model_name}: {e}")
            
            # Gerar análise temporal individual
            print("\n📅 Gerando análise temporal para cada modelo...")
            for model_name in self.results.keys():
                try:
                    self.plot_individual_temporal_analysis(model_name=model_name, save_plot=save_plots)
                    print(f"  ✓ Análise temporal gerada para {model_name}")
                except Exception as e:
                    print(f"  ❌ Erro ao gerar análise temporal para {model_name}: {e}")
            
            # Gerar gráficos de dispersão melhorados individuais
            print("\n🎯 Gerando gráficos de dispersão melhorados...")
            for model_name in self.results.keys():
                try:
                    self.plot_enhanced_scatter(model_name=model_name, save_plot=save_plots)
                    print(f"  ✓ Scatter melhorado gerado para {model_name}")
                except Exception as e:
                    print(f"  ❌ Erro ao gerar scatter melhorado para {model_name}: {e}")
            
            # Gerar feature importance apenas para modelos baseados em árvore
            tree_based_models = [name for name in self.results.keys() 
                               if 'RandomForest' in name or 'ExtraTrees' in name or 'XGBoost' in name or 'GradientBoosting' in name]
            
            if tree_based_models:
                print("\n🔍 Gerando feature importance para modelos baseados em árvore...")
                for model_name in tree_based_models:
                    try:
                        self.plot_feature_importance(model_name=model_name, save_plot=save_plots)
                        print(f"  ✓ Feature importance gerado para {model_name}")
                    except Exception as e:
                        print(f"  ❌ Erro ao gerar feature importance para {model_name}: {e}")
            else:
                print("\n⚠️ Nenhum modelo baseado em árvore encontrado para feature importance")
            
            print(f"✅ Visualizações geradas para estação {self.station_name}!")
            if invalid_models:
                print(f"⚠️ Modelos com resultados absurdos (sem visualizações): {', '.join(invalid_models)}")
                
        except Exception as e:
            print(f"❌ Erro ao gerar visualizações: {e}")
        finally:
            # Restaurar self.results original
            self.results = original_results

    def plot_individual_model_results(self, model_name, save_plot=True):
        """Gera gráficos individuais para um modelo específico"""
        if model_name not in self.results:
            print(f"Modelo {model_name} não encontrado")
            return
        
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        model_data = self.results[model_name]
        y_pred = model_data['predictions']
        metrics = model_data['metrics']
        
        if self.variable == 'precipitation':
            y_true = np.expm1(self.y_test)
        else:
            y_true = self.y_test
        
        # Gerar 4 gráficos separados
        self._plot_scatter_plot(model_name, y_true, y_pred, metrics, save_plot)
        self._plot_timeseries_90days(model_name, y_true, y_pred, save_plot)
        self._plot_residuals_distribution(model_name, y_true, y_pred, save_plot)
        self._plot_qq_plot(model_name, y_true, y_pred, save_plot)
    
    def _plot_scatter_plot(self, model_name, y_true, y_pred, metrics, save_plot):
        """Gráfico de dispersão individual"""
        fig, ax = plt.subplots(figsize=(8, 8))
        self._plot_scatter_density(ax, y_true, y_pred, metrics, model_name)
        plt.tight_layout()
        
        if save_plot:
            output_dir = self._get_station_output_dir()
            model_name_clean = model_name.lower().replace(' ', '_')
            filename = os.path.join(output_dir, f'scatter_{self.variable}_{model_name_clean}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_timeseries_90days(self, model_name, y_true, y_pred, save_plot):
        """Série temporal dos últimos 90 dias"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        n_points = min(90, len(self.test_dates))
        indices = np.arange(len(self.test_dates) - n_points, len(self.test_dates))
        
        dates_subset = self.test_dates.iloc[indices]
        y_true_subset = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_subset = y_pred[indices]
        
        ax.plot(dates_subset, y_true_subset, 'b-', alpha=0.7, label='Observado', linewidth=1.5)
        ax.plot(dates_subset, y_pred_subset, 'r-', alpha=0.7, label='Predito', linewidth=1.5)
        ax.set_xlabel('Data', fontweight='bold')
        ax.set_ylabel(f'{self.variable.capitalize()} ({self._get_unit()})', fontweight='bold')
        ax.set_title(f'Série Temporal (Últimos 90 dias) - {model_name} - Estação {self.station_name}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        
        if save_plot:
            output_dir = self._get_station_output_dir()
            model_name_clean = model_name.lower().replace(' ', '_')
            filename = os.path.join(output_dir, f'timeseries_{self.variable}_90days_{model_name_clean}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_residuals_distribution(self, model_name, y_true, y_pred, save_plot):
        """Distribuição dos resíduos"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        residuals = y_true - y_pred
        
        # Histograma com KDE
        n_bins = min(50, len(residuals)//10)
        n, bins, patches = ax.hist(residuals, bins=n_bins, density=True, 
                                alpha=0.7, color='lightgreen', edgecolor='black')
        
        # Adicionar KDE
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
        ax.set_title(f'Distribuição dos Resíduos - {model_name} - Estação {self.station_name}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Estatísticas
        bias = np.mean(residuals)
        std_res = np.std(residuals)
        skew = residuals.skew() if hasattr(residuals, 'skew') else 0
        kurt = residuals.kurtosis() if hasattr(residuals, 'kurtosis') else 0
        
        stats_text = f'Bias = {bias:.3f}\nStd = {std_res:.3f}\nSkew = {skew:.3f}\nKurt = {kurt:.3f}'
        ax.text(0.75, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = self._get_station_output_dir()
            model_name_clean = model_name.lower().replace(' ', '_')
            filename = os.path.join(output_dir, f'distributions_{self.variable}_{model_name_clean}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_qq_plot(self, model_name, y_true, y_pred, save_plot):
        """Q-Q Plot dos resíduos"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        residuals = y_true - y_pred
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot dos Resíduos - {model_name} - Estação {self.station_name}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = self._get_station_output_dir()
            model_name_clean = model_name.lower().replace(' ', '_')
            filename = os.path.join(output_dir, f'qq_plot_{self.variable}_{model_name_clean}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


    def plot_models_comparison(self, save_plot=True):
        """Gera gráficos comparativos separados de todos os modelos"""
        if not self.results:
            print("Nenhum resultado para plotar")
            return
        
        print("\n📊 Gerando gráficos de comparação individuais...")
        
        # Gerar cada gráfico de comparação separadamente
        try:
            self._plot_rmse_mae_comparison(save_plot)
            print("  ✓ Gráfico RMSE vs MAE gerado")
        except Exception as e:
            print(f"  ❌ Erro no gráfico RMSE vs MAE: {e}")
        
        try:
            self._plot_r2_correlation_comparison(save_plot)
            print("  ✓ Gráfico R² vs Correlação gerado")
        except Exception as e:
            print(f"  ❌ Erro no gráfico R² vs Correlação: {e}")
        
        try:
            self._plot_bias_comparison(save_plot)
            print("  ✓ Gráfico de Bias gerado")
        except Exception as e:
            print(f"  ❌ Erro no gráfico de Bias: {e}")
        
        try:
            self._plot_skill_score_comparison(save_plot)
            print("  ✓ Gráfico de Skill Score gerado")
        except Exception as e:
            print(f"  ❌ Erro no gráfico de Skill Score: {e}")
        
        try:
            self._plot_metrics_heatmap(save_plot)
            print("  ✓ Heatmap de métricas gerado")
        except Exception as e:
            print(f"  ❌ Erro no heatmap de métricas: {e}")
        
        try:
            self._plot_models_ranking(save_plot)
            print("  ✓ Ranking de modelos gerado")
        except Exception as e:
            print(f"  ❌ Erro no ranking de modelos: {e}")
    
    def _plot_rmse_mae_comparison(self, save_plot):
        """Gráfico de comparação RMSE e MAE"""
        models = list(self.results.keys())
        x = np.arange(len(models))
        width = 0.35
        
        rmse_values = [self.results[m]['metrics']['RMSE'] for m in models]
        mae_values = [self.results[m]['metrics']['MAE'] for m in models]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars1 = ax.bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.8, color='lightcoral')
        bars2 = ax.bar(x + width/2, mae_values, width, label='MAE', alpha=0.8, color='lightblue')
        
        ax.set_ylabel('Erro', fontweight='bold', fontsize=12)
        ax.set_title(f'RMSE e MAE por Modelo - {self.variable.capitalize()} - Estação {self.station_name}', 
                    fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Adicionar valores
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = self._get_station_output_dir()
            filename = os.path.join(output_dir, f'rmse_mae_comparison_{self.variable}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_r2_correlation_comparison(self, save_plot):
        """Gráfico de comparação R² e Correlação"""
        models = list(self.results.keys())
        x = np.arange(len(models))
        width = 0.35
        
        r2_values = [self.results[m]['metrics']['R2'] for m in models]
        corr_values = [self.results[m]['metrics']['Correlation'] for m in models]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars1 = ax.bar(x - width/2, r2_values, width, label='R²', alpha=0.8, color='lightgreen')
        bars2 = ax.bar(x + width/2, corr_values, width, label='Correlação', alpha=0.8, color='gold')
        
        ax.set_ylabel('Valor', fontweight='bold', fontsize=12)
        ax.set_title(f'R² e Correlação por Modelo - {self.variable.capitalize()} - Estação {self.station_name}', 
                    fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Adicionar valores
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = self._get_station_output_dir()
            filename = os.path.join(output_dir, f'r2_correlation_comparison_{self.variable}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_bias_comparison(self, save_plot):
        """Gráfico de comparação de Bias"""
        models = list(self.results.keys())
        bias_values = [self.results[m]['metrics']['Bias'] for m in models]
        colors = ['green' if abs(b) < 0.5 else 'orange' if abs(b) < 1 else 'red' 
                for b in bias_values]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(models, bias_values, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax.set_ylabel(f'Bias ({self._get_unit()})', fontweight='bold', fontsize=12)
        ax.set_title(f'Bias por Modelo - {self.variable.capitalize()} - Estação {self.station_name}', 
                    fontweight='bold', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Adicionar valores
        for bar, value in zip(bars, bias_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = self._get_station_output_dir()
            filename = os.path.join(output_dir, f'bias_comparison_{self.variable}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_skill_score_comparison(self, save_plot):
        """Gráfico de comparação de Skill Score"""
        models = list(self.results.keys())
        skill_values = [self.results[m]['metrics']['Skill_Score'] for m in models]
        colors = ['green' if s > 0.5 else 'orange' if s > 0 else 'red' for s in skill_values]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(models, skill_values, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_ylabel('Skill Score', fontweight='bold', fontsize=12)
        ax.set_title(f'Skill Score por Modelo - {self.variable.capitalize()} - Estação {self.station_name}', 
                    fontweight='bold', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Adicionar valores
        for bar, value in zip(bars, skill_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = self._get_station_output_dir()
            filename = os.path.join(output_dir, f'skill_score_comparison_{self.variable}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_heatmap(self, save_plot):
        """Heatmap de todas as métricas com normalização melhorada"""
        models = list(self.results.keys())
        metrics_names = ['RMSE', 'MAE', 'R2', 'Correlation', 'Bias', 'Skill_Score']
        
        # Coletar dados das métricas
        metrics_matrix = []
        for model in models:
            row = []
            for metric in metrics_names:
                value = self.results[model]['metrics'].get(metric, np.nan)
                row.append(value)
            metrics_matrix.append(row)
        
        metrics_matrix = np.array(metrics_matrix)
        
        # Normalização melhorada por tipo de métrica
        normalized_matrix = np.zeros_like(metrics_matrix)
        for j, metric in enumerate(metrics_names):
            col = metrics_matrix[:, j]
            valid_mask = ~np.isnan(col)
            
            if not np.any(valid_mask):
                normalized_matrix[:, j] = 0
                continue
                
            valid_values = col[valid_mask]
            
            if metric in ['RMSE', 'MAE']:
                # Menor é melhor - inversão simples
                max_val = float(np.max(valid_values))
                min_val = float(np.min(valid_values))
                if max_val != min_val:
                    normalized_matrix[:, j] = 1 - (col - min_val) / (max_val - min_val)
                else:
                    normalized_matrix[:, j] = 1
                    
            elif metric == 'Bias':
                # Melhor quando próximo de 0 - distância absoluta do zero
                abs_max = float(np.max(np.abs(valid_values)))
                if abs_max > 0:
                    normalized_matrix[:, j] = 1 - np.abs(col) / abs_max
                else:
                    normalized_matrix[:, j] = 1
                    
            elif metric == 'Skill_Score':
                # Skill Score: >0 é bom, <0 é ruim, 1 é perfeito
                # Normalizar para [0,1] onde 1 representa melhor performance
                normalized_matrix[:, j] = np.clip((col + 1) / 2, 0, 1)
                
            else:  # R2, Correlation - maior é melhor
                max_val = float(np.max(valid_values))
                min_val = float(np.min(valid_values))
                if max_val != min_val:
                    normalized_matrix[:, j] = (col - min_val) / (max_val - min_val)
                else:
                    normalized_matrix[:, j] = 1
        
        # Tratar NaN values na matriz normalizada
        normalized_matrix = np.nan_to_num(normalized_matrix, nan=0.5)
        
        # Determinar limites e colormap
        vmin_actual = np.nanmin(normalized_matrix)
        vmax_actual = np.nanmax(normalized_matrix)
        
        # Usar colormap divergente se houver valores bem distintos
        if (vmax_actual - vmin_actual) > 0.7:
            cmap = 'RdYlGn'  # Verde = melhor, Vermelho = pior
        else:
            cmap = 'viridis'  # Para ranges menores
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Criar heatmap sem colorbar
        im = ax.imshow(normalized_matrix, cmap=cmap, aspect='auto', 
                      vmin=vmin_actual, vmax=vmax_actual)
        
        ax.set_xticks(np.arange(len(metrics_names)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax.set_yticklabels(models)
        ax.set_title(f'Heatmap de Métricas (Performance Normalizada) - {self.variable.capitalize()} - Estação {self.station_name}', 
                    fontweight='bold', fontsize=14)
        
        # Adicionar valores normalizados nas células
        for i in range(len(models)):
            for j in range(len(metrics_names)):
                # Mostrar valor normalizado com interpretação
                norm_val = normalized_matrix[i, j]
                orig_val = metrics_matrix[i, j]
                
                # Escolher cor do texto baseada no fundo
                text_color = 'white' if norm_val < 0.5 else 'black'
                
                # Mostrar apenas valor normalizado (0-1)
                if not np.isnan(orig_val):
                    text = ax.text(j, i, f'{norm_val:.2f}',
                                ha='center', va='center', color=text_color, 
                                fontsize=10, weight='bold')
                else:
                    text = ax.text(j, i, 'N/A',
                                ha='center', va='center', color=text_color, 
                                fontsize=10)
        
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = self._get_station_output_dir()
            filename = os.path.join(output_dir, f'metrics_heatmap_{self.variable}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_models_ranking(self, save_plot):
        """Ranking dos modelos"""
        models = list(self.results.keys())
        
        # Calcular score composto (quanto menor melhor)
        scores = []
        for model in models:
            m = self.results[model]['metrics']
            # Normalizar e ponderar métricas
            score = (
                m['RMSE'] * 0.3 +  # 30% peso
                m['MAE'] * 0.2 +   # 20% peso
                (1 - m['R2']) * 0.3 +  # 30% peso (invertido)
                abs(m['Bias']) * 0.2   # 20% peso
            )
            scores.append(score)
        
        # Ordenar modelos por score
        model_scores = list(zip(models, scores))
        model_scores.sort(key=lambda x: x[1])
        
        ranked_models = [m[0] for m in model_scores]
        ranked_scores = [m[1] for m in model_scores]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plotar ranking
        y_pos = np.arange(len(ranked_models))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(ranked_models)))
        
        bars = ax.barh(y_pos, ranked_scores, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{i+1}. {m}" for i, m in enumerate(ranked_models)])
        ax.set_xlabel('Score Composto (menor é melhor)', fontweight='bold', fontsize=12)
        ax.set_title(f'Ranking de Modelos - {self.variable.capitalize()} - Estação {self.station_name}', 
                    fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Adicionar valores
        for i, (bar, score) in enumerate(zip(bars, ranked_scores)):
            ax.text(bar.get_width() + max(ranked_scores) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = self._get_station_output_dir()
            filename = os.path.join(output_dir, f'models_ranking_{self.variable}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


    def plot_individual_temporal_analysis(self, model_name, save_plot=True):
        """Análise temporal individual para um modelo"""
        if model_name not in self.results:
            print(f"Modelo {model_name} não encontrado")
            return
        
        model_data = self.results[model_name]
        y_pred = model_data['predictions']
        
        if self.variable == 'precipitation':
            y_true = np.expm1(self.y_test)
        else:
            y_true = self.y_test
        
        # Criar DataFrame para análise
        temporal_df = pd.DataFrame({
            'date': self.test_dates.values,
            'observed': y_true.values if hasattr(y_true, 'values') else y_true,
            'predicted': y_pred,
            'residual': (y_true.values if hasattr(y_true, 'values') else y_true) - y_pred
        })
        
        temporal_df['month'] = pd.to_datetime(temporal_df['date']).dt.month
        temporal_df['year'] = pd.to_datetime(temporal_df['date']).dt.year
        temporal_df['day_of_year'] = pd.to_datetime(temporal_df['date']).dt.dayofyear
        
        # Gerar 4 gráficos separados de análise temporal
        self._plot_annual_cycle(model_name, temporal_df, save_plot)
        self._plot_bias_temporal(model_name, temporal_df, save_plot)
        self._plot_temporal_error_evolution(model_name, temporal_df, save_plot)
        self._plot_percentiles_analysis(model_name, temporal_df, save_plot)
    
    def _plot_annual_cycle(self, model_name, temporal_df, save_plot):
        """Ciclo anual médio"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        monthly_means = temporal_df.groupby('month').agg({
            'observed': ['mean', 'std'],
            'predicted': ['mean', 'std']
        })
        
        months = range(1, 13)
        month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        
        obs_means = monthly_means['observed']['mean'].values
        pred_means = monthly_means['predicted']['mean'].values
        obs_stds = monthly_means['observed']['std'].values
        pred_stds = monthly_means['predicted']['std'].values
        
        ax.plot(months, obs_means, 'o-', label='Observado', linewidth=2, markersize=8, color='blue')
        ax.plot(months, pred_means, 's-', label='Predito', linewidth=2, markersize=8, color='red')
        
        ax.fill_between(months, obs_means - obs_stds, obs_means + obs_stds,
                    alpha=0.2, color='blue')
        ax.fill_between(months, pred_means - pred_stds, pred_means + pred_stds,
                    alpha=0.2, color='red')
        
        ax.set_xlabel('Mês', fontweight='bold')
        ax.set_ylabel(f'{self.variable.capitalize()} ({self._get_unit()})', fontweight='bold')
        ax.set_title(f'Ciclo Anual Médio - {model_name} - Estação {self.station_name}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(months)
        ax.set_xticklabels(month_names)
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = self._get_station_output_dir()
            model_name_clean = model_name.lower().replace(' ', '_')
            filename = os.path.join(output_dir, f'annual_cycle_{self.variable}_{model_name_clean}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_bias_temporal(self, model_name, temporal_df, save_plot):
        """Padrão de erro anual"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        daily_errors = temporal_df.groupby('day_of_year')['residual'].agg(['mean', 'std'])
        
        ax.plot(daily_errors.index, daily_errors['mean'], color='darkred', linewidth=1.5)
        ax.fill_between(daily_errors.index,
                    daily_errors['mean'] - daily_errors['std'],
                    daily_errors['mean'] + daily_errors['std'],
                    alpha=0.3, color='red')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        ax.set_xlabel('Dia do Ano', fontweight='bold')
        ax.set_ylabel(f'Erro Médio ({self._get_unit()})', fontweight='bold')
        ax.set_title(f'Padrão de Erro Anual - {model_name} - Estação {self.station_name}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = self._get_station_output_dir()
            model_name_clean = model_name.lower().replace(' ', '_')
            filename = os.path.join(output_dir, f'bias_temporal_{self.variable}_{model_name_clean}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_temporal_error_evolution(self, model_name, temporal_df, save_plot):
        """Evolução temporal do erro"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        temporal_df['abs_error'] = np.abs(temporal_df['residual'])
        
        # Média móvel de 30 dias
        rolling_error = temporal_df.set_index('date')['abs_error'].rolling('30D').mean()
        
        ax.plot(temporal_df['date'], temporal_df['abs_error'], alpha=0.3, color='gray', label='Erro absoluto')
        ax.plot(rolling_error.index, rolling_error.values, color='red', linewidth=2, label='Média móvel 30d')
        
        ax.set_xlabel('Data', fontweight='bold')
        ax.set_ylabel(f'Erro Absoluto ({self._get_unit()})', fontweight='bold')
        ax.set_title(f'Evolução Temporal do Erro - {model_name} - Estação {self.station_name}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = self._get_station_output_dir()
            model_name_clean = model_name.lower().replace(' ', '_')
            filename = os.path.join(output_dir, f'temporal_error_{self.variable}_{model_name_clean}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_percentiles_analysis(self, model_name, temporal_df, save_plot):
        """Análise de percentis"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Percentis de observações e predições
        percentiles = [10, 25, 50, 75, 90]
        obs_percentiles = [np.percentile(temporal_df['observed'], p) for p in percentiles]
        pred_percentiles = [np.percentile(temporal_df['predicted'], p) for p in percentiles]
        
        x = np.arange(len(percentiles))
        width = 0.35
        
        ax.bar(x - width/2, obs_percentiles, width, label='Observado', alpha=0.8)
        ax.bar(x + width/2, pred_percentiles, width, label='Predito', alpha=0.8)
        
        ax.set_ylabel(f'{self.variable.capitalize()} ({self._get_unit()})', fontweight='bold')
        ax.set_title(f'Análise de Percentis - {model_name} - Estação {self.station_name}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'P{p}' for p in percentiles])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = self._get_station_output_dir()
            model_name_clean = model_name.lower().replace(' ', '_')
            filename = os.path.join(output_dir, f'percentiles_{self.variable}_{model_name_clean}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, top_n=20, model_name='RandomForest', save_plot=True):
        """Plota importância das features para modelos baseados em árvore"""
        if model_name not in self.results:
            available_models = [name for name in self.results.keys() 
                              if 'RandomForest' in name or 'ExtraTrees' in name or 'XGBoost' in name or 'GradientBoosting' in name]
            if available_models:
                model_name = available_models[0]
            else:
                print("Nenhum modelo baseado em árvore disponível")
                return

        model = self.results[model_name]['model']

        # Verificar se é modelo baseado em árvore
        try:
            if hasattr(model, 'named_steps') and 'model' in model.named_steps:
                base_model = model.named_steps['model']
            else:
                base_model = model
                
            if not hasattr(base_model, 'feature_importances_'):
                print(f"Modelo {model_name} não possui feature_importances_ (não é baseado em árvore)")
                return
                
            importances = base_model.feature_importances_
            # Garantir que importances é 1D
            if len(importances.shape) > 1:
                importances = importances.flatten()
        except Exception as e:
            print(f"Erro ao acessar feature_importances_ do modelo {model_name}: {e}")
            return
        
        # Obter nomes das features do modelo
        try:
            if hasattr(self.model, 'features') and self.model.features is not None:
                # Se features for um DataFrame, pegar as colunas
                if hasattr(self.model.features, 'columns'):
                    feature_names = list(self.model.features.columns)
                # Se features for uma lista/array
                elif hasattr(self.model.features, '__iter__') and not isinstance(self.model.features, str):
                    feature_names = list(self.model.features)
                else:
                    feature_names = [f'feature_{i}' for i in range(len(importances))]
            else:
                # Fallback: criar nomes genéricos
                feature_names = [f'feature_{i}' for i in range(len(importances))]
        except Exception as e:
            print(f"Erro ao obter nomes das features: {e}")
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Ajustar se há diferença no número de features
        if len(importances) != len(feature_names):
            min_len = min(len(importances), len(feature_names))
            importances = importances[:min_len]
            feature_names = feature_names[:min_len]
        
        # Criar DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plotar
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(feature_importance)), feature_importance['importance'], 
                       color='steelblue', alpha=0.8)
        
        # Personalizar gráfico
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Importância', fontweight='bold', fontsize=12)
        plt.title(f'Top {top_n} Features Mais Importantes - {model_name}\n{self.variable.capitalize()} - Estação {self.station_name}', 
                 fontweight='bold', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        # Adicionar valores nas barras
        for i, (bar, value) in enumerate(zip(bars, feature_importance['importance'])):
            plt.text(value + value*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = self._get_station_output_dir()
            filename = os.path.join(output_dir, f'feature_importance_{self.variable}_{model_name.lower().replace(" ", "_")}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo: {filename}")
        plt.close()
        
        return feature_importance
            


    def _plot_scatter_density(self, ax, y_true, y_pred, metrics, model_name):
        """
        Cria um scatter plot melhorado com densidade colorida baseado no exemplo fornecido
        """
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.colors as mcolors
        from scipy.stats import gaussian_kde
        
        # Converter para arrays numpy se necessário
        if hasattr(y_true, 'values'):
            y_true_vals = y_true.values
        else:
            y_true_vals = np.array(y_true)
        
        y_pred_vals = np.array(y_pred)
        
        # Remover valores NaN
        valid_mask = ~(np.isnan(y_true_vals) | np.isnan(y_pred_vals))
        y_true_clean = y_true_vals[valid_mask]
        y_pred_clean = y_pred_vals[valid_mask]
        
        # Calcular densidade dos pontos usando KDE
        try:
            # Stack the data
            xy = np.vstack([y_true_clean, y_pred_clean])
            z = gaussian_kde(xy)(xy)
            
            # Sort the points by density, so that the densest points are plotted last
            idx = z.argsort()
            x_sorted, y_sorted, z_sorted = y_true_clean[idx], y_pred_clean[idx], z[idx]
            
            # Criar scatter plot com densidade colorida
            scatter = ax.scatter(x_sorted, y_sorted, c=z_sorted, s=20, alpha=0.7, 
                               cmap='viridis', edgecolors='none', rasterized=True)
            
            # Adicionar colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label('Densidade', rotation=270, labelpad=20, fontsize=10)
            
        except Exception as e:
            # Fallback para scatter plot simples se KDE falhar
            print(f"Aviso: Usando scatter plot simples para {model_name}. Erro KDE: {e}")
            ax.scatter(y_true_clean, y_pred_clean, alpha=0.6, s=20, c='steelblue', edgecolors='none')
        
        # Linha 1:1
        min_val = min(y_true_clean.min(), y_pred_clean.min())
        max_val = max(y_true_clean.max(), y_pred_clean.max())
        
        # Adicionar margem de 5%
        margin = (max_val - min_val) * 0.05
        min_val -= margin
        max_val += margin
        
        ax.plot([min_val, max_val], [min_val, max_val], 'r-', lw=2, label='Linha 1:1', alpha=0.8)
        
        # Configurar limites dos eixos
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
        # Estatísticas - usar SEMPRE os valores das métricas armazenadas para consistência
        r2 = metrics['R2']
        rmse = metrics['RMSE']
        bias = metrics['Bias']
        
        # SEMPRE usar correlação das métricas armazenadas para consistência com o relatório
        correlation = metrics['Correlation']
        
        # Texto das estatísticas (formato similar ao exemplo)
        stats_text = f'r² = {r2:.2f}\nCorrelação = {correlation:.2f}\nRMSE = {rmse:.3f}\nbias = {bias:.3f}'
        
        # Posicionar texto no canto superior esquerdo
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, 
                         edgecolor='gray', linewidth=0.5))
        
        # Labels e título
        unit = self._get_unit()
        ax.set_xlabel(f'Observado ({unit})', fontweight='bold', fontsize=12)
        ax.set_ylabel(f'Predito ({unit})', fontweight='bold', fontsize=12)
        ax.set_title(f'Observado vs Predito - {model_name} - Estação {self.station_name}', fontweight='bold', fontsize=13)
        
        # Grid sutil
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Configurar aspecto igual para manter proporção
        ax.set_aspect('equal', adjustable='box')
        
        # Melhorar aparência dos ticks
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        return ax

    def plot_enhanced_scatter(self, model_name=None, save_plot=True, figsize=(8, 8)):
        """
        Cria um gráfico de dispersão individual melhorado, similar ao exemplo fornecido
        """
        if not self.results:
            print("Nenhum resultado disponível")
            return
        
        # Se não especificar modelo, usar o primeiro disponível
        if model_name is None:
            model_name = list(self.results.keys())[0]
        
        if model_name not in self.results:
            print(f"Modelo {model_name} não encontrado. Modelos disponíveis: {list(self.results.keys())}")
            return
        
        # Obter dados
        model_data = self.results[model_name]
        y_pred = model_data['predictions']
        metrics = model_data['metrics']
        
        if self.variable == 'precipitation':
            y_true = np.expm1(self.y_test)
        else:
            y_true = self.y_test
        
        # Criar figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Usar a função melhorada de scatter
        self._plot_scatter_density(ax, y_true, y_pred, metrics, model_name)
        
        # Ajustar layout
        plt.tight_layout()
        
        if save_plot:
            output_dir = self._get_station_output_dir()
            model_name_clean = model_name.lower().replace(' ', '_')
            filename = os.path.join(output_dir, 
                                f'scatter_enhanced_{self.variable}_{model_name_clean}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Gráfico de dispersão melhorado salvo: {filename}")
        
        return fig, ax

    def _get_unit(self):
        """Retorna a unidade de medida"""
        return '°C' if self.variable == 'temperature' else 'mm'