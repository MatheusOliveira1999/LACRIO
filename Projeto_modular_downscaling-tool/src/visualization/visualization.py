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

    # --- CORREÇÃO 1: Adicionado um método principal para gerar todos os gráficos ---
    def generate_all_plots(self, save_plots=True):
        """
        Ponto de entrada principal para gerar todas as visualizações.
        Chama os métodos de plotagem específicos.
        """
        if not self.results:
            print("Nenhum resultado para plotar")
            return

        print("\n📊 Gerando todas as visualizações...")
        try:
            self.plot_models_comparison(save_plot=save_plots)
            self.plot_all_models_results(save_plots=save_plots)
            self.plot_temporal_analysis_all_models(save_plots=save_plots)
            
            # Gerar feature importance apenas para modelos baseados em árvore
            tree_based_models = [name for name in self.results.keys() 
                               if 'RandomForest' in name or 'ExtraTrees' in name or 'XGBoost' in name or 'GradientBoosting' in name]
            
            if tree_based_models:
                print("\n🔍 Gerando feature importance para modelos baseados em árvore...")
                for model_name in tree_based_models:
                    try:
                        self.plot_feature_importance(model_name=model_name, save_plot=save_plots)
                    except Exception as e:
                        print(f"  ❌ Erro ao gerar feature importance para {model_name}: {e}")
            else:
                print("\n⚠️ Nenhum modelo baseado em árvore encontrado para feature importance")
            
            print("✅ Todas as visualizações geradas com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao gerar visualizações: {e}")

    def plot_all_models_results(self, save_plots=True):
        """Gera visualizações para TODOS os modelos treinados"""
        if not self.results:
            print("Nenhum resultado para plotar")
            return
        
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Para cada modelo treinado
        for model_name, model_data in self.results.items():
            print(f"\nGerando gráficos para {model_name}...")
            
            # Criar figura com subplots para este modelo
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Resultados do Downscaling - {model_name} - {self.variable.capitalize()}', 
                        fontsize=16, fontweight='bold')
            
            y_pred = model_data['predictions']
            metrics = model_data['metrics']
            
            if self.variable == 'precipitation':
                y_true = np.expm1(self.y_test)
            else:
                y_true = self.y_test
            
            # 1. Observado vs Predito
            ax = axes[0, 0]
            scatter = ax.scatter(y_true, y_pred, alpha=0.6, s=20, c='blue', edgecolors='none')
            
            # Linha 1:1
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
            
            # Estatísticas
            r2 = metrics['R2']
            rmse = metrics['RMSE']
            mae = metrics['MAE']
            bias = metrics['Bias']
            
            stats_text = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nBias = {bias:.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel(f'Observado ({self._get_unit()})', fontweight='bold')
            ax.set_ylabel(f'Predito ({self._get_unit()})', fontweight='bold')
            ax.set_title('Observado vs Predito', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 2. Série temporal (últimos 90 dias)
            ax = axes[0, 1]
            n_points = min(90, len(self.test_dates))
            indices = np.arange(len(self.test_dates) - n_points, len(self.test_dates))
            
            dates_subset = self.test_dates.iloc[indices]
            y_true_subset = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
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
            ax.set_title('Distribuição dos Resíduos', fontweight='bold')
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
            
            # 4. Q-Q Plot
            ax = axes[1, 1]
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.set_title('Q-Q Plot dos Resíduos', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                # Caminho absoluto para static/img na raiz do projeto
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                output_dir = os.path.join(project_root, 'static', 'img')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Nome único para cada modelo
                model_name_clean = model_name.lower().replace(' ', '_')
                filename = os.path.join(output_dir, 
                                    f'resultados_{self.variable}_{model_name_clean}.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  ✓ Gráfico salvo: {filename}")
            
            plt.close()  # Fechar figura para economizar memória


    def plot_models_comparison(self, save_plot=True):
        """Gera gráfico comparativo de todos os modelos"""
        if not self.results:
            print("Nenhum resultado para plotar")
            return
        
        # Configurar figura
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Comparação de Modelos - {self.variable.capitalize()}', 
                    fontsize=16, fontweight='bold')
        
        # Preparar dados
        models = list(self.results.keys())
        metrics_names = ['RMSE', 'MAE', 'R2', 'Correlation', 'Bias', 'Skill_Score']
        
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
        
        # Adicionar valores
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
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
        
        # 5. Heatmap de todas as métricas
        ax = axes[1, 1]
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
            if metric in ['RMSE', 'MAE', 'Bias']:  # Métricas onde menor é melhor
                normalized_matrix[:, j] = 1 - (col - col.min()) / (col.max() - col.min() + 1e-10)
            else:  # Métricas onde maior é melhor
                normalized_matrix[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-10)
        
        im = ax.imshow(normalized_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(metrics_names)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax.set_yticklabels(models)
        ax.set_title('Heatmap de Métricas (Normalizado)', fontweight='bold')
        
        # Adicionar valores
        for i in range(len(models)):
            for j in range(len(metrics_names)):
                text = ax.text(j, i, f'{metrics_matrix[i, j]:.3f}',
                            ha='center', va='center', color='black', fontsize=8)
        
        # 6. Ranking dos modelos
        ax = axes[1, 2]
        
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
        
        # Plotar ranking
        y_pos = np.arange(len(ranked_models))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(ranked_models)))
        
        bars = ax.barh(y_pos, ranked_scores, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{i+1}. {m}" for i, m in enumerate(ranked_models)])
        ax.set_xlabel('Score Composto (menor é melhor)', fontweight='bold')
        ax.set_title('Ranking de Modelos', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Adicionar valores
        for i, (bar, score) in enumerate(zip(bars, ranked_scores)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_plot:
            # Caminho absoluto para static/img na raiz do projeto
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(project_root, 'static', 'img')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = os.path.join(output_dir, f'comparacao_modelos_{self.variable}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico de comparação salvo: {filename}")

            plt.close()


    def plot_temporal_analysis_all_models(self, save_plots=True):
        """Análise temporal para todos os modelos"""
        if not self.results:
            print("Nenhum resultado disponível")
            return
        
        # Para cada modelo
        for model_name, model_data in self.results.items():
            print(f"\nGerando análise temporal para {model_name}...")
            
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
            
            # Criar figura
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Análise Temporal - {model_name} - {self.variable.capitalize()}',
                        fontsize=16, fontweight='bold')
            
            # 1. Ciclo anual médio
            ax = axes[0, 0]
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
            ax.set_title('Ciclo Anual Médio', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(months)
            ax.set_xticklabels(month_names)
            
            # 2. Erro por dia do ano
            ax = axes[0, 1]
            daily_errors = temporal_df.groupby('day_of_year')['residual'].agg(['mean', 'std'])
            
            ax.plot(daily_errors.index, daily_errors['mean'], color='darkred', linewidth=1.5)
            ax.fill_between(daily_errors.index,
                        daily_errors['mean'] - daily_errors['std'],
                        daily_errors['mean'] + daily_errors['std'],
                        alpha=0.3, color='red')
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
            
            ax.set_xlabel('Dia do Ano', fontweight='bold')
            ax.set_ylabel(f'Erro Médio ({self._get_unit()})', fontweight='bold')
            ax.set_title('Padrão de Erro Anual', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # 3. Série temporal de erros absolutos
            ax = axes[1, 0]
            temporal_df['abs_error'] = np.abs(temporal_df['residual'])
            
            # Média móvel de 30 dias
            rolling_error = temporal_df.set_index('date')['abs_error'].rolling('30D').mean()
            
            ax.plot(temporal_df['date'], temporal_df['abs_error'], alpha=0.3, color='gray', label='Erro absoluto')
            ax.plot(rolling_error.index, rolling_error.values, color='red', linewidth=2, label='Média móvel 30d')
            
            ax.set_xlabel('Data', fontweight='bold')
            ax.set_ylabel(f'Erro Absoluto ({self._get_unit()})', fontweight='bold')
            ax.set_title('Evolução Temporal do Erro', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # 4. Análise de extremos
            ax = axes[1, 1]
            
            # Percentis de observações e predições
            percentiles = [10, 25, 50, 75, 90]
            obs_percentiles = [np.percentile(temporal_df['observed'], p) for p in percentiles]
            pred_percentiles = [np.percentile(temporal_df['predicted'], p) for p in percentiles]
            
            x = np.arange(len(percentiles))
            width = 0.35
            
            ax.bar(x - width/2, obs_percentiles, width, label='Observado', alpha=0.8)
            ax.bar(x + width/2, pred_percentiles, width, label='Predito', alpha=0.8)
            
            ax.set_ylabel(f'{self.variable.capitalize()} ({self._get_unit()})', fontweight='bold')
            ax.set_title('Análise de Percentis', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'P{p}' for p in percentiles])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                # Caminho absoluto para static/img na raiz do projeto
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                output_dir = os.path.join(project_root, 'static', 'img')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                model_name_clean = model_name.lower().replace(' ', '_')
                filename = os.path.join(output_dir, 
                                    f'analise_temporal_{self.variable}_{model_name_clean}.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  ✓ Análise temporal salva: {filename}")
            
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
        if not hasattr(model.named_steps['model'], 'feature_importances_'):
            print(f"Modelo {model_name} não possui feature_importances_ (não é baseado em árvore)")
            return
        
        importances = model.named_steps['model'].feature_importances_
        
        # Obter nomes das features do modelo
        if hasattr(self.model, 'features') and self.model.features is not None:
            feature_names = self.model.features
        else:
            # Fallback: criar nomes genéricos
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
            # Caminho absoluto para static/img na raiz do projeto
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(project_root, 'static', 'img')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = os.path.join(output_dir, f'feature_importance_{self.variable}_{model_name.lower().replace(" ", "_")}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo: {filename}")
        plt.close()
        
        return feature_importance
            


    def _get_unit(self):
        """Retorna a unidade de medida"""
        return '°C' if self.variable == 'temperature' else 'mm'