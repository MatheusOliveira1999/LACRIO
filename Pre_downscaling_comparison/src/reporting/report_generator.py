"""
report_generator.py
Geração de relatórios detalhados de comparação
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import markdown
from jinja2 import Template


class ReportGenerator:
    """Geração de relatórios detalhados"""
    
    def __init__(self, output_dir: str = "./results/reports"):
        self.output_dir = output_dir
        self.templates_dir = "./src/reporting/templates"
        
        # Criar diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_full_report(self, analysis_results: Dict, 
                           metadata: Dict = None,
                           output_filename: str = None) -> str:
        """
        Gera relatório completo em Markdown
        
        Parameters:
        -----------
        analysis_results : dict
            Resultados completos da análise
        metadata : dict, optional
            Metadados da análise
        output_filename : str, optional
            Nome do arquivo de saída
            
        Returns:
        --------
        str
            Caminho do arquivo gerado
        """
        print("📋 Gerando relatório completo...")
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"relatorio_comparacao_{timestamp}.md"
            
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Preparar dados para o template
        report_data = self._prepare_report_data(analysis_results, metadata)
        
        # Gerar conteúdo do relatório
        report_content = self._generate_markdown_content(report_data)
        
        # Salvar arquivo
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        print(f"✅ Relatório salvo em: {output_path}")
        return output_path
        
    def generate_summary_report(self, analysis_results: Dict, 
                              output_filename: str = None) -> str:
        """
        Gera resumo executivo
        
        Parameters:
        -----------
        analysis_results : dict
            Resultados da análise
        output_filename : str, optional
            Nome do arquivo de saída
            
        Returns:
        --------
        str
            Caminho do arquivo gerado
        """
        print("📊 Gerando resumo executivo...")
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"resumo_executivo_{timestamp}.md"
            
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Gerar resumo
        summary_content = self._generate_summary_content(analysis_results)
        
        # Salvar arquivo
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
            
        print(f"✅ Resumo salvo em: {output_path}")
        return output_path
        
    def generate_html_report(self, markdown_content: str, 
                           output_filename: str = None) -> str:
        """
        Converte relatório Markdown para HTML
        
        Parameters:
        -----------
        markdown_content : str
            Conteúdo em Markdown
        output_filename : str, optional
            Nome do arquivo HTML
            
        Returns:
        --------
        str
            Caminho do arquivo HTML gerado
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"relatorio_comparacao_{timestamp}.html"
            
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Converter Markdown para HTML
        html_content = markdown.markdown(
            markdown_content,
            extensions=['tables', 'toc', 'codehilite']
        )
        
        # Template HTML básico
        html_template = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Comparação Pré-Downscaling</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 { color: #2c3e50; }
        h1 { border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { border-bottom: 1px solid #ecf0f1; padding-bottom: 5px; }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0;
            background-color: white;
        }
        th, td { 
            border: 1px solid #ddd; 
            padding: 12px; 
            text-align: left; 
        }
        th { 
            background-color: #3498db; 
            color: white; 
            font-weight: bold;
        }
        tr:nth-child(even) { background-color: #f8f9fa; }
        .metric-good { color: #27ae60; font-weight: bold; }
        .metric-warning { color: #f39c12; font-weight: bold; }
        .metric-bad { color: #e74c3c; font-weight: bold; }
        .summary-box {
            background-color: #ecf0f1;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }
        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
        }
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        {{ content }}
    </div>
</body>
</html>
"""
        
        # Renderizar template
        template = Template(html_template)
        final_html = template.render(content=html_content)
        
        # Salvar arquivo
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
            
        print(f"✅ Relatório HTML salvo em: {output_path}")
        return output_path
        
    def _prepare_report_data(self, analysis_results: Dict, metadata: Dict = None) -> Dict:
        """Prepara dados para o relatório"""
        
        if metadata is None:
            metadata = {}
            
        # Data atual
        current_date = datetime.now().strftime("%d/%m/%Y %H:%M")
        
        # Extrair informações básicas
        comparison_results = analysis_results.get('comparison_results', {})
        seasonal_results = analysis_results.get('seasonal_results', {})
        extreme_results = analysis_results.get('extreme_results', {})
        temporal_results = analysis_results.get('temporal_results', {})
        
        # Variáveis analisadas
        variables = list(comparison_results.keys())
        
        # Estatísticas gerais
        general_stats = {}
        for var in variables:
            if var in comparison_results:
                metrics = comparison_results[var].get('comparison_metrics', {})
                general_stats[var] = {
                    'correlation': metrics.get('pearson_correlation', 0),
                    'r_squared': metrics.get('r_squared', 0),
                    'bias': metrics.get('bias', 0),
                    'rmse': metrics.get('rmse', 0),
                    'mae': metrics.get('mae', 0)
                }
        
        return {
            'metadata': {
                'generation_date': current_date,
                'station_name': metadata.get('station_name', 'Não informado'),
                'latitude': metadata.get('latitude', 'Não informado'),
                'longitude': metadata.get('longitude', 'Não informado'),
                'period_start': metadata.get('period_start', 'Não informado'),
                'period_end': metadata.get('period_end', 'Não informado'),
                'total_records': metadata.get('total_records', 'Não informado')
            },
            'variables': variables,
            'general_stats': general_stats,
            'comparison_results': comparison_results,
            'seasonal_results': seasonal_results,
            'extreme_results': extreme_results,
            'temporal_results': temporal_results
        }
        
    def _generate_markdown_content(self, report_data: Dict) -> str:
        """Gera conteúdo completo em Markdown"""
        
        md_content = f"""# Relatório de Comparação Pré-Downscaling

**Data de Geração:** {report_data['metadata']['generation_date']}

---

## 1. Resumo Executivo

### 🌍 **Informações Gerais**
- **Estação:** {report_data['metadata']['station_name']}
- **Localização:** {report_data['metadata']['latitude']}, {report_data['metadata']['longitude']}
- **Período:** {report_data['metadata']['period_start']} a {report_data['metadata']['period_end']}
- **Total de Registros:** {report_data['metadata']['total_records']}

### 📊 **Variáveis Analisadas**
{', '.join([var.title() for var in report_data['variables']])}

### 🎯 **Principais Achados**

"""
        
        # Resumo das métricas principais
        if report_data['general_stats']:
            md_content += "| Variável | Correlação | R² | Bias | RMSE |\n"
            md_content += "|----------|------------|----|----- |------|\n"
            
            for var, stats in report_data['general_stats'].items():
                corr = stats.get('correlation', 0)
                r2 = stats.get('r_squared', 0)
                bias = stats.get('bias', 0)
                rmse = stats.get('rmse', 0)
                
                md_content += f"| {var.title()} | {corr:.3f} | {r2:.3f} | {bias:.3f} | {rmse:.3f} |\n"
        
        md_content += "\n---\n\n"
        
        # Análise detalhada por variável
        md_content += "## 2. Análise Detalhada por Variável\n\n"
        
        for var in report_data['variables']:
            md_content += f"### 2.{report_data['variables'].index(var) + 1} {var.title()}\n\n"
            
            # Estatísticas básicas
            if var in report_data['comparison_results']:
                var_results = report_data['comparison_results'][var]
                
                md_content += "#### Estatísticas Básicas\n\n"
                
                era5_stats = var_results.get('era5_stats', {})
                station_stats = var_results.get('station_stats', {})
                
                md_content += "| Estatística | ERA5 | Estação |\n"
                md_content += "|-------------|------|----------|\n"
                md_content += f"| Média | {era5_stats.get('mean', 0):.3f} | {station_stats.get('mean', 0):.3f} |\n"
                md_content += f"| Desvio Padrão | {era5_stats.get('std', 0):.3f} | {station_stats.get('std', 0):.3f} |\n"
                md_content += f"| Mínimo | {era5_stats.get('min', 0):.3f} | {station_stats.get('min', 0):.3f} |\n"
                md_content += f"| Máximo | {era5_stats.get('max', 0):.3f} | {station_stats.get('max', 0):.3f} |\n"
                md_content += f"| Mediana | {era5_stats.get('median', 0):.3f} | {station_stats.get('median', 0):.3f} |\n\n"
                
                # Métricas de comparação
                metrics = var_results.get('comparison_metrics', {})
                
                md_content += "#### Métricas de Comparação\n\n"
                md_content += f"- **Correlação de Pearson:** {metrics.get('pearson_correlation', 0):.3f} (p-value: {metrics.get('pearson_p_value', 0):.3e})\n"
                md_content += f"- **Correlação de Spearman:** {metrics.get('spearman_correlation', 0):.3f}\n"
                md_content += f"- **R²:** {metrics.get('r_squared', 0):.3f}\n"
                md_content += f"- **Bias:** {metrics.get('bias', 0):.3f}\n"
                md_content += f"- **RMSE:** {metrics.get('rmse', 0):.3f}\n"
                md_content += f"- **MAE:** {metrics.get('mae', 0):.3f}\n"
                md_content += f"- **Nash-Sutcliffe:** {metrics.get('nse', 0):.3f}\n\n"
                
                # Análise de bias
                bias_analysis = var_results.get('bias_analysis', {})
                if bias_analysis:
                    md_content += "#### Análise de Bias\n\n"
                    md_content += f"- **Bias Sistemático:** {bias_analysis.get('systematic_bias', 0):.3f}\n"
                    md_content += f"- **Bias Aleatório:** {bias_analysis.get('random_bias', 0):.3f}\n"
                    md_content += f"- **Desvio Padrão do Bias:** {bias_analysis.get('bias_std', 0):.3f}\n"
                    
                    bias_range = bias_analysis.get('bias_range', [0, 0])
                    md_content += f"- **Amplitude do Bias:** [{bias_range[0]:.3f}, {bias_range[1]:.3f}]\n\n"
                
                # Análise de distribuições
                dist_analysis = var_results.get('distribution_analysis', {})
                if dist_analysis:
                    md_content += "#### Análise de Distribuições\n\n"
                    
                    era5_norm = dist_analysis.get('era5_normality', {})
                    station_norm = dist_analysis.get('station_normality', {})
                    ks_test = dist_analysis.get('ks_test', {})
                    
                    md_content += f"- **ERA5 Normal:** {'Sim' if era5_norm.get('is_normal', False) else 'Não'} (p = {era5_norm.get('shapiro_p', 0):.3f})\n"
                    md_content += f"- **Estação Normal:** {'Sim' if station_norm.get('is_normal', False) else 'Não'} (p = {station_norm.get('shapiro_p', 0):.3f})\n"
                    md_content += f"- **Distribuições Similares:** {'Sim' if ks_test.get('distributions_similar', False) else 'Não'} (KS p = {ks_test.get('p_value', 0):.3f})\n\n"
            
            # Análise sazonal
            if var in report_data.get('seasonal_results', {}):
                seasonal = report_data['seasonal_results'][var]
                md_content += "#### Análise Sazonal\n\n"
                
                seasons = ['DJF', 'MAM', 'JJA', 'SON']
                season_names = ['Verão', 'Outono', 'Inverno', 'Primavera']
                
                md_content += "| Estação | Correlação | Bias | RMSE | N |\n"
                md_content += "|---------|------------|------|------|---|\n"
                
                for season, name in zip(seasons, season_names):
                    if season in seasonal:
                        s_data = seasonal[season]
                        corr = s_data.get('pearson_correlation', 0)
                        bias = s_data.get('bias', 0)
                        rmse = s_data.get('rmse', 0)
                        n = s_data.get('n_points', 0)
                        md_content += f"| {name} | {corr:.3f} | {bias:.3f} | {rmse:.3f} | {n} |\n"
                
                md_content += "\n"
            
            # Eventos extremos
            if var in report_data.get('extreme_results', {}):
                extreme = report_data['extreme_results'][var]
                md_content += "#### Análise de Eventos Extremos\n\n"
                
                if 'p95' in extreme:
                    p95_data = extreme['p95']
                    md_content += f"**Percentil 95:**\n"
                    md_content += f"- Threshold: {p95_data.get('high_threshold', 0):.3f}\n"
                    md_content += f"- Taxa de Detecção: {p95_data.get('high_hit_rate', 0):.3f}\n"
                    md_content += f"- Taxa de Falso Alarme: {p95_data.get('high_false_alarm_rate', 0):.3f}\n\n"
            
            md_content += "---\n\n"
        
        # Análise temporal
        if report_data.get('temporal_results'):
            md_content += "## 3. Análise Temporal\n\n"
            
            temporal = report_data['temporal_results']
            
            # Análise de dados faltantes
            if 'missing_analysis' in temporal:
                missing = temporal['missing_analysis']
                md_content += "### 3.1 Disponibilidade de Dados\n\n"
                
                if 'data_availability' in missing:
                    avail = missing['data_availability']['common_period']
                    md_content += f"- **Período Comum:** {avail.get('start_date', '')} a {avail.get('end_date', '')}\n"
                    md_content += f"- **Dias Totais:** {avail.get('total_days', 0)}\n"
                    md_content += f"- **Dias com Dados:** {avail.get('common_days', 0)}\n"
                    md_content += f"- **Disponibilidade:** {avail.get('availability_percent', 0):.1f}%\n\n"
        
        # Conclusões e recomendações
        md_content += "## 4. Conclusões e Recomendações\n\n"
        md_content += self._generate_conclusions(report_data)
        
        # Interpretação das métricas
        md_content += "\n## 5. Interpretação das Métricas\n\n"
        md_content += self._generate_metric_interpretation()
        
        return md_content
        
    def _generate_summary_content(self, analysis_results: Dict) -> str:
        """Gera conteúdo do resumo executivo"""
        
        current_date = datetime.now().strftime("%d/%m/%Y %H:%M")
        comparison_results = analysis_results.get('comparison_results', {})
        
        md_content = f"""# Resumo Executivo - Comparação Pré-Downscaling

**Data:** {current_date}

## 🎯 Principais Resultados

"""
        
        if comparison_results:
            # Encontrar melhor e pior variável
            best_var = None
            worst_var = None
            best_corr = -1
            worst_corr = 2
            
            for var, results in comparison_results.items():
                metrics = results.get('comparison_metrics', {})
                corr = metrics.get('pearson_correlation', 0)
                
                if corr > best_corr:
                    best_corr = corr
                    best_var = var
                    
                if corr < worst_corr:
                    worst_corr = corr
                    worst_var = var
            
            if best_var:
                best_metrics = comparison_results[best_var].get('comparison_metrics', {})
                md_content += f"### ✅ **Melhor Performance: {best_var.title()}**\n"
                md_content += f"- Correlação: {best_metrics.get('pearson_correlation', 0):.3f}\n"
                md_content += f"- R²: {best_metrics.get('r_squared', 0):.3f}\n"
                md_content += f"- RMSE: {best_metrics.get('rmse', 0):.3f}\n"
                md_content += f"- Bias: {best_metrics.get('bias', 0):.3f}\n\n"
            
            if worst_var and worst_var != best_var:
                worst_metrics = comparison_results[worst_var].get('comparison_metrics', {})
                md_content += f"### ⚠️ **Menor Performance: {worst_var.title()}**\n"
                md_content += f"- Correlação: {worst_metrics.get('pearson_correlation', 0):.3f}\n"
                md_content += f"- R²: {worst_metrics.get('r_squared', 0):.3f}\n"
                md_content += f"- RMSE: {worst_metrics.get('rmse', 0):.3f}\n"
                md_content += f"- Bias: {worst_metrics.get('bias', 0):.3f}\n\n"
        
        # Recomendações gerais
        md_content += "## 💡 Recomendações para Downscaling\n\n"
        md_content += self._generate_quick_recommendations(comparison_results)
        
        return md_content
        
    def _generate_conclusions(self, report_data: Dict) -> str:
        """Gera conclusões automáticas baseadas nos resultados"""
        
        conclusions = []
        general_stats = report_data.get('general_stats', {})
        
        # Analisar correlações
        correlations = [stats.get('correlation', 0) for stats in general_stats.values()]
        if correlations:
            avg_corr = sum(correlations) / len(correlations)
            
            if avg_corr > 0.8:
                conclusions.append("🟢 **Correlações Excelentes**: ERA5 apresenta correlações muito fortes com as observações (r > 0.8).")
            elif avg_corr > 0.6:
                conclusions.append("🟡 **Correlações Boas**: ERA5 apresenta correlações moderadas a fortes com as observações (r > 0.6).")
            else:
                conclusions.append("🔴 **Correlações Baixas**: ERA5 apresenta correlações fracas com as observações (r < 0.6). Considere pré-processamento adicional.")
        
        # Analisar bias
        biases = [abs(stats.get('bias', 0)) for stats in general_stats.values()]
        if biases:
            avg_bias = sum(biases) / len(biases)
            
            if avg_bias < 0.1:
                conclusions.append("🟢 **Bias Baixo**: ERA5 apresenta bias sistemático mínimo.")
            elif avg_bias < 0.5:
                conclusions.append("🟡 **Bias Moderado**: ERA5 apresenta bias sistemático moderado. Considere correção de bias.")
            else:
                conclusions.append("🔴 **Bias Alto**: ERA5 apresenta bias sistemático significativo. Correção de bias é recomendada.")
        
        # Recomendações específicas
        conclusions.append("\\n### Recomendações Específicas:")
        
        for var, stats in general_stats.items():
            corr = stats.get('correlation', 0)
            bias = abs(stats.get('bias', 0))
            
            if corr > 0.7 and bias < 0.2:
                conclusions.append(f"- **{var.title()}**: Excelente para downscaling direto.")
            elif corr > 0.5:
                conclusions.append(f"- **{var.title()}**: Adequado para downscaling com pré-processamento.")
            else:
                conclusions.append(f"- **{var.title()}**: Requer análise adicional ou métodos alternativos.")
        
        return "\\n".join(conclusions)
        
    def _generate_quick_recommendations(self, comparison_results: Dict) -> str:
        """Gera recomendações rápidas para o resumo"""
        
        recommendations = []
        
        if not comparison_results:
            return "Dados insuficientes para recomendações."
        
        # Contar variáveis por qualidade
        good_vars = []
        moderate_vars = []
        poor_vars = []
        
        for var, results in comparison_results.items():
            metrics = results.get('comparison_metrics', {})
            corr = metrics.get('pearson_correlation', 0)
            
            if corr > 0.7:
                good_vars.append(var)
            elif corr > 0.4:
                moderate_vars.append(var)
            else:
                poor_vars.append(var)
        
        if good_vars:
            recommendations.append(f"✅ **Proceder com downscaling direto:** {', '.join([v.title() for v in good_vars])}")
        
        if moderate_vars:
            recommendations.append(f"🔄 **Aplicar pré-processamento:** {', '.join([v.title() for v in moderate_vars])}")
        
        if poor_vars:
            recommendations.append(f"⚠️ **Revisar métodos ou dados:** {', '.join([v.title() for v in poor_vars])}")
        
        recommendations.append("\\n📊 **Próximos Passos:**")
        recommendations.append("1. Executar o sistema de downscaling com os dados validados")
        recommendations.append("2. Aplicar correções de bias conforme necessário")
        recommendations.append("3. Validar resultados do downscaling com dados independentes")
        
        return "\\n".join(recommendations)
        
    def _generate_metric_interpretation(self) -> str:
        """Gera guia de interpretação das métricas"""
        
        interpretation = """
### Correlação (r)
- **> 0.8**: Correlação muito forte - Excelente
- **0.6 - 0.8**: Correlação forte - Boa
- **0.4 - 0.6**: Correlação moderada - Aceitável
- **< 0.4**: Correlação fraca - Problemática

### Coeficiente de Determinação (R²)
- **> 0.64**: Mais de 64% da variância explicada - Excelente
- **0.36 - 0.64**: 36-64% da variância explicada - Bom
- **0.16 - 0.36**: 16-36% da variância explicada - Moderado
- **< 0.16**: Menos de 16% da variância explicada - Pobre

### Bias
- **Próximo de 0**: Sem tendência sistemática - Ideal
- **Positivo**: ERA5 superestima em relação à estação
- **Negativo**: ERA5 subestima em relação à estação

### RMSE (Root Mean Square Error)
- Indica a magnitude típica dos erros
- Menor valor = melhor performance
- Mesma unidade da variável analisada

### Nash-Sutcliffe Efficiency (NSE)
- **> 0.75**: Muito bom
- **0.65 - 0.75**: Bom
- **0.50 - 0.65**: Satisfatório
- **< 0.50**: Insatisfatório
"""
        
        return interpretation
        
    def create_comparison_table(self, statistics: Dict) -> str:
        """
        Cria tabela comparativa de estatísticas
        
        Parameters:
        -----------
        statistics : dict
            Estatísticas para comparação
            
        Returns:
        --------
        str
            Tabela em formato Markdown
        """
        if not statistics:
            return "Sem dados para comparação."
        
        # Cabeçalho da tabela
        table = "| Variável | Correlação | R² | Bias | RMSE | MAE | NSE |\n"
        table += "|----------|------------|----|----- |------|-----|-----|\n"
        
        # Linhas da tabela
        for var, stats in statistics.items():
            metrics = stats.get('comparison_metrics', {})
            
            corr = metrics.get('pearson_correlation', 0)
            r2 = metrics.get('r_squared', 0)
            bias = metrics.get('bias', 0)
            rmse = metrics.get('rmse', 0)
            mae = metrics.get('mae', 0)
            nse = metrics.get('nse', 0)
            
            table += f"| {var.title()} | {corr:.3f} | {r2:.3f} | {bias:.3f} | {rmse:.3f} | {mae:.3f} | {nse:.3f} |\n"
        
        return table
        
    def export_results_json(self, analysis_results: Dict, 
                           output_filename: str = None) -> str:
        """
        Exporta resultados em formato JSON
        
        Parameters:
        -----------
        analysis_results : dict
            Resultados da análise
        output_filename : str, optional
            Nome do arquivo de saída
            
        Returns:
        --------
        str
            Caminho do arquivo JSON gerado
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"resultados_comparacao_{timestamp}.json"
            
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Adicionar metadata
        export_data = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'version': '1.0.0',
                'description': 'Resultados da comparação pré-downscaling ERA5 vs. Estação'
            },
            'results': analysis_results
        }
        
        # Salvar JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
        print(f"✅ Resultados JSON salvos em: {output_path}")
        return output_path