"""
report_generator.py
Módulo para geração de relatórios
"""

import os
from datetime import datetime
import pandas as pd


class ReportGenerator:
    """Gera relatórios dos resultados do modelo"""
    
    def __init__(self, model):
        self.model = model
        self.variable = model.variable
        self.results = model.results
        self.config = model.config
        
    def generate_report(self):
        """Gera relatório completo dos resultados"""
        if not self.results:
            print("Nenhum resultado disponível para relatório")
            return
        
        report_lines = []
        
        # Cabeçalho
        report_lines.append("="*80)
        report_lines.append(f"RELATÓRIO DE DOWNSCALING CLIMÁTICO - {self.variable.upper()}")
        report_lines.append("="*80)
        report_lines.append(f"Data de geração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Informações dos dados
        report_lines.append("\n📊 INFORMAÇÕES DOS DADOS:")
        report_lines.append(f"   • Total de registros: {len(self.model.data)}")
        report_lines.append(f"   • Features utilizadas: {len(self.model.features)}")
        report_lines.append(f"   • Dados de treino: {len(self.model.X_train)}")
        report_lines.append(f"   • Dados de teste: {len(self.model.X_test)}")
        report_lines.append(f"   • Período de teste: {self.model.test_dates.min().strftime('%Y-%m-%d')} a {self.model.test_dates.max().strftime('%Y-%m-%d')}")
        
        # Configurações
        report_lines.append("\n⚙️ CONFIGURAÇÕES:")
        for key, value in self.config.items():
            report_lines.append(f"   • {key}: {value}")
        
        # Resultados dos modelos
        report_lines.append("\n🎯 RESULTADOS DOS MODELOS:")
        report_lines.append("-"*60)
        
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['metrics']['RMSE'])
        
        for i, (name, result) in enumerate(sorted_results, 1):
            metrics = result['metrics']
            report_lines.append(f"\n{i}. {name}")
            report_lines.append(f"   ├─ RMSE: {metrics['RMSE']:.4f} {self._get_unit()}")
            report_lines.append(f"   ├─ MAE: {metrics['MAE']:.4f} {self._get_unit()}")
            report_lines.append(f"   ├─ R²: {metrics['R2']:.4f}")
            report_lines.append(f"   ├─ Correlação: {metrics['Correlation']:.4f}")
            report_lines.append(f"   ├─ Bias: {metrics['Bias']:.4f} {self._get_unit()}")
            report_lines.append(f"   └─ Skill Score: {metrics['Skill_Score']:.4f}")
            
            if self.variable == 'precipitation':
                prec_metrics = []
                if 'Rain_Detection_Accuracy' in metrics:
                    prec_metrics.append(f"   ├─ Detecção de Chuva: {metrics['Rain_Detection_Accuracy']:.1f}%")
                if 'POD' in metrics and not pd.isna(metrics['POD']):
                    prec_metrics.append(f"   ├─ POD: {metrics['POD']:.3f}")
                if 'FAR' in metrics and not pd.isna(metrics['FAR']):
                    prec_metrics.append(f"   ├─ FAR: {metrics['FAR']:.3f}")
                if 'R2_Rain_Days' in metrics and not pd.isna(metrics['R2_Rain_Days']):
                    prec_metrics.append(f"   └─ R² (dias com chuva): {metrics['R2_Rain_Days']:.4f}")
                
                if prec_metrics and "└─ Skill Score" in report_lines[-1]:
                    report_lines[-1] = report_lines[-1].replace('└─', '├─')
                
                report_lines.extend(prec_metrics)
        
        # Melhor modelo
        best_model = sorted_results[0]
        report_lines.append(f"\n🏆 MELHOR MODELO: {best_model[0]}")
        report_lines.append(f"   • RMSE: {best_model[1]['metrics']['RMSE']:.4f} {self._get_unit()}")
        report_lines.append(f"   • R²: {best_model[1]['metrics']['R2']:.4f}")
        report_lines.append(f"   • Melhoria sobre climatologia: {best_model[1]['metrics']['Skill_Score']:.1%}")
        
        # Interpretação dos resultados
        report_lines.append(f"\n📈 INTERPRETAÇÃO:")
        r2 = best_model[1]['metrics']['R2']
        if r2 >= 0.8:
            quality = "Excelente"
        elif r2 >= 0.6:
            quality = "Boa"
        elif r2 >= 0.4:
            quality = "Moderada"
        else:
            quality = "Baixa"
        
        report_lines.append(f"   • Qualidade da predição: {quality} (R² = {r2:.3f})")
        
        skill = best_model[1]['metrics']['Skill_Score']
        if skill > 0:
            report_lines.append(f"   • O modelo é {skill:.1%} melhor que a climatologia")
        else:
            report_lines.append(f"   • O modelo não supera a climatologia")
        
        report_lines.append("\n" + "="*80)
        
        # Juntar as linhas em uma única string
        full_report_text = "\n".join(report_lines)
        
        # Imprimir o relatório no terminal
        print(full_report_text)
        
        # Salvar o relatório em arquivo
        self._save_report(full_report_text)
        
        return full_report_text
    
    def _save_report(self, report_text):
        """Salva relatório em arquivo organizando por estação"""
        # Caminho absoluto para results na raiz do projeto
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Obter nome da estação do modelo
        station_name = getattr(self.model, 'station_name', None) or 'unknown_station'
        
        # Criar diretório específico da estação
        station_results_dir = os.path.join(project_root, 'results', station_name)
        if not os.path.exists(station_results_dir):
            os.makedirs(station_results_dir)
        
        report_path = os.path.join(station_results_dir, f'relatorio_downscaling_{self.variable}.txt')
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"\n📄 Relatório salvo em: {report_path}")
        
        except Exception as e:
            print(f"Erro ao salvar relatório: {str(e)}")
    
    def _get_unit(self):
        """Retorna a unidade de medida"""
        return '°C' if self.variable == 'temperature' else 'mm'