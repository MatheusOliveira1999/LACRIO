"""
Gerador de relatórios
"""

import os
from datetime import datetime
from config.settings import RESULTS_FOLDER


def generate_report(model, variable: str):
    """Gera relatório simples"""
    
    if not model.results:
        return
    
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    report_path = os.path.join(RESULTS_FOLDER, f'relatorio_downscaling_{variable}.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"RELATÓRIO - {variable.upper()}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Data: {datetime.now()}\n\n")
        
        sorted_results = sorted(model.results.items(), 
                               key=lambda x: x[1]['metrics']['RMSE'])
        
        for name, result in sorted_results:
            metrics = result['metrics']
            f.write(f"{name}:\n")
            f.write(f"  RMSE: {metrics['RMSE']:.4f}\n")
            f.write(f"  R²: {metrics['R2']:.4f}\n\n")
    
    print(f"📄 Relatório: {report_path}")
