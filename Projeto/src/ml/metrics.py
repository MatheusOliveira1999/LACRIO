"""
Módulo para cálculo de métricas de avaliação
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any


class MetricsCalculator:
    """Classe para calcular métricas de avaliação"""
    
    def __init__(self, variable: str = 'temperature'):
        self.variable = variable
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula todas as métricas relevantes"""
        
        # Converter para numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Métricas básicas
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        
        # Skill Score
        ss_clim = 1 - (rmse**2 / np.var(y_true))
        
        # Correlação
        if len(y_true) > 1:
            corr = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            corr = np.nan
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Bias': bias,
            'Skill_Score': ss_clim,
            'Correlation': corr
        }
        
        # Métricas específicas para precipitação
        if self.variable == 'precipitation':
            rain_threshold = 0.1
            rain_true = y_true > rain_threshold
            rain_pred = y_pred > rain_threshold
            
            if len(rain_true) > 0:
                rain_accuracy = np.mean(rain_true == rain_pred) * 100
                metrics['Rain_Detection_Accuracy'] = rain_accuracy
                
                if np.sum(rain_true) > 0:
                    pod = np.sum(rain_true & rain_pred) / np.sum(rain_true)
                    metrics['POD'] = pod
                
                if np.sum(rain_pred) > 0:
                    far = np.sum(~rain_true & rain_pred) / np.sum(rain_pred)
                    metrics['FAR'] = far
        
        return metrics
