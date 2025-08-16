"""
validador_downscaling.py
Módulo para validação específica de downscaling com dados horários
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import warnings


class ValidadorDownscaling:
    """Valida modelos de downscaling e detecta problemas específicos com dados horários"""
    
    def __init__(self):
        self.problemas_detectados = []
        
    def validar_resultados_completos(self, data: pd.DataFrame, results: Dict[str, Any], 
                                   variable: str = 'temperature') -> Dict[str, Any]:
        """
        Validação completa dos resultados de downscaling
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dados processados usados no modelo
        results : Dict
            Resultados dos modelos (deve conter métricas como R², RMSE, etc.)
        variable : str
            Variável sendo modelada
            
        Returns:
        --------
        Dict com diagnóstico completo
        """
        diagnostico = {
            'vazamento_temporal': False,
            'overfitting_extremo': False,
            'resolucao_temporal': None,
            'autocorrelacao': {},
            'qualidade_dados': {},
            'recomendacoes': [],
            'metricas_ajustadas': {}
        }
        
        # 1. Detectar resolução temporal
        diagnostico['resolucao_temporal'] = self._detectar_resolucao_temporal(data)
        
        # 2. Analisar autocorrelação
        diagnostico['autocorrelacao'] = self._analisar_autocorrelacao(data, variable)
        
        # 3. Detectar vazamento temporal
        diagnostico['vazamento_temporal'] = self._detectar_vazamento_temporal(
            data, results, diagnostico['resolucao_temporal']
        )
        
        # 4. Detectar overfitting extremo
        diagnostico['overfitting_extremo'] = self._detectar_overfitting_extremo(results)
        
        # 5. Avaliar qualidade dos dados
        diagnostico['qualidade_dados'] = self._avaliar_qualidade_dados(data, variable)
        
        # 6. Gerar recomendações
        diagnostico['recomendacoes'] = self._gerar_recomendacoes(diagnostico)
        
        # 7. Calcular métricas ajustadas para dados horários
        if diagnostico['resolucao_temporal'] == 'horaria':
            diagnostico['metricas_ajustadas'] = self._calcular_metricas_ajustadas(
                data, results, variable
            )
        
        return diagnostico
    
    def _detectar_resolucao_temporal(self, data: pd.DataFrame) -> str:
        """Detecta se os dados são horários, diários ou outro intervalo"""
        if 'date' not in data.columns:
            return 'indefinida'
        
        try:
            dates = pd.to_datetime(data['date'])
            diff_hours = dates.diff().dt.total_seconds() / 3600
            mode_diff = diff_hours.mode().iloc[0] if len(diff_hours.mode()) > 0 else 24
            
            if 0.8 <= mode_diff <= 1.2:  # Aproximadamente 1 hora
                return 'horaria'
            elif 23 <= mode_diff <= 25:  # Aproximadamente 24 horas
                return 'diaria'
            elif mode_diff < 1:
                return 'sub-horaria'
            else:
                return f'custom_{mode_diff:.1f}h'
        except:
            return 'indefinida'
    
    def _analisar_autocorrelacao(self, data: pd.DataFrame, variable: str) -> Dict:
        """Analisa autocorrelação da variável target"""
        target_col = 'temp_obs' if variable == 'temperature' else 'prec_obs'
        
        if target_col not in data.columns:
            return {'erro': f'Coluna {target_col} não encontrada'}
        
        serie = data[target_col].dropna()
        
        autocorr = {}
        lags = [1, 6, 12, 24, 48, 168]  # 1h, 6h, 12h, 1d, 2d, 1semana
        
        for lag in lags:
            if len(serie) > lag:
                try:
                    autocorr[f'lag_{lag}h'] = float(serie.autocorr(lag=lag))
                except:
                    autocorr[f'lag_{lag}h'] = np.nan
        
        return autocorr
    
    def _detectar_vazamento_temporal(self, data: pd.DataFrame, results: Dict, 
                                   resolucao: str) -> bool:
        """Detecta vazamento temporal baseado em R² extremamente alto e resolução"""
        # Critérios diferentes por resolução temporal
        if 'horaria' in resolucao or 'custom' in resolucao:
            # Para dados sub-diários (horários, 6h, etc.)
            r2_threshold = 0.95  # Mais rigoroso para dados temporais
        else:
            r2_threshold = 0.98  # Menos rigoroso para dados diários
        
        # Verificar se algum modelo tem R² extremo
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and 'r2' in metrics:
                if metrics['r2'] > r2_threshold:
                    self.problemas_detectados.append(
                        f"Possível vazamento temporal: {model_name} R²={metrics['r2']:.4f}"
                    )
                    return True
        
        return False
    
    def _detectar_overfitting_extremo(self, results: Dict) -> bool:
        """Detecta overfitting extremo baseado em métricas irreais"""
        for model_name, metrics in results.items():
            if isinstance(metrics, dict):
                # R² > 99.5% é quase sempre problemático
                if metrics.get('r2', 0) > 0.995:
                    return True
                
                # RMSE extremamente baixo
                if metrics.get('rmse', float('inf')) < 0.1:
                    return True
        
        return False
    
    def _avaliar_qualidade_dados(self, data: pd.DataFrame, variable: str) -> Dict:
        """Avalia qualidade dos dados para downscaling"""
        target_col = 'temp_obs' if variable == 'temperature' else 'prec_obs'
        
        qualidade = {
            'registros_totais': len(data),
            'registros_validos': 0,
            'taxa_completude': 0,
            'variancia_target': 0,
            'range_target': (0, 0)
        }
        
        if target_col in data.columns:
            valid_data = data[target_col].dropna()
            qualidade['registros_validos'] = len(valid_data)
            qualidade['taxa_completude'] = len(valid_data) / len(data)
            
            if len(valid_data) > 0:
                qualidade['variancia_target'] = float(valid_data.var())
                qualidade['range_target'] = (float(valid_data.min()), float(valid_data.max()))
        
        return qualidade
    
    def _gerar_recomendacoes(self, diagnostico: Dict) -> list:
        """Gera recomendações baseadas no diagnóstico"""
        recomendacoes = []
        
        if 'horaria' in diagnostico['resolucao_temporal'] or 'custom' in diagnostico['resolucao_temporal']:
            recomendacoes.append("DADOS SUB-DIÁRIOS DETECTADOS:")
            
            # Verificar autocorrelação alta e dar recomendações específicas
            autocorr = diagnostico['autocorrelacao']
            
            # Autocorrelação muito alta em 24h indica problema sério
            lag_24h = autocorr.get('lag_24h', 0)
            lag_12h = autocorr.get('lag_12h', 0)
            
            if lag_24h > 0.8:
                recomendacoes.append("⚠️  AUTOCORRELAÇÃO MUITO ALTA (24h):")
                recomendacoes.append("- Usar APENAS lags ≥ 24 horas para temperatura")
                recomendacoes.append("- Remover features de médias móveis < 24h")
                recomendacoes.append("- Considerar agregar para dados diários")
            elif lag_12h > 0.8:
                recomendacoes.append("⚠️  AUTOCORRELAÇÃO ALTA (12h):")
                recomendacoes.append("- Usar lags ≥ 12 horas")
                recomendacoes.append("- Janelas de média móvel ≥ 24h")
            else:
                recomendacoes.append("- Usar lags ≥ 6 horas para features temporais")
            
            recomendacoes.append("- TimeSeriesSplit com cv_folds ≤ 4")
            recomendacoes.append("- Validação temporal rigorosa")
            
            # Recomendação específica para alta autocorrelação
            if lag_24h > 0.8:
                recomendacoes.append("💡 SOLUÇÃO: Converter para dados diários pode resolver o overfitting")
        
        if diagnostico['vazamento_temporal']:
            recomendacoes.append("⚠️  VAZAMENTO TEMPORAL DETECTADO:")
            recomendacoes.append("- Revisar estratégia de criação de features")
            recomendacoes.append("- Implementar validação temporal rigorosa")
            recomendacoes.append("- Remover features de lag muito pequenos")
        
        if diagnostico['overfitting_extremo']:
            recomendacoes.append("⚠️  OVERFITTING EXTREMO DETECTADO:")
            recomendacoes.append("- Reduzir complexidade dos modelos")
            recomendacoes.append("- Aumentar regularização")
            recomendacoes.append("- Usar validação cruzada mais rigorosa")
        
        if diagnostico['qualidade_dados']['taxa_completude'] < 0.8:
            recomendacoes.append("- Melhorar qualidade dos dados (completude < 80%)")
        
        return recomendacoes
    
    def _calcular_metricas_ajustadas(self, data: pd.DataFrame, results: Dict, 
                                   variable: str) -> Dict:
        """Calcula métricas ajustadas para dados horários"""
        # Para dados horários, as métricas devem ser interpretadas diferentemente
        # R² > 0.9 pode ser normal devido à alta correlação temporal
        
        metricas_ajustadas = {}
        
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and 'r2' in metrics:
                r2_original = metrics['r2']
                
                # Penalizar R² extremamente alto em dados horários
                if r2_original > 0.99:
                    r2_penalizado = r2_original * 0.8  # Penalidade por suspeita de vazamento
                    metricas_ajustadas[f'{model_name}_r2_penalizado'] = r2_penalizado
                
                # Calcular interpretação qualitativa
                if r2_original > 0.99:
                    interpretacao = "Suspeito - possível vazamento"
                elif r2_original > 0.95:
                    interpretacao = "Muito alto - verificar features"
                elif r2_original > 0.8:
                    interpretacao = "Bom para dados horários"
                else:
                    interpretacao = "Adequado"
                
                metricas_ajustadas[f'{model_name}_interpretacao'] = interpretacao
        
        return metricas_ajustadas
    
    def gerar_relatorio_validacao(self, diagnostico: Dict) -> str:
        """Gera relatório de validação em formato texto"""
        relatorio = []
        relatorio.append("=" * 80)
        relatorio.append("RELATÓRIO DE VALIDAÇÃO DE DOWNSCALING")
        relatorio.append("=" * 80)
        
        # Informações básicas
        relatorio.append(f"Resolução temporal: {diagnostico['resolucao_temporal']}")
        relatorio.append(f"Vazamento temporal detectado: {'SIM' if diagnostico['vazamento_temporal'] else 'NÃO'}")
        relatorio.append(f"Overfitting extremo: {'SIM' if diagnostico['overfitting_extremo'] else 'NÃO'}")
        
        # Autocorrelação
        relatorio.append("\nAUTOCORRELAÇÃO:")
        for lag, valor in diagnostico['autocorrelacao'].items():
            if not pd.isna(valor):
                relatorio.append(f"  {lag}: {valor:.4f}")
        
        # Qualidade dos dados
        qualidade = diagnostico['qualidade_dados']
        relatorio.append(f"\nQUALIDADE DOS DADOS:")
        relatorio.append(f"  Registros totais: {qualidade['registros_totais']}")
        relatorio.append(f"  Registros válidos: {qualidade['registros_validos']}")
        relatorio.append(f"  Taxa de completude: {qualidade['taxa_completude']:.1%}")
        
        # Recomendações
        relatorio.append("\nRECOMENDAÇÕES:")
        for rec in diagnostico['recomendacoes']:
            relatorio.append(f"  • {rec}")
        
        # Métricas ajustadas
        if diagnostico['metricas_ajustadas']:
            relatorio.append("\nMÉTRICAS AJUSTADAS:")
            for metrica, valor in diagnostico['metricas_ajustadas'].items():
                relatorio.append(f"  {metrica}: {valor}")
        
        relatorio.append("=" * 80)
        
        return "\n".join(relatorio)