"""
config.py
Gerenciamento de configurações do sistema
"""

import os
import yaml
from typing import Dict, Any


class Config:
    """Gerenciador de configurações"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Carrega configurações do arquivo YAML"""
        
        if not os.path.exists(self.config_path):
            return self.get_default_config()
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
            return self.get_default_config()
            
    def get_default_config(self) -> Dict[str, Any]:
        """Configuração padrão"""
        
        return {
            'app': {
                'name': 'Sistema de Comparação Pré-Downscaling',
                'version': '1.0.0',
                'debug': False,
                'port': 5001,
                'host': '0.0.0.0'
            },
            'paths': {
                'uploads': './uploads',
                'results': './results',
                'plots': './static/img',
                'reports': './results/reports'
            },
            'analysis': {
                'variables': {
                    'temperature': {
                        'units': '°C',
                        'min_valid': -50,
                        'max_valid': 60
                    },
                    'precipitation': {
                        'units': 'mm',
                        'min_valid': 0,
                        'max_valid': 1000
                    }
                }
            },
            'plotting': {
                'figsize': [12, 8],
                'dpi': 300,
                'colors': {
                    'era5': '#1f77b4',
                    'station': '#ff7f0e',
                    'comparison': '#2ca02c'
                }
            }
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """Obter valor de configuração por chave"""
        
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def set(self, key: str, value: Any) -> None:
        """Definir valor de configuração"""
        
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def save_config(self) -> None:
        """Salvar configurações no arquivo"""
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        except Exception as e:
            print(f"Erro ao salvar configuração: {e}")


# Instância global
config = Config()