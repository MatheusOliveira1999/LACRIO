# config/__init__.py
"""Configurações do sistema de downscaling"""

from .settings import *

# src/__init__.py
"""Sistema de downscaling climático"""

__version__ = "1.0.0"
__author__ = "Sistema de Downscaling Climático"

# src/data/__init__.py
"""Módulos de carregamento e processamento de dados"""

from .loader import DataLoader
from .feature_engineer import FeatureEngineer

__all__ = ['DataLoader', 'FeatureEngineer']

# src/models/__init__.py
"""Modelos de downscaling"""

from .downscaling_model import WeatherDownscalingModel

__all__ = ['WeatherDownscalingModel']

# src/ml/__init__.py
"""Algoritmos de Machine Learning"""

from .model_factory import ModelFactory
from .metrics import MetricsCalculator

__all__ = ['ModelFactory', 'MetricsCalculator']

# src/utils/__init__.py
"""Utilitários diversos"""

# src/visualization/__init__.py
"""Módulos de visualização"""

# src/web/__init__.py
"""Interface web"""

from .app import create_app

__all__ = ['create_app']