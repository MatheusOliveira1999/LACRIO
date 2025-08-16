"""
Pacote src - Codigo fonte principal
"""
import os
import sys

# Adicionar o diretorio raiz ao sys.path automaticamente
_current_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_current_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)