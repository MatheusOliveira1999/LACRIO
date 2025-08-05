"""
Validação de dados
"""

import pandas as pd


class DataValidator:
    """Classe para validação de dados"""
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Valida DataFrame básico"""
        if data is None or data.empty:
            return False
        if len(data) < 100:
            return False
        return True
