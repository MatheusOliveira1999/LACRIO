"""
Criação de modelos ensemble
"""

from sklearn.ensemble import VotingRegressor


class EnsembleCreator:
    """Classe para criar modelos ensemble"""
    
    def create_voting_ensemble(self, results: dict, ensemble_size: int = 3):
        """Cria ensemble por votação"""
        
        if len(results) < 2:
            raise ValueError("Necessário pelo menos 2 modelos")
        
        # Selecionar melhores modelos
        best_models = sorted(results.items(), 
                           key=lambda x: x[1]['metrics']['RMSE'])[:ensemble_size]
        
        # Criar ensemble
        ensemble_models = [(name, result['model']) for name, result in best_models]
        ensemble = VotingRegressor(ensemble_models)
        
        return ensemble
