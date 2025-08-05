#!/usr/bin/env python3
"""
Exemplo avançado com todas as funcionalidades
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import run_downscaling_analysis


def main():
    """Exemplo avançado"""
    
    # Configuração personalizada
    config = {
        'optimize_hyperparams': True,
        'feature_selection': True,
        'max_features': 50,
        'ensemble_size': 3,
        'fast_optimization': False,
        'cv_folds': 5,
        'verbose': True
    }
    
    # Parâmetros
    era5_path = "dados_era5_era5land.csv"
    station_path = "dados_estacao_cuchillacocha.csv" 
    dem_path = "Quilcayhuanca_static.nc"
    
    print("🚀 Executando análise avançada de downscaling...")
    
    # Executar análise completa
    results = run_downscaling_analysis(
        era5_path=era5_path,
        station_path=station_path,
        dem_path=dem_path,
        variables=['temperature', 'precipitation'],
        lat=-9.41,
        lon=-77.35,
        split_date='2018-01-01',
        selected_models=['Ridge', 'RandomForest', 'XGBoost', 'MLP'],
        optimize_models=True,
        use_ensemble=True,
        save_results=True,
        generate_plots=True,
        config=config
    )
    
    # Analisar resultados
    for variable, result in results.items():
        if result.get('success'):
            print(f"\n✅ {variable.upper()}:")
            print(f"   Melhor modelo: {result['best_model']}")
            print(f"   RMSE: {result['best_metrics']['RMSE']:.4f}")
            print(f"   R²: {result['best_metrics']['R2']:.4f}")
        else:
            print(f"\n❌ {variable.upper()}: {result.get('error')}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
