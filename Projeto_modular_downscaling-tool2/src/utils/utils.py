"""
utils.py
Funções utilitárias e auxiliares
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Imports locais usando imports absolutos
from src.visualization.visualization import Visualizer
from src.utils.report_generator import ReportGenerator


def run_weather_downscaling(era5_path, station_path, dem_path, 
                           variables=['temperature', 'precipitation'],
                           lat=-9.41, lon=-77.35, 
                           split_date='2018-01-01',
                           selected_models=None,
                           optimize_models=True,
                           use_ensemble=True,
                           save_results=True,
                           generate_plots=True):
    """
    Executa o pipeline completo de downscaling climático
    
    Parameters:
    -----------
    era5_path : str
        Caminho para dados ERA5/ERA5-Land
    station_path : str
        Caminho para dados da estação
    dem_path : str
        Caminho para modelo de elevação digital
    variables : list
        Lista de variáveis ['temperature', 'precipitation']
    lat, lon : float
        Coordenadas da estação
    split_date : str
        Data para divisão treino/teste
    selected_models : list
        Modelos a treinar (None = todos)
    optimize_models : bool
        Otimizar hiperparâmetros
    use_ensemble : bool
        Criar modelo ensemble
    save_results : bool
        Salvar modelos e resultados
    generate_plots : bool
        Gerar gráficos
    
    Returns:
    --------
    dict : Resultados para cada variável
    """
    
    from src.core.weather_model import WeatherDownscalingModel
    
    if selected_models is None:
        selected_models = ['LinearRegression', 'Ridge', 'RandomForest', 
                          'ExtraTrees', 'GradientBoosting', 'MLP', 'XGBoost']
    
    results = {}
    
    for variable in variables:
        print(f"\n{'='*70}")
        print(f"🌦️  DOWNSCALING CLIMÁTICO - {variable.upper()}")
        print(f"{'='*70}")
        
        try:
            # Configuração
            config = {
                'optimize_hyperparams': optimize_models,
                'feature_selection': True,
                'max_features': 100,
                'ensemble_size': min(3, len(selected_models))
            }
            
            # Criar modelo
            model = WeatherDownscalingModel(variable=variable, config=config)
            
            # Pipeline completo
            print("📊 Carregando dados...")
            model.load_and_merge_data(era5_path, station_path, dem_path, lat, lon)
            
            print("🔧 Criando features...")
            model.create_features()
            
            print("📈 Preparando dados...")
            model.prepare_data(split_date)
            
            print("🤖 Treinando modelos...")
            model.train_models(selected_models=selected_models, optimize=optimize_models)
            
            if use_ensemble and len(model.results) >= 2:
                print("🔗 Criando ensemble...")
                model.create_ensemble()
            
            if generate_plots:
                print("📊 Gerando visualizações...")
                visualizer = Visualizer(model)
                visualizer.generate_all_plots(save_plots=save_results)

            print("📋 Gerando relatório...")
            report_gen = ReportGenerator(model)
            report_gen.generate_report()
            
            if save_results:
                print("💾 Salvando modelos...")
                model.save_models()
            
            # Coletar resultados
            results[variable] = {
                'model': model,
                'best_model_name': min(model.results.items(), 
                                     key=lambda x: x[1]['metrics']['RMSE'])[0],
                'metrics': {name: data['metrics'] for name, data in model.results.items()},
                'success': True
            }
            
            print(f"✅ {variable.upper()} concluído com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro em {variable}: {str(e)}")
            results[variable] = {'error': str(e), 'success': False}
    
    # Resumo final
    print(f"\n{'='*70}")
    print("📊 RESUMO FINAL")
    print(f"{'='*70}")
    
    for variable, result in results.items():
        if result.get('success', False):
            best_model = result['best_model_name']
            best_metrics = result['metrics'][best_model]
            print(f"\n🎯 {variable.upper()}:")
            print(f"   • Melhor modelo: {best_model}")
            print(f"   • RMSE: {best_metrics['RMSE']:.4f}")
            print(f"   • R²: {best_metrics['R2']:.4f}")
        else:
            print(f"\n❌ {variable.upper()}: FALHOU")
    
    return results


def run_downscaling_pipeline_advanced(files, params, selected_models, variables, 
                                    use_ensemble, optimize_params, opt_config,
                                    temporal_resolution='auto', enable_validation=True):
    """Pipeline principal com configurações avançadas de otimização para web app"""
    
    from src.core.weather_model import WeatherDownscalingModel
    from src.visualization.visualization import Visualizer
    from src.utils.report_generator import ReportGenerator
    
    lat = params['latitude']
    lon = params['longitude']
    split_date = params['split_date']

    era5_path = files['era5']
    station_path = files['station']
    dem_path = files['dem']

    results_all = {}

    for variable in variables:
        print(f"\n{'='*60}")
        print(f"PROCESSANDO {variable.upper()}")
        print(f"{'='*60}")

        try:
            # Configuração do modelo com parâmetros avançados
            config = {
                'optimize_hyperparams': optimize_params,
                'feature_selection': opt_config.get('feature_selection', True),
                'max_features': 80,
                'ensemble_size': 3,
                'temporal_resolution': temporal_resolution,  # Nova configuração
                'optimize_all_models': opt_config.get('optimize_all_models', True),
                'fast_optimization': opt_config.get('fast_optimization', True),
                'use_random_search': opt_config.get('use_random_search', False),
                'random_search_iter': opt_config.get('random_search_iter', 30),
                'cv_folds': opt_config.get('cv_folds', 3),
                'show_top_params': False,
                'verbose': False
            }

            # Criar modelo
            model = WeatherDownscalingModel(variable=variable, config=config)

            # Pipeline completo
            model.load_and_merge_data(era5_path, station_path, dem_path, lat, lon)
            model.create_features()
            model.prepare_data(split_date)
            model.train_models(selected_models=selected_models, optimize=optimize_params)

            # Criar ensemble se solicitado
            if use_ensemble and len(model.results) >= 2:
                model.create_ensemble()

            # Gerar visualizações
            visualizer = Visualizer(model)
            visualizer.generate_all_plots(save_plots=True)
            
            # Gerar relatório
            report_gen = ReportGenerator(model)
            report_gen.generate_report()

            # Salvar modelos
            model.save_models()

            # Coletar resultados
            results_all[variable] = {
                'models': {
                    model_name: {
                        'metrics': data['metrics'],
                        'model_type': type(data['model']).__name__
                    }
                    for model_name, data in model.results.items()
                },
                'best_model': min(model.results.items(), 
                                key=lambda x: x[1]['metrics']['RMSE'])[0] if len(model.results) > 0 else None,
                'data_info': {
                    'total_records': len(model.data) if hasattr(model, 'data') and not (model.data is None) and hasattr(model.data, '__len__') else 0,
                    'train_records': len(model.X_train) if hasattr(model, 'X_train') and not (model.X_train is None) and hasattr(model.X_train, '__len__') else 0,
                    'test_records': len(model.X_test) if hasattr(model, 'X_test') and not (model.X_test is None) and hasattr(model.X_test, '__len__') else 0,
                    'features_count': len(model.features) if hasattr(model, 'features') and not (model.features is None) and hasattr(model.features, '__len__') else 0
                }
            }

            print(f"\n✅ {variable.upper()} processado com sucesso!")

        except Exception as e:
            print(f"\n❌ Erro ao processar {variable}: {str(e)}")
            results_all[variable] = {'error': str(e)}

    return results_all