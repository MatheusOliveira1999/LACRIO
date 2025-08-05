#!/usr/bin/env python3
"""
Script principal para execução do sistema de downscaling climático
"""

import os
import sys
import argparse
from typing import List, Dict, Any

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.downscaling_model import WeatherDownscalingModel
from src.visualization.plots import PlotGenerator
from src.utils.file_utils import create_project_directories, validate_input_files
from config.settings import create_directories, AVAILABLE_MODELS, CLIMATE_VARIABLES


def run_downscaling_analysis(
    era5_path: str,
    station_path: str, 
    dem_path: str,
    variables: List[str] = None,
    lat: float = -9.41,
    lon: float = -77.35,
    split_date: str = '2018-01-01',
    selected_models: List[str] = None,
    optimize_models: bool = True,
    use_ensemble: bool = True,
    save_results: bool = True,
    generate_plots: bool = True,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Executa análise completa de downscaling climático
    
    Parameters:
    -----------
    era5_path : str
        Caminho para dados ERA5/ERA5-Land
    station_path : str
        Caminho para dados da estação meteorológica
    dem_path : str
        Caminho para modelo de elevação digital
    variables : list
        Lista de variáveis climáticas ['temperature', 'precipitation']
    lat, lon : float
        Coordenadas da estação
    split_date : str
        Data para divisão treino/teste
    selected_models : list
        Modelos específicos para treinar
    optimize_models : bool
        Se deve otimizar hiperparâmetros
    use_ensemble : bool
        Se deve criar modelo ensemble
    save_results : bool
        Se deve salvar modelos e resultados
    generate_plots : bool
        Se deve gerar visualizações
    config : dict
        Configurações adicionais
        
    Returns:
    --------
    dict : Resultados da análise
    """
    
    # Configurações padrão
    if variables is None:
        variables = ['temperature', 'precipitation']
    
    if selected_models is None:
        selected_models = ['LinearRegression', 'Ridge', 'RandomForest', 
                          'ExtraTrees', 'GradientBoosting', 'MLP', 'XGBoost']
    
    # Validar arquivos de entrada
    print("🔍 Validando arquivos de entrada...")
    validation_result = validate_input_files(era5_path, station_path, dem_path)
    
    if not all(validation_result.values()):
        print("❌ Alguns arquivos são inválidos:")
        for file_type, is_valid in validation_result.items():
            status = "✅" if is_valid else "❌"
            print(f"  {status} {file_type}")
        return {'error': 'Arquivos de entrada inválidos'}
    
    print("✅ Todos os arquivos são válidos")
    
    # Criar diretórios necessários
    create_directories()
    
    # Resultados finais
    results = {}
    
    # Processar cada variável
    for variable in variables:
        print(f"\n{'='*70}")
        print(f"🌦️  PROCESSANDO {variable.upper()}")
        print(f"{'='*70}")
        
        try:
            # Configurar modelo
            model_config = config.copy() if config else {}
            
            # Criar modelo
            model = WeatherDownscalingModel(variable=variable, config=model_config)
            
            # Pipeline completo
            print("📊 Carregando e mesclando dados...")
            model.load_and_merge_data(era5_path, station_path, dem_path, lat, lon)
            
            print("🔧 Criando features...")
            model.create_features()
            
            print("📈 Preparando dados para modelagem...")
            model.prepare_data(split_date)
            
            print("🤖 Treinando modelos...")
            model.train_models(selected_models=selected_models, optimize=optimize_models)
            
            # Criar ensemble se solicitado
            if use_ensemble and len(model.results) >= 2:
                print("🔗 Criando modelo ensemble...")
                model.create_ensemble()
            
            # Gerar visualizações
            if generate_plots:
                print("📊 Gerando visualizações...")
                plot_generator = PlotGenerator(model)
                plot_generator.generate_all_plots(save_plots=save_results)
            
            # Salvar modelos
            if save_results:
                print("💾 Salvando modelos...")
                model.save_models()
            
            # Coletar resultados
            best_model_name = model.best_model
            best_metrics = model.results[best_model_name]['metrics'] if best_model_name else {}
            
            results[variable] = {
                'success': True,
                'best_model': best_model_name,
                'best_metrics': best_metrics,
                'all_models': {name: data['metrics'] for name, data in model.results.items()},
                'data_info': {
                    'total_records': len(model.data),
                    'train_records': len(model.X_train),
                    'test_records': len(model.X_test),
                    'features_count': len(model.features)
                }
            }
            
            print(f"✅ {variable.upper()} processado com sucesso!")
            print(f"🏆 Melhor modelo: {best_model_name}")
            print(f"📊 RMSE: {best_metrics.get('RMSE', 'N/A'):.4f}")
            print(f"📊 R²: {best_metrics.get('R2', 'N/A'):.4f}")
            
        except Exception as e:
            print(f"❌ Erro ao processar {variable}: {str(e)}")
            results[variable] = {
                'success': False,
                'error': str(e)
            }
    
    # Resumo final
    print(f"\n{'='*70}")
    print("📊 RESUMO FINAL")
    print(f"{'='*70}")
    
    successful_vars = []
    failed_vars = []
    
    for variable, result in results.items():
        if result.get('success', False):
            successful_vars.append(variable)
            best_model = result['best_model']
            rmse = result['best_metrics'].get('RMSE', 0)
            r2 = result['best_metrics'].get('R2', 0)
            
            print(f"\n✅ {variable.upper()}:")
            print(f"   🏆 Melhor modelo: {best_model}")
            print(f"   📊 RMSE: {rmse:.4f}")
            print(f"   📊 R²: {r2:.4f}")
        else:
            failed_vars.append(variable)
            print(f"\n❌ {variable.upper()}: FALHOU")
            print(f"   💥 Erro: {result.get('error', 'Erro desconhecido')}")
    
    print(f"\n📈 Variáveis processadas com sucesso: {len(successful_vars)}/{len(variables)}")
    
    if failed_vars:
        print(f"⚠️ Variáveis com falha: {failed_vars}")
    
    return results


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Sistema de Downscaling Climático com Machine Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

1. Análise básica:
   python main.py --era5 dados_era5.csv --station dados_estacao.csv --dem modelo_elevacao.nc

2. Apenas temperatura com modelos específicos:
   python main.py --era5 dados_era5.csv --station dados_estacao.csv --dem modelo_elevacao.nc 
                  --variables temperature --models RandomForest XGBoost

3. Análise completa com otimização:
   python main.py --era5 dados_era5.csv --station dados_estacao.csv --dem modelo_elevacao.nc 
                  --optimize --ensemble --save-plots

4. Coordenadas personalizadas:
   python main.py --era5 dados_era5.csv --station dados_estacao.csv --dem modelo_elevacao.nc 
                  --lat -10.5 --lon -75.2
        """
    )
    
    # Argumentos obrigatórios
    parser.add_argument('--era5', required=True, 
                       help='Caminho para arquivo CSV com dados ERA5/ERA5-Land')
    parser.add_argument('--station', required=True,
                       help='Caminho para arquivo CSV com dados da estação')
    parser.add_argument('--dem', required=True,
                       help='Caminho para arquivo NetCDF com modelo de elevação')
    
    # Localização
    parser.add_argument('--lat', type=float, default=-9.41,
                       help='Latitude da estação (padrão: -9.41)')
    parser.add_argument('--lon', type=float, default=-77.35,
                       help='Longitude da estação (padrão: -77.35)')
    
    # Configurações de processamento
    parser.add_argument('--variables', nargs='+', choices=CLIMATE_VARIABLES,
                       default=CLIMATE_VARIABLES,
                       help=f'Variáveis climáticas para processar (padrão: {CLIMATE_VARIABLES})')
    
    parser.add_argument('--models', nargs='+', choices=AVAILABLE_MODELS,
                       help=f'Modelos específicos para treinar (padrão: todos disponíveis)')
    
    parser.add_argument('--split-date', default='2018-01-01',
                       help='Data para divisão treino/teste (formato: YYYY-MM-DD)')
    
    # Opções de treinamento
    parser.add_argument('--optimize', action='store_true',
                       help='Otimizar hiperparâmetros dos modelos')
    parser.add_argument('--no-optimize', action='store_true',
                       help='Não otimizar hiperparâmetros (mais rápido)')
    
    parser.add_argument('--ensemble', action='store_true',
                       help='Criar modelo ensemble')
    parser.add_argument('--no-ensemble', action='store_true',
                       help='Não criar modelo ensemble')
    
    # Saídas
    parser.add_argument('--save-models', action='store_true',
                       help='Salvar modelos treinados')
    parser.add_argument('--save-plots', action='store_true',
                       help='Gerar e salvar gráficos')
    parser.add_argument('--no-plots', action='store_true',
                       help='Não gerar gráficos (mais rápido)')
    
    # Configurações avançadas
    parser.add_argument('--fast', action='store_true',
                       help='Modo rápido: sem otimização e menos gráficos')
    
    parser.add_argument('--config', 
                       help='Arquivo JSON com configurações adicionais')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Saída detalhada')
    
    args = parser.parse_args()
    
    # Validar argumentos conflitantes
    if args.optimize and args.no_optimize:
        parser.error("--optimize e --no-optimize são mutuamente exclusivos")
    
    if args.ensemble and args.no_ensemble:
        parser.error("--ensemble e --no-ensemble são mutuamente exclusivos")
    
    if args.save_plots and args.no_plots:
        parser.error("--save-plots e --no-plots são mutuamente exclusivos")
    
    # Configurar opções baseado nos argumentos
    if args.fast:
        optimize_models = False
        use_ensemble = False
        generate_plots = False
        save_results = False
    else:
        optimize_models = args.optimize if (args.optimize or args.no_optimize) else True
        use_ensemble = args.ensemble if (args.ensemble or args.no_ensemble) else True
        generate_plots = not args.no_plots if args.no_plots else True
        save_results = args.save_models or args.save_plots
    
    # Carregar configurações adicionais
    config = {}
    if args.config and os.path.exists(args.config):
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"📁 Configurações carregadas de {args.config}")
    
    # Configurações baseadas em argumentos
    if args.verbose:
        config['verbose'] = True
    
    print("🚀 SISTEMA DE DOWNSCALING CLIMÁTICO")
    print("="*50)
    print(f"📊 ERA5: {args.era5}")
    print(f"🌡️ Estação: {args.station}")
    print(f"🏔️ DEM: {args.dem}")
    print(f"📍 Coordenadas: ({args.lat}, {args.lon})")
    print(f"🌦️ Variáveis: {args.variables}")
    print(f"🤖 Modelos: {args.models or 'Todos disponíveis'}")
    print(f"⚙️ Otimização: {optimize_models}")
    print(f"🔗 Ensemble: {use_ensemble}")
    print(f"📊 Gráficos: {generate_plots}")
    print("="*50)
    
    try:
        # Executar análise
        results = run_downscaling_analysis(
            era5_path=args.era5,
            station_path=args.station,
            dem_path=args.dem,
            variables=args.variables,
            lat=args.lat,
            lon=args.lon,
            split_date=args.split_date,
            selected_models=args.models,
            optimize_models=optimize_models,
            use_ensemble=use_ensemble,
            save_results=save_results,
            generate_plots=generate_plots,
            config=config
        )
        
        # Verificar se houve sucesso
        successful_count = sum(1 for r in results.values() if r.get('success', False))
        total_count = len(results)
        
        if successful_count == total_count:
            print(f"\n🎉 ANÁLISE CONCLUÍDA COM SUCESSO!")
            print(f"📊 {successful_count}/{total_count} variáveis processadas")
            return 0
        elif successful_count > 0:
            print(f"\n⚠️ ANÁLISE PARCIALMENTE CONCLUÍDA")
            print(f"📊 {successful_count}/{total_count} variáveis processadas")
            return 1
        else:
            print(f"\n❌ ANÁLISE FALHOU")
            print("Todas as variáveis falharam no processamento")
            return 2
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Análise interrompida pelo usuário")
        return 130
    except Exception as e:
        print(f"\n💥 Erro crítico: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())