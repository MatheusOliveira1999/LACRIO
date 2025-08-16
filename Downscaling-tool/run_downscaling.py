#!/usr/bin/env python3
"""
CLI para Sistema de Downscaling Climático
Executa downscaling estatístico usando Machine Learning
"""

import os
import sys
import argparse
from pathlib import Path

# Adicionar src ao path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.utils import run_downscaling_pipeline_advanced


def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Downscaling Climático com Machine Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

1. Uso básico:
   python run_downscaling.py --era5 dados_era5.csv --station dados_estacao.csv --dem modelo_elevacao.nc --lat -9.41 --lon -77.35

2. Com configurações avançadas:
   python run_downscaling.py --era5 dados_era5.csv --station dados_estacao.csv --dem modelo_elevacao.nc --lat -9.41 --lon -77.35 --models RandomForest XGBoost --variables temperature precipitation --split-date 2018-01-01 --optimize
        """
    )
    
    # Argumentos obrigatórios
    required = parser.add_argument_group('argumentos obrigatórios')
    required.add_argument('--era5', required=True, help='Caminho para arquivo ERA5/ERA5-Land (CSV)')
    required.add_argument('--station', required=True, help='Caminho para dados da estação (CSV)')
    required.add_argument('--dem', required=True, help='Caminho para modelo de elevação digital (NetCDF)')
    required.add_argument('--lat', type=float, required=True, help='Latitude da estação meteorológica')
    required.add_argument('--lon', type=float, required=True, help='Longitude da estação meteorológica')
    
    # Argumentos opcionais
    parser.add_argument('--variables', nargs='+', default=['temperature', 'precipitation'],
                       choices=['temperature', 'precipitation'],
                       help='Variáveis para downscaling (default: temperature precipitation)')
    
    parser.add_argument('--models', nargs='+', 
                       default=['LinearRegression', 'Ridge', 'RandomForest', 'ExtraTrees', 
                               'GradientBoosting', 'XGBoost', 'SVR', 'MLP'],
                       choices=['LinearRegression', 'Ridge', 'RandomForest', 'ExtraTrees', 
                               'GradientBoosting', 'XGBoost', 'SVR', 'MLP'],
                       help='Modelos ML para treinar (default: todos os 8 modelos)')
    
    parser.add_argument('--split-date', default='2018-01-01',
                       help='Data para divisão treino/teste (YYYY-MM-DD) (default: 2018-01-01)')
    
    parser.add_argument('--no-ensemble', action='store_true',
                       help='Não criar modelo ensemble')
    
    parser.add_argument('--optimize', action='store_true',
                       help='Ativar otimização de hiperparâmetros (mais lento)')
    
    parser.add_argument('--no-validation', action='store_true',
                       help='Desativar validação detalhada')
    
    parser.add_argument('--output-dir', default='.',
                       help='Diretório para salvar resultados (default: diretório atual)')
    
    args = parser.parse_args()
    
    # Validar arquivos de entrada
    for file_path, name in [(args.era5, 'ERA5'), (args.station, 'Station'), (args.dem, 'DEM')]:
        if not os.path.exists(file_path):
            print(f"❌ Erro: Arquivo {name} não encontrado: {file_path}")
            sys.exit(1)
    
    # Configurar diretório de saída
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True)
    os.chdir(output_dir)
    
    print("🌍 Sistema de Downscaling Climático")
    print("=" * 50)
    print(f"📂 ERA5: {args.era5}")
    print(f"📂 Estação: {args.station}")
    print(f"📂 DEM: {args.dem}")
    print(f"📍 Coordenadas: {args.lat}, {args.lon}")
    print(f"📊 Variáveis: {', '.join(args.variables)}")
    print(f"🤖 Modelos: {', '.join(args.models)}")
    print(f"📅 Data divisão: {args.split_date}")
    print(f"🔧 Otimização: {'Sim' if args.optimize else 'Não'}")
    print(f"🎯 Ensemble: {'Não' if args.no_ensemble else 'Sim'}")
    print(f"✅ Validação: {'Não' if args.no_validation else 'Sim'}")
    print(f"💾 Saída: {output_dir}")
    print("=" * 50)
    
    try:
        # Preparar arquivos
        files = {
            'era5': args.era5,
            'station': args.station,
            'dem': args.dem
        }
        
        # Parâmetros
        params = {
            'latitude': args.lat,
            'longitude': args.lon,
            'split_date': args.split_date
        }
        
        # Configurações
        use_ensemble = not args.no_ensemble
        enable_validation = not args.no_validation
        
        print("\n🚀 Iniciando processamento...")
        
        # Executar downscaling
        results = run_downscaling_pipeline_advanced(
            files=files,
            params=params,
            selected_models=args.models,
            variables=args.variables,
            use_ensemble=use_ensemble,
            optimize_params=args.optimize,
            opt_config={},
            temporal_resolution='auto',
            enable_validation=enable_validation
        )
        
        print("\n" + "=" * 50)
        print("🎉 PROCESSAMENTO CONCLUÍDO!")
        print("=" * 50)
        
        # Resumo dos resultados
        for variable, result in results.items():
            if 'error' in result:
                print(f"❌ {variable.upper()}: {result['error']}")
            else:
                best_model = result.get('best_model', 'N/A')
                n_models = len(result.get('models', {}))
                print(f"✅ {variable.upper()}: {n_models} modelos treinados, melhor: {best_model}")
        
        print(f"\n📁 Resultados salvos em:")
        print(f"   📊 Relatórios: {output_dir}/results/")
        print(f"   🤖 Modelos: {output_dir}/models/")
        print(f"   📈 Gráficos: {output_dir}/static/img/")
        
    except Exception as e:
        print(f"\n❌ Erro durante o processamento:")
        print(f"   {str(e)}")
        print(f"\n💡 Verifique:")
        print(f"   - Formato dos arquivos de entrada")
        print(f"   - Coordenadas válidas")
        print(f"   - Dependências instaladas (pip install -r requirements.txt)")
        sys.exit(1)


if __name__ == "__main__":
    main()