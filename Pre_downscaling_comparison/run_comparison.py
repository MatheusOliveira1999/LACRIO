#!/usr/bin/env python3
"""
Script CLI para comparação de dados pré-downscaling
Análise comparativa entre dados ERA5 e estações meteorológicas
"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar módulos
try:
    from src.core.data_loader import DataLoader
    from src.core.data_comparator import DataComparator
    from src.visualization.individual_plots import IndividualPlots
    from src.reporting.report_generator import ReportGenerator
except ImportError as e:
    print(f"❌ Erro na importação dos módulos: {e}")
    print("Certifique-se de que está no diretório correto e que os módulos estão disponíveis")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Sistema de Comparação Pré-Downscaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

# Comparação básica
python run_comparison.py \\
    --era5 dados_era5.csv \\
    --station dados_estacao.csv \\
    --station-name "Llanganuco"

# Comparação com variáveis específicas
python run_comparison.py \\
    --era5 dados_era5.csv \\
    --station dados_estacao.csv \\
    --station-name "Llanganuco" \\
    --variables temperature precipitation \\
    --time-offset 6

# Comparação com relatório detalhado
python run_comparison.py \\
    --era5 dados_era5.csv \\
    --station dados_estacao.csv \\
    --station-name "Llanganuco" \\
    --generate-report \\
    --output-dir resultados
        """
    )
    
    # Argumentos obrigatórios
    parser.add_argument(
        '--era5', 
        required=True,
        help='Arquivo CSV com dados ERA5/ERA5-Land processados'
    )
    parser.add_argument(
        '--station', 
        required=True,
        help='Arquivo CSV com dados da estação meteorológica'
    )
    parser.add_argument(
        '--station-name', 
        required=True,
        help='Nome da estação meteorológica para identificação nos gráficos'
    )
    
    # Argumentos opcionais
    parser.add_argument(
        '--variables',
        nargs='+',
        default=['temperature', 'precipitation'],
        choices=['temperature', 'precipitation', 'humidity', 'pressure'],
        help='Variáveis a serem analisadas (padrão: temperature precipitation)'
    )
    
    parser.add_argument(
        '--time-offset',
        type=int,
        default=0,
        help='Offset temporal em horas para sincronização (padrão: 0)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Diretório de saída para resultados (padrão: diretório atual)'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Gerar relatório detalhado em markdown'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Não gerar gráficos (apenas estatísticas)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Modo verboso com informações detalhadas'
    )
    
    return parser.parse_args()


def load_and_validate_data(era5_path, station_path, time_offset_hours=0, verbose=False):
    """Carrega e valida os dados de entrada"""
    if verbose:
        print(f"📁 Carregando dados...")
        print(f"   ERA5: {era5_path}")
        print(f"   Estação: {station_path}")
    
    # Verificar se arquivos existem
    if not os.path.exists(era5_path):
        raise FileNotFoundError(f"Arquivo ERA5 não encontrado: {era5_path}")
    if not os.path.exists(station_path):
        raise FileNotFoundError(f"Arquivo da estação não encontrado: {station_path}")
    
    # Carregar dados
    loader = DataLoader()
    era5_df = loader.load_era5_data(era5_path)
    station_df = loader.load_station_data(station_path)
    
    if verbose:
        print(f"   ERA5: {era5_df.shape[0]} registros, {era5_df.shape[1]} colunas")
        print(f"   Estação: {station_df.shape[0]} registros, {station_df.shape[1]} colunas")
    
    # Sincronizar datasets
    era5_sync, station_sync = loader.synchronize_datasets(
        era5_df, station_df, time_offset_hours=time_offset_hours
    )
    
    if verbose:
        print(f"✅ Dados sincronizados: {era5_sync.shape[0]} registros comuns")
        common_vars = [col for col in era5_sync.columns if col in station_sync.columns and col != 'date']
        print(f"📊 Variáveis comuns: {common_vars}")
    
    return era5_sync, station_sync


def run_comparison_analysis(era5_sync, station_sync, variables, verbose=False):
    """Executa análise comparativa dos dados"""
    if verbose:
        print(f"\\n🔍 Executando análise comparativa para {len(variables)} variáveis...")
    
    comparator = DataComparator()
    results = {}
    
    for variable in variables:
        if variable in era5_sync.columns and variable in station_sync.columns:
            if verbose:
                print(f"   Analisando {variable}...")
            
            # Análise básica
            var_results = comparator.compare_variables(
                era5_sync, station_sync, variables=[variable]
            )
            results[variable] = var_results
        else:
            print(f"⚠️  Variável '{variable}' não encontrada nos dados")
    
    return results


def generate_plots(era5_sync, station_sync, variables, station_name, output_dir, verbose=False):
    """Gera gráficos de comparação"""
    if verbose:
        print(f"\\n🎨 Gerando gráficos para {len(variables)} variáveis...")
    
    # Criar diretório de saída para gráficos
    plots_dir = Path(output_dir) / "static" / "img" / station_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plotter = IndividualPlots(output_dir=str(plots_dir))
    generated_plots = []
    
    for variable in variables:
        if variable in era5_sync.columns and variable in station_sync.columns:
            if verbose:
                print(f"   Gerando gráficos para {variable}...")
            
            era5_data = era5_sync[variable]
            station_data = station_sync[variable]
            dates = era5_sync['date']
            
            try:
                # Gráficos principais
                plots = []
                
                # 1. Scatter plot
                scatter_path = plotter.scatter_density_plot(
                    era5_data, station_data, variable,
                    title=f"Dispersão ERA5 vs Observado - {variable.title()} - {station_name}"
                )
                plots.append(scatter_path)
                
                # 2. Série temporal completa
                ts_path = plotter.time_series_complete(
                    era5_data, station_data, dates, variable,
                    title=f"Série Temporal Completa - {variable.title()} - {station_name}"
                )
                plots.append(ts_path)
                
                # 3. Ciclo anual
                annual_path = plotter.annual_cycle_plot(
                    era5_sync, station_sync, variable,
                    title=f"Ciclo Anual Médio - {variable.title()} - {station_name}"
                )
                plots.append(annual_path)
                
                # 4. Distribuições
                dist_path = plotter.distributions_plot(
                    era5_data, station_data, variable,
                    title=f"Distribuições - {variable.title()} - {station_name}"
                )
                plots.append(dist_path)
                
                # 5. Análise de bias
                bias_path = plotter.bias_analysis_plot(
                    era5_data, station_data, dates, variable,
                    title=f"Evolução do Bias - {variable.title()} - {station_name}"
                )
                plots.append(bias_path)
                
                generated_plots.extend(plots)
                
                if verbose:
                    print(f"      ✅ {len(plots)} gráficos gerados para {variable}")
                    
            except Exception as e:
                print(f"❌ Erro ao gerar gráficos para {variable}: {e}")
    
    return generated_plots


def generate_report(results, station_name, output_dir, verbose=False):
    """Gera relatório detalhado"""
    if verbose:
        print(f"\\n📝 Gerando relatório detalhado...")
    
    # Criar diretório de saída para relatórios
    reports_dir = Path(output_dir) / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    generator = ReportGenerator()
    
    # Gerar relatório
    report_path = reports_dir / f"relatorio_comparacao_{station_name}.md"
    
    try:
        generator.generate_full_report(
            results, str(report_path), station_name=station_name
        )
        
        if verbose:
            print(f"   ✅ Relatório gerado: {report_path}")
        
        return str(report_path)
        
    except Exception as e:
        print(f"❌ Erro ao gerar relatório: {e}")
        return None


def print_summary(results, variables, generated_plots, report_path, verbose=False):
    """Imprime resumo dos resultados"""
    print(f"\\n🎉 ANÁLISE CONCLUÍDA!")
    print(f"=" * 50)
    
    # Resumo das variáveis analisadas
    print(f"📊 Variáveis analisadas: {len(variables)}")
    for var in variables:
        if var in results:
            var_stats = results[var].get(var, {})
            correlation = var_stats.get('correlation', {}).get('pearson', 0)
            rmse = var_stats.get('metrics', {}).get('rmse', 0)
            print(f"   • {var.title()}: Correlação = {correlation:.3f}, RMSE = {rmse:.3f}")
    
    # Resumo dos gráficos
    if generated_plots:
        print(f"\\n🎨 Gráficos gerados: {len(generated_plots)}")
        plots_dir = Path(generated_plots[0]).parent
        print(f"   📁 Local: {plots_dir}")
    
    # Resumo do relatório
    if report_path:
        print(f"\\n📝 Relatório detalhado: {report_path}")
    
    print(f"\\n✅ Comparação pré-downscaling finalizada com sucesso!")


def main():
    """Função principal"""
    print("🌍 SISTEMA DE COMPARAÇÃO PRÉ-DOWNSCALING")
    print("=" * 50)
    print("Análise comparativa ERA5 vs. Estações Meteorológicas")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # 1. Carregar e validar dados
        era5_sync, station_sync = load_and_validate_data(
            args.era5, args.station, args.time_offset, args.verbose
        )
        
        # 2. Executar análise comparativa
        results = run_comparison_analysis(
            era5_sync, station_sync, args.variables, args.verbose
        )
        
        # 3. Gerar gráficos (se solicitado)
        generated_plots = []
        if not args.no_plots:
            generated_plots = generate_plots(
                era5_sync, station_sync, args.variables, 
                args.station_name, args.output_dir, args.verbose
            )
        
        # 4. Gerar relatório (se solicitado)
        report_path = None
        if args.generate_report:
            report_path = generate_report(
                results, args.station_name, args.output_dir, args.verbose
            )
        
        # 5. Imprimir resumo
        print_summary(results, args.variables, generated_plots, report_path, args.verbose)
        
    except Exception as e:
        print(f"\\n💥 ERRO: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()