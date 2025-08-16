#!/usr/bin/env python3
"""
Script para gerar TODOS os gráficos INDIVIDUAIS separados
Cada análise em seu próprio arquivo, sem combinações
"""

import sys
import os
import pandas as pd
import numpy as np

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar módulos
try:
    from src.core.data_loader import DataLoader
    from src.visualization.individual_plots import IndividualPlots
    print("✅ Módulos importados com sucesso!")
except ImportError as e:
    print(f"❌ Erro na importação: {e}")
    sys.exit(1)

def load_and_sync_data():
    """Carrega e sincroniza os dados"""
    print("\n📁 Carregando dados...")
    
    loader = DataLoader()
    
    # Caminhos dos arquivos
    era5_path = "/home/matheus/Documentos/GitHub/LACRIO/Pre_downscaling_comparison/uploads/dados_era5_horario_CasaDeAgua_processado.csv"
    station_path = "/home/matheus/Documentos/GitHub/LACRIO/Pre_downscaling_comparison/uploads/CasaDeAgua_horario_processado.csv"
    
    if not os.path.exists(era5_path) or not os.path.exists(station_path):
        print("❌ Arquivos de dados não encontrados")
        return None, None, None
    
    # Carregar dados
    era5_df = loader.load_era5_data(era5_path)
    station_df = loader.load_station_data(station_path)
    
    # Sincronizar com offset temporal de 6 horas
    era5_sync, station_sync = loader.synchronize_datasets(era5_df, station_df, time_offset_hours=6)
    
    print(f"✅ Dados sincronizados: {era5_sync.shape[0]} registros")
    print(f"📊 Variáveis comuns: {[col for col in era5_sync.columns if col != 'date']}")
    
    # DEBUG: Verificar se os dados são idênticos (possível problema na sincronização)
    for var in ['temperature', 'precipitation']:
        if var in era5_sync.columns and var in station_sync.columns:
            era5_values = era5_sync[var].dropna()
            station_values = station_sync[var].dropna()
            
            if len(era5_values) > 0 and len(station_values) > 0:
                # Verificar se há alinhamento correto
                common_idx = era5_sync[var].notna() & station_sync[var].notna()
                era5_common = era5_sync.loc[common_idx, var]
                station_common = station_sync.loc[common_idx, var]
                
                print(f"🔍 DEBUG Sincronização - {var}:")
                print(f"    ERA5: {len(era5_common)} valores, exemplo: {era5_common.iloc[:3].tolist() if len(era5_common) >= 3 else era5_common.tolist()}")
                print(f"    Estação: {len(station_common)} valores, exemplo: {station_common.iloc[:3].tolist() if len(station_common) >= 3 else station_common.tolist()}")
                
                if len(era5_common) > 0 and len(station_common) > 0:
                    simple_corr = era5_common.corr(station_common)
                    print(f"    Correlação simples: {simple_corr:.6f}")
                    
                    # Verificar se são idênticos
                    if np.allclose(era5_common, station_common, rtol=1e-10):
                        print(f"    ⚠️  PROBLEMA: Dados são praticamente idênticos!")
    print("---")
    
    return era5_sync, station_sync, station_path

def generate_all_individual_plots():
    """Gera TODOS os gráficos individuais separados"""
    print("\n🎨 Gerando TODOS os gráficos individuais...")
    
    # Carregar dados
    era5_sync, station_sync, station_path = load_and_sync_data()
    
    if era5_sync is None or station_sync is None or station_path is None:
        print("❌ Não foi possível carregar os dados")
        return []
    
    # Extrair automaticamente o primeiro nome da estação do caminho atual do arquivo
    station_filename = os.path.basename(station_path)
    station_first_name = station_filename.split('_')[0]  # Extrai o primeiro nome antes do primeiro underscore
    
    # Criar diretório específico para a estação
    station_dir = f"static/img/{station_first_name}"
    os.makedirs(station_dir, exist_ok=True)
    print(f"📁 Criando pasta para estação: {station_first_name}")
    print(f"📁 Salvando gráficos em: {station_dir}/")
    
    # Inicializar plotador com diretório específico da estação
    plotter = IndividualPlots(output_dir=f"./{station_dir}")
    
    # Variáveis para analisar
    variables = ['temperature', 'precipitation']
    
    generated_plots = []
    total_plots = 0
    
    for variable in variables:
        if variable in era5_sync.columns and variable in station_sync.columns:
            print(f"\\n📈 Gerando gráficos individuais para {variable.upper()}...")
            
            era5_data = era5_sync[variable]
            station_data = station_sync[variable]
            dates = era5_sync['date']
            
            plots_for_var = []
            
            try:
                # 1. SCATTER PLOT COM DENSIDADE
                print(f"  🔍 1. Scatter plot com densidade...")
                scatter_path = plotter.scatter_density_plot(
                    era5_data, station_data, variable,
                    title=f"Dispersão ERA5 vs Observado - {variable.title()} - {station_first_name}"
                )
                plots_for_var.append(scatter_path)
                print(f"      ✅ {os.path.basename(scatter_path)}")
                
                # 2. SÉRIE TEMPORAL - ÚLTIMOS 90 DIAS
                print(f"  📅 2. Série temporal (últimos 90 dias)...")
                ts_recent_path = plotter.time_series_recent(
                    era5_data, station_data, dates, variable, n_days=90,
                    title=f"Série Temporal (90 dias) - {variable.title()} - {station_first_name}"
                )
                plots_for_var.append(ts_recent_path)
                print(f"      ✅ {os.path.basename(ts_recent_path)}")
                
                # 3. SÉRIE TEMPORAL - COMPLETA
                print(f"  📊 3. Série temporal completa...")
                ts_complete_path = plotter.time_series_complete(
                    era5_data, station_data, dates, variable,
                    title=f"Série Temporal Completa - {variable.title()} - {station_first_name}"
                )
                plots_for_var.append(ts_complete_path)
                print(f"      ✅ {os.path.basename(ts_complete_path)}")
                
                # 4. CICLO ANUAL MÉDIO
                print(f"  🔄 4. Ciclo anual médio...")
                annual_path = plotter.annual_cycle_plot(
                    era5_sync, station_sync, variable,
                    title=f"Ciclo Anual Médio - {variable.title()} - {station_first_name}"
                )
                plots_for_var.append(annual_path)
                print(f"      ✅ {os.path.basename(annual_path)}")
                
                # 5. DISTRIBUIÇÕES (HISTOGRAMAS)
                print(f"  📈 5. Distribuições...")
                dist_path = plotter.distributions_plot(
                    era5_data, station_data, variable,
                    title=f"Distribuições - {variable.title()} - {station_first_name}"
                )
                plots_for_var.append(dist_path)
                print(f"      ✅ {os.path.basename(dist_path)}")
                
                # 6. BOXPLOTS MENSAIS
                print(f"  📦 6. Boxplots mensais...")
                box_path = plotter.boxplot_monthly(
                    era5_sync, station_sync, variable,
                    title=f"Variabilidade Mensal - {variable.title()} - {station_first_name}"
                )
                plots_for_var.append(box_path)
                print(f"      ✅ {os.path.basename(box_path)}")
                
                # 7. Q-Q PLOT
                print(f"  🎯 7. Q-Q Plot...")
                qq_path = plotter.qq_plot(
                    era5_data, station_data, variable,
                    title=f"Q-Q Plot - {variable.title()} - {station_first_name}"
                )
                plots_for_var.append(qq_path)
                print(f"      ✅ {os.path.basename(qq_path)}")
                
                # 8. ANÁLISE DE BIAS TEMPORAL
                print(f"  ⚖️  8. Análise de bias temporal...")
                bias_path = plotter.bias_analysis_plot(
                    era5_data, station_data, dates, variable,
                    title=f"Evolução do Bias - {variable.title()} - {station_first_name}"
                )
                plots_for_var.append(bias_path)
                print(f"      ✅ {os.path.basename(bias_path)}")
                
                
                generated_plots.extend(plots_for_var)
                total_plots += len(plots_for_var)
                
                print(f"    📊 Total para {variable}: {len(plots_for_var)} gráficos")
                
            except Exception as e:
                print(f"  ❌ Erro ao gerar gráficos para {variable}: {e}")
                import traceback
                traceback.print_exc()
    
    
    # Resumo final
    print(f"\\n🎉 GERAÇÃO CONCLUÍDA!")
    print(f"📁 Total de gráficos individuais gerados: {total_plots}")
    print(f"📊 Total de variáveis processadas: {len(variables)}")
    print(f"📈 Gráficos por variável: {total_plots // len(variables) if variables else 0}")
    
    # Listar todos os arquivos gerados
    print(f"\\n📋 LISTA DE GRÁFICOS GERADOS:")
    for i, plot_path in enumerate(generated_plots, 1):
        if os.path.exists(plot_path):
            size_mb = os.path.getsize(plot_path) / (1024*1024)
            print(f"  {i:2d}. {os.path.basename(plot_path)} ({size_mb:.2f} MB)")
        else:
            print(f"  {i:2d}. ❌ {os.path.basename(plot_path)} - arquivo não encontrado")
    
    return generated_plots

def main():
    """Função principal"""
    print("🚀 GERADOR DE GRÁFICOS INDIVIDUAIS")
    print("=" * 50)
    print("📝 Cada análise será salva em arquivo separado")
    print("🎯 Comparação: ERA5 vs Estação (Observado)")
    print("📊 Tipos de gráficos:")
    print("   1. Scatter plot com densidade")
    print("   2. Série temporal (90 dias)")
    print("   3. Série temporal completa")
    print("   4. Ciclo anual médio")
    print("   5. Distribuições (histogramas)")
    print("   6. Boxplots mensais")
    print("   7. Q-Q Plot")
    print("   8. Análise de bias temporal")
    print("=" * 50)
    
    try:
        plots = generate_all_individual_plots()
        
        if plots:
            print("\\n✅ SCRIPT EXECUTADO COM SUCESSO!")
            print("🔍 Todos os gráficos estão na pasta static/img/")
            print("📁 Cada análise foi salva em arquivo individual")
        else:
            print("\\n❌ NENHUM GRÁFICO FOI GERADO")
            
    except Exception as e:
        print(f"\\n💥 ERRO GERAL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()