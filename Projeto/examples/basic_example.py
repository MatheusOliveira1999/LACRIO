#!/usr/bin/env python3
"""
Exemplo básico de uso do sistema de downscaling
"""

import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.downscaling_model import WeatherDownscalingModel


def main():
    """Exemplo básico"""
    
    # Caminhos dos arquivos (ajustar conforme necessário)
    era5_path = "dados_era5_era5land.csv"
    station_path = "dados_estacao_cuchillacocha.csv"
    dem_path = "Quilcayhuanca_static.nc"
    
    # Coordenadas da estação
    lat = -9.41
    lon = -77.35
    
    # Verificar se arquivos existem
    files = [era5_path, station_path, dem_path]
    missing_files = [f for f in files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Arquivos não encontrados:")
        for f in missing_files:
            print(f"  - {f}")
        print("💡 Ajuste os caminhos dos arquivos neste script")
        return 1
    
    print("🚀 Executando exemplo de downscaling...")
    
    try:
        # Criar modelo para temperatura
        model = WeatherDownscalingModel('temperature')
        
        # Pipeline completo
        print("📊 Carregando dados...")
        model.load_and_merge_data(era5_path, station_path, dem_path, lat, lon)
        
        print("🔧 Criando features...")
        model.create_features()
        
        print("📈 Preparando dados...")
        model.prepare_data('2018-01-01')
        
        print("🤖 Treinando modelos...")
        # Usar apenas alguns modelos para o exemplo
        selected_models = ['LinearRegression', 'Ridge', 'RandomForest']
        model.train_models(selected_models=selected_models, optimize=False)
        
        print("🔗 Criando ensemble...")
        model.create_ensemble()
        
        print("💾 Salvando resultados...")
        model.save_models()
        
        # Mostrar resultados
        best_model = model.best_model
        best_metrics = model.results[best_model]['metrics']
        
        print(f"\n🏆 RESULTADOS:")
        print(f"   Melhor modelo: {best_model}")
        print(f"   RMSE: {best_metrics['RMSE']:.4f}°C")
        print(f"   R²: {best_metrics['R2']:.4f}")
        
        print("\n✅ Exemplo concluído com sucesso!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
