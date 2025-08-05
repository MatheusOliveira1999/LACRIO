"""
Módulo para carregamento de dados
"""

import pandas as pd
import xarray as xr
import numpy as np
from typing import Tuple, Dict, Any
import os


class DataLoader:
    """Classe para carregar dados ERA5, estação e DEM"""
    
    def __init__(self):
        self.era5_data = None
        self.station_data = None
        self.dem_data = None
    
    def load_era5_data(self, era5_path: str) -> pd.DataFrame:
        """
        Carrega dados ERA5/ERA5-Land
        
        Parameters:
        -----------
        era5_path : str
            Caminho para arquivo CSV com dados ERA5
            
        Returns:
        --------
        pd.DataFrame : Dados ERA5 carregados
        """
        try:
            print(f"Carregando dados ERA5: {era5_path}")
            
            if not os.path.exists(era5_path):
                raise FileNotFoundError(f"Arquivo ERA5 não encontrado: {era5_path}")
            
            era5_df = pd.read_csv(era5_path, parse_dates=['date'])
            
            if era5_df.empty:
                raise ValueError("Arquivo ERA5 está vazio")
            
            # Validar colunas essenciais
            required_cols = ['date']
            missing_cols = [col for col in required_cols if col not in era5_df.columns]
            if missing_cols:
                raise ValueError(f"Colunas obrigatórias faltando no ERA5: {missing_cols}")
            
            print(f"ERA5 carregado: {era5_df.shape[0]} registros, {era5_df.shape[1]} variáveis")
            
            self.era5_data = era5_df
            return era5_df
            
        except Exception as e:
            print(f"Erro ao carregar ERA5: {str(e)}")
            raise
    
    def load_station_data(self, station_path: str) -> pd.DataFrame:
        """
        Carrega dados da estação meteorológica
        
        Parameters:
        -----------
        station_path : str
            Caminho para arquivo CSV com dados da estação
            
        Returns:
        --------
        pd.DataFrame : Dados da estação carregados
        """
        try:
            print(f"Carregando dados da estação: {station_path}")
            
            if not os.path.exists(station_path):
                raise FileNotFoundError(f"Arquivo da estação não encontrado: {station_path}")
            
            station_df = pd.read_csv(station_path, parse_dates=['data'])
            
            if station_df.empty:
                raise ValueError("Arquivo da estação está vazio")
            
            # Padronizar nomes das colunas
            station_df = self._standardize_station_columns(station_df)
            
            print(f"Estação carregada: {station_df.shape[0]} registros")
            
            self.station_data = station_df
            return station_df
            
        except Exception as e:
            print(f"Erro ao carregar dados da estação: {str(e)}")
            raise
    
    def load_dem_data(self, dem_path: str, lat: float, lon: float) -> Dict[str, float]:
        """
        Carrega modelo de elevação digital
        
        Parameters:
        -----------
        dem_path : str
            Caminho para arquivo NetCDF com DEM
        lat, lon : float
            Coordenadas da estação
            
        Returns:
        --------
        dict : Informações de elevação
        """
        try:
            print(f"Carregando DEM: {dem_path}")
            
            if not os.path.exists(dem_path):
                raise FileNotFoundError(f"Arquivo DEM não encontrado: {dem_path}")
            
            dem = xr.open_dataset(dem_path)
            
            # Buscar possíveis nomes para elevação
            elevation_vars = ['HGT', 'elevation', 'dem', 'z', 'height']
            elevation_var = None
            
            for var in elevation_vars:
                if var in dem.variables:
                    elevation_var = var
                    break
            
            if elevation_var is None:
                print("⚠️ Variável de elevação não encontrada. Usando elevação padrão.")
                elev_local = 3000  # Elevação padrão para regiões montanhosas
            else:
                elev_local = float(dem[elevation_var].sel(lat=lat, lon=lon, method='nearest'))
            
            dem_info = {
                'elevation': elev_local,
                'variable_used': elevation_var,
                'coordinates': (lat, lon)
            }
            
            print(f"Elevação local: {elev_local:.1f}m")
            
            self.dem_data = dem_info
            return dem_info
            
        except Exception as e:
            print(f"Erro ao carregar DEM: {str(e)}")
            # Retornar valores padrão em caso de erro
            dem_info = {
                'elevation': 3000,
                'variable_used': None,
                'coordinates': (lat, lon)
            }
            self.dem_data = dem_info
            return dem_info
    
    def _standardize_station_columns(self, station_df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza nomes das colunas da estação"""
        # Mapeamento de possíveis nomes de colunas
        column_mapping = {
            'data': 'date',
            'Data': 'date',
            'DATE': 'date',
            'Temperatura': 'temp_obs',
            'temperatura': 'temp_obs',
            'Temperature': 'temp_obs',
            'TEMP': 'temp_obs',
            'Precipitação': 'prec_obs',
            'precipitacao': 'prec_obs',
            'Precipitation': 'prec_obs',
            'PREC': 'prec_obs',
            'Rain': 'prec_obs'
        }
        
        # Renomear colunas
        station_df = station_df.rename(columns=column_mapping)
        
        # Garantir que temos as colunas necessárias
        if 'temp_obs' not in station_df.columns:
            station_df['temp_obs'] = np.nan
        if 'prec_obs' not in station_df.columns:
            station_df['prec_obs'] = 0.0
            
        return station_df
    
    def merge_data(self) -> pd.DataFrame:
        """
        Mescla dados ERA5 e estação
        
        Returns:
        --------
        pd.DataFrame : Dados mesclados
        """
        if self.era5_data is None or self.station_data is None:
            raise ValueError("Dados ERA5 e/ou estação não foram carregados")
        
        try:
            # Mesclar dados
            merged_data = pd.merge(self.era5_data, self.station_data, on='date', how='inner')
            
            if merged_data.empty:
                raise ValueError("Nenhum dado encontrado após a mesclagem. Verifique as datas.")
            
            # Adicionar informações de elevação
            if self.dem_data:
                merged_data['elevation'] = self.dem_data['elevation']
                
                # Calcular diferença de altitude com ERA5 se disponível
                if 'z' in merged_data.columns:
                    merged_data['alt_ERA5'] = merged_data['z'] / 9.80665  # Conversão geopotencial
                    merged_data['alt_diff'] = self.dem_data['elevation'] - merged_data['alt_ERA5']
                else:
                    merged_data['alt_diff'] = 0
            else:
                merged_data['elevation'] = 3000
                merged_data['alt_diff'] = 0
            
            print(f"Dados mesclados: {merged_data.shape}")
            return merged_data
            
        except Exception as e:
            print(f"Erro ao mesclar dados: {str(e)}")
            raise
    
    def load_all_data(self, era5_path: str, station_path: str, dem_path: str, 
                      lat: float, lon: float) -> pd.DataFrame:
        """
        Carrega todos os dados de uma vez
        
        Parameters:
        -----------
        era5_path : str
            Caminho para dados ERA5
        station_path : str  
            Caminho para dados da estação
        dem_path : str
            Caminho para DEM
        lat, lon : float
            Coordenadas da estação
            
        Returns:
        --------
        pd.DataFrame : Dados mesclados
        """
        print("🔄 Carregando todos os dados...")
        
        # Carregar cada fonte de dados
        self.load_era5_data(era5_path)
        self.load_station_data(station_path)
        self.load_dem_data(dem_path, lat, lon)
        
        # Mesclar dados
        merged_data = self.merge_data()
        
        print("✅ Todos os dados carregados com sucesso!")
        return merged_data


def validate_data_files(era5_path: str, station_path: str, dem_path: str) -> Dict[str, bool]:
    """
    Valida se os arquivos de dados existem e são acessíveis
    
    Parameters:
    -----------
    era5_path : str
        Caminho para arquivo ERA5
    station_path : str
        Caminho para arquivo da estação
    dem_path : str
        Caminho para arquivo DEM
        
    Returns:
    --------
    dict : Status de validação para cada arquivo
    """
    validation_results = {}
    
    files_to_check = {
        'era5': era5_path,
        'station': station_path,
        'dem': dem_path
    }
    
    for file_type, file_path in files_to_check.items():
        try:
            if os.path.exists(file_path):
                if os.path.getsize(file_path) > 0:
                    validation_results[file_type] = True
                else:
                    validation_results[file_type] = False
                    print(f"⚠️ Arquivo {file_type} está vazio: {file_path}")
            else:
                validation_results[file_type] = False
                print(f"❌ Arquivo {file_type} não encontrado: {file_path}")
        except Exception as e:
            validation_results[file_type] = False
            print(f"❌ Erro ao verificar arquivo {file_type}: {str(e)}")
    
    return validation_results


if __name__ == "__main__":
    # Teste básico
    loader = DataLoader()
    print("DataLoader inicializado com sucesso!")