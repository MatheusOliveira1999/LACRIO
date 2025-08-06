"""
data_processor.py
Módulo para carregar e processar dados climáticos
"""

import numpy as np
import pandas as pd
import xarray as xr


class DataProcessor:
    """Processa e carrega dados de diferentes fontes"""
    
    def __init__(self, variable='temperature'):
        self.variable = variable
    
    def load_and_merge_data(self, era5_path, station_path, dem_path, lat, lon):
        """
        Carrega e mescla todos os dados necessários
        
        Parameters:
        -----------
        era5_path : str
            Caminho para dados ERA5/ERA5-Land
        station_path : str
            Caminho para dados da estação
        dem_path : str
            Caminho para modelo de elevação digital
        lat, lon : float
            Coordenadas da estação
        """
        try:
            print(f"Carregando dados para {self.variable}...")
            
            # Carregar ERA5
            era5_df = pd.read_csv(era5_path, parse_dates=['date'])
            print(f"ERA5 carregado: {era5_df.shape}")
            
            # Carregar estação
            station_df = pd.read_csv(station_path, parse_dates=['data'])
            print(f"Estação carregada: {station_df.shape}")
            
            # Padronizar nomes das colunas da estação
            station_df = self._standardize_station_columns(station_df)
            
            # Mesclar dados
            data = pd.merge(era5_df, station_df, on='date', how='inner')
            print(f"Dados mesclados: {data.shape}")
            
            if data.empty:
                raise ValueError("Nenhum dado encontrado após a mesclagem. Verifique as datas.")
            
            # Carregar e processar DEM
            data = self._process_elevation_data(data, dem_path, lat, lon)
            
            # Validar dados
            self._validate_data(data)
            
            print(f"Dados carregados com sucesso: {data.shape}")
            return data
            
        except Exception as e:
            print(f"Erro ao carregar dados: {str(e)}")
            raise
    
    def _standardize_station_columns(self, station_df):
        """Padroniza nomes das colunas da estação"""
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
        
        station_df = station_df.rename(columns=column_mapping)
        
        if 'temp_obs' not in station_df.columns:
            station_df['temp_obs'] = np.nan
        if 'prec_obs' not in station_df.columns:
            station_df['prec_obs'] = 0.0
            
        return station_df
    
    def _process_elevation_data(self, data, dem_path, lat, lon):
        """Processa dados de elevação"""
        try:
            dem = xr.open_dataset(dem_path)
            
            elevation_vars = ['HGT', 'elevation', 'dem', 'z', 'height']
            elevation_var = None
            
            for var in elevation_vars:
                if var in dem.variables:
                    elevation_var = var
                    break
            
            if elevation_var is None:
                print("Variável de elevação não encontrada. Usando elevação padrão.")
                elev_local = 3000
            else:
                elev_local = float(dem[elevation_var].sel(lat=lat, lon=lon, method='nearest'))
            
            if 'z' in data.columns:
                data['alt_ERA5'] = data['z'] / 9.80665
                data['alt_diff'] = elev_local - data['alt_ERA5']
            else:
                data['alt_diff'] = 0
            
            data['elevation'] = elev_local
            print(f"Elevação local: {elev_local:.1f}m")
            
        except Exception as e:
            print(f"Erro ao processar elevação: {str(e)}")
            data['alt_diff'] = 0
            data['elevation'] = 3000
        
        return data
    
    def _validate_data(self, data):
        """Valida os dados carregados"""
        min_records = 365
        if len(data) < min_records:
            raise ValueError(f"Dados insuficientes: {len(data)} registros (mínimo: {min_records})")
        
        target_col = 'temp_obs' if self.variable == 'temperature' else 'prec_obs'
        if target_col not in data.columns:
            raise ValueError(f"Coluna alvo '{target_col}' não encontrada")
        
        valid_data = data[target_col].notna().sum()
        if valid_data < min_records:
            raise ValueError(f"Dados válidos insuficientes: {valid_data} registros")
        
        print(f"Validação concluída: {valid_data} registros válidos")