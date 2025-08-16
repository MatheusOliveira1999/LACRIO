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
        # Armazenar o caminho da estação para uso posterior
        self.station_file = station_path
        try:
            print(f"Carregando dados para {self.variable}...")
            
            # Carregar ERA5 (detectar coluna de data automaticamente)
            era5_df = pd.read_csv(era5_path)
            date_columns = ['date', 'time', 'Date', 'Time', 'datetime', 'timestamp']
            era5_date_col = None
            
            # Priorizar 'date' se já existir
            if 'date' in era5_df.columns:
                era5_date_col = 'date'
            else:
                for col in date_columns:
                    if col in era5_df.columns:
                        era5_date_col = col
                        break
                    
            if era5_date_col:
                era5_df[era5_date_col] = pd.to_datetime(era5_df[era5_date_col])
                if era5_date_col != 'date':
                    era5_df['date'] = era5_df[era5_date_col]
                    # Remover coluna original se diferente de 'date' para evitar duplicação
                    era5_df.drop(columns=[era5_date_col], inplace=True)
                
                # Remover outras colunas de data que possam causar confusão
                other_date_cols = [col for col in ['timestamp', 'time', 'datetime'] 
                                 if col in era5_df.columns and col != 'date']
                if other_date_cols:
                    era5_df.drop(columns=other_date_cols, inplace=True)
                    print(f"Removidas colunas de data extras: {other_date_cols}")
            else:
                raise ValueError("Coluna de data não encontrada no arquivo ERA5")
                
            print(f"ERA5 carregado: {era5_df.shape}")
            
            # Carregar estação (detectar coluna de data automaticamente)
            station_df = pd.read_csv(station_path)
            station_date_col = None
            
            # Priorizar 'date' se já existir
            if 'date' in station_df.columns:
                station_date_col = 'date'
            else:
                for col in date_columns + ['data']:
                    if col in station_df.columns:
                        station_date_col = col
                        break
                    
            if station_date_col:
                station_df[station_date_col] = pd.to_datetime(station_df[station_date_col])
                if station_date_col != 'date':
                    station_df['date'] = station_df[station_date_col]
                    # Remover coluna original se diferente de 'date' para evitar duplicação
                    station_df.drop(columns=[station_date_col], inplace=True)
                
                # Remover outras colunas de data que possam causar confusão
                other_date_cols = [col for col in ['timestamp', 'time', 'datetime', 'data'] 
                                 if col in station_df.columns and col != 'date']
                if other_date_cols:
                    station_df.drop(columns=other_date_cols, inplace=True)
                    print(f"Removidas colunas de data extras da estação: {other_date_cols}")
            else:
                raise ValueError("Coluna de data não encontrada no arquivo da estação")
                
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
            'Datetime': 'date',
            'DATETIME': 'date',
            'Temperatura': 'temp_obs',
            'temperatura': 'temp_obs',
            'Temperature': 'temp_obs',
            'TEMP': 'temp_obs',
            'Precipitação': 'prec_obs',
            'precipitacao': 'prec_obs',
            'Precipitation': 'prec_obs',
            'PREC': 'prec_obs',
            'Rain': 'prec_obs',
            # Mapeamento adicional para outras variáveis meteorológicas
            'Wind Speed': 'wind_speed',
            'Wind_Speed': 'wind_speed',
            'Wind Direction': 'wind_direction',
            'Wind_Direction': 'wind_direction',
            'Dew Point': 'dew_point',
            'Dew_Point': 'dew_point',
            'RH': 'relative_humidity',
            'Relative_Humidity': 'relative_humidity',
            'Solar': 'solar_radiation',
            'Solar_Radiation': 'solar_radiation'
        }
        
        station_df = station_df.rename(columns=column_mapping)
        
        # Debug: mostrar colunas após mapeamento inicial
        print(f"🔍 Colunas após mapeamento: {list(station_df.columns)}")
        
        # Garantir que as colunas principais existam
        if 'temp_obs' not in station_df.columns:
            # Procurar por colunas de temperatura que não foram mapeadas (busca mais abrangente)
            temp_cols = [col for col in station_df.columns if any(term in col.lower() for term in ['temp', 'temperature', 'temperatura'])]
            print(f"🔍 Colunas de temperatura encontradas: {temp_cols}")
            if temp_cols:
                station_df['temp_obs'] = station_df[temp_cols[0]]
                print(f"✅ Usando coluna {temp_cols[0]} para temp_obs")
            else:
                print("❌ Nenhuma coluna de temperatura encontrada, usando NaN")
                station_df['temp_obs'] = np.nan
                
        if 'prec_obs' not in station_df.columns:
            # Procurar por colunas de precipitação que não foram mapeadas (busca mais abrangente)
            prec_cols = [col for col in station_df.columns if any(term in col.lower() for term in ['prec', 'rain', 'precipitation', 'precipita', 'chuva'])]
            print(f"🔍 Colunas de precipitação encontradas: {prec_cols}")
            if prec_cols:
                station_df['prec_obs'] = station_df[prec_cols[0]]
                print(f"✅ Usando coluna {prec_cols[0]} para prec_obs")
            else:
                print("❌ Nenhuma coluna de precipitação encontrada, usando 0.0")
                station_df['prec_obs'] = 0.0
        
        # Debug: verificar valores das colunas observacionais
        if 'temp_obs' in station_df.columns:
            print(f"🔍 Estatísticas temp_obs: min={station_df['temp_obs'].min():.2f}, max={station_df['temp_obs'].max():.2f}, mean={station_df['temp_obs'].mean():.2f}")
            print(f"🔍 Valores NaN em temp_obs: {station_df['temp_obs'].isna().sum()}")
            
        if 'prec_obs' in station_df.columns:
            print(f"🔍 Estatísticas prec_obs: min={station_df['prec_obs'].min():.2f}, max={station_df['prec_obs'].max():.2f}, mean={station_df['prec_obs'].mean():.2f}")
            print(f"🔍 Valores NaN em prec_obs: {station_df['prec_obs'].isna().sum()}")
        
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
        
        target_col = 'temp_obs' if self.variable in ['temperature', 'temperatura'] else 'prec_obs'
        if target_col not in data.columns:
            raise ValueError(f"Coluna alvo '{target_col}' não encontrada")
        
        valid_data = data[target_col].notna().sum()
        if valid_data < min_records:
            raise ValueError(f"Dados válidos insuficientes: {valid_data} registros")
        
        print(f"Validação concluída: {valid_data} registros válidos")