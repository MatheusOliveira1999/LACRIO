import xarray as xr
import os
import pandas as pd
import argparse
from pathlib import Path
import warnings

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore')

def iniciar_processamento(era5_dir, era5_land_dir, station_file, lat, lon):
    """
    Função principal que orquestra o processamento dos dados
    ERA5 e da estação meteorológica.
    """
    try:
        # --- 1. Define e cria o diretório de saída ---
        output_dir = Path("dados_csv")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Arquivos CSV serão salvos em: '{output_dir.resolve()}'")
        # --- 2. Processamento dos dados da estação ---
        print("\nIniciando processamento dos dados da estação...")
        
        # Verifica se o arquivo existe
        if not os.path.exists(station_file):
            raise FileNotFoundError(f"Arquivo da estação não encontrado: {station_file}")
        
        # Tenta diferentes encodings
        encodings = ['latin1', 'utf-8', 'cp1252', 'iso-8859-1']
        df_estacao = None
        
        for encoding in encodings:
            try:
                df_estacao = pd.read_csv(station_file, encoding=encoding, on_bad_lines='skip')
                print(f"Arquivo lido com encoding: {encoding}")
                break
            except Exception as e:
                print(f"Falha com encoding {encoding}: {e}")
                continue
        
        if df_estacao is None:
            raise ValueError("Não foi possível ler o arquivo da estação com nenhum encoding testado")
        
        print(f"Colunas encontradas na estação: {list(df_estacao.columns)}")
        
        # Verifica se as colunas necessárias existem
        required_cols = ["Datetime", "Temperature (°C)", "Precipitation (mm)"]
        missing_cols = [col for col in required_cols if col not in df_estacao.columns]
        
        if missing_cols:
            print(f"Colunas disponíveis: {list(df_estacao.columns)}")
            raise ValueError(f"Colunas necessárias não encontradas: {missing_cols}")
        
        # Processa temperatura
        df_temperatura = df_estacao[["Datetime", "Temperature (°C)"]].copy()
        df_temperatura["Datetime"] = pd.to_datetime(df_temperatura["Datetime"], errors='coerce')
        df_temperatura = df_temperatura.dropna()
        
        # Processa precipitação
        df_precipitacao = df_estacao[["Datetime", "Precipitation (mm)"]].copy()
        df_precipitacao["Datetime"] = pd.to_datetime(df_precipitacao["Datetime"], errors='coerce')
        df_precipitacao = df_precipitacao.dropna()
        
        if df_temperatura.empty or df_precipitacao.empty:
            raise ValueError("Nenhum dado válido de temperatura ou precipitação encontrado")
        
        # Agregação diária
        media_diaria_temperatura = df_temperatura.set_index('Datetime')['Temperature (°C)'].resample('D').mean()
        soma_diaria_precipitacao = df_precipitacao.set_index('Datetime')['Precipitation (mm)'].resample('D').sum()

        df_temp_final = pd.DataFrame({
            'data': media_diaria_temperatura.index, 
            'Temperatura': media_diaria_temperatura.values
        })
        df_precip_final = pd.DataFrame({
            'data': soma_diaria_precipitacao.index, 
            'Precipitação': soma_diaria_precipitacao.values
        })
        
        df_estacao_final = pd.merge(df_precip_final, df_temp_final, on='data', how='outer')
        df_estacao_final = df_estacao_final.dropna(subset=['Temperatura', 'Precipitação'])
        
        if df_estacao_final.empty:
            raise ValueError("Nenhum dado válido de estação após processamento")
        
        station_input_path = Path(station_file)
        station_output_name = f"{station_input_path.stem}_processado.csv"
        output_path_station = output_dir / station_output_name
        
        df_estacao_final.to_csv(output_path_station, index=False)
        print(f"Dados da estação processados e salvos em: {output_path_station}")

        # --- 3. Processamento dos dados ERA5 e ERA5-Land ---
        print("\nIniciando processamento dos dados ERA5 e ERA5-Land...")
        
        # Carrega os datasets ERA5-Land
        arquivos_nc_land = sorted([os.path.join(era5_land_dir, f) for f in os.listdir(era5_land_dir) if f.endswith(".nc")])
        if not arquivos_nc_land:
            raise FileNotFoundError(f"Nenhum arquivo .nc encontrado em {era5_land_dir}")
        
        print(f"Carregando {len(arquivos_nc_land)} arquivos ERA5-Land...")
        ds_era5_land = xr.open_mfdataset(arquivos_nc_land, combine='by_coords')
        
        # Carrega os datasets ERA5
        arquivos_nc = sorted([os.path.join(era5_dir, f) for f in os.listdir(era5_dir) if f.endswith(".nc")])
        if not arquivos_nc:
            raise FileNotFoundError(f"Nenhum arquivo .nc encontrado em {era5_dir}")
        
        print(f"Carregando {len(arquivos_nc)} arquivos ERA5...")
        ds_era5 = xr.open_mfdataset(arquivos_nc, combine='by_coords')

        # --- Processamento ERA5-Land ---
        print("Processando dados ERA5-Land...")
        
        # Padroniza nomes das coordenadas
        coord_mapping_land = {}
        if 'lat' in ds_era5_land.coords:
            coord_mapping_land['lat'] = 'latitude'
        if 'lon' in ds_era5_land.coords:
            coord_mapping_land['lon'] = 'longitude'
        if coord_mapping_land:
            ds_era5_land = ds_era5_land.rename(coord_mapping_land)
        
        # Encontra o ponto mais próximo
        lat_idx_land = abs(ds_era5_land['latitude'] - lat).argmin()
        lon_idx_land = abs(ds_era5_land['longitude'] - lon).argmin()
        ponto_land = ds_era5_land.isel(latitude=lat_idx_land, longitude=lon_idx_land)
        
        # Extrai apenas variáveis com dimensão temporal
        variaveis_temporais_land = [var for var in ds_era5_land.data_vars 
                                   if 'valid_time' in ds_era5_land[var].dims or 'time' in ds_era5_land[var].dims]
        
        dados_land = {}
        for var in variaveis_temporais_land:
            try:
                dados_land[var] = ponto_land[var].values
            except Exception as e:
                print(f"Aviso: Não foi possível extrair a variável {var}: {e}")
                continue
        
        if not dados_land:
            raise ValueError("Nenhuma variável temporal válida encontrada no ERA5-Land")
        
        df_land = pd.DataFrame(dados_land)
        
        # Determina a dimensão temporal correta
        time_dim = 'valid_time' if 'valid_time' in ds_era5_land.dims else 'time'
        df_land[time_dim] = ds_era5_land[time_dim].values
        
        # --- Processamento ERA5 ---
        print("Processando dados ERA5...")
        
        # Padroniza nomes das coordenadas
        coord_mapping_era5 = {}
        if 'lat' in ds_era5.coords:
            coord_mapping_era5['lat'] = 'latitude'
        if 'lon' in ds_era5.coords:
            coord_mapping_era5['lon'] = 'longitude'
        if coord_mapping_era5:
            ds_era5 = ds_era5.rename(coord_mapping_era5)
        
        lat_idx_era5 = abs(ds_era5['latitude'] - lat).argmin()
        lon_idx_era5 = abs(ds_era5['longitude'] - lon).argmin()
        
        # Verifica se existe dimensão de níveis de pressão
        pressure_dim = None
        if "pressure_level" in ds_era5.dims:
            pressure_dim = "pressure_level"
        elif "level" in ds_era5.dims:
            pressure_dim = "level"
        elif "plev" in ds_era5.dims:
            pressure_dim = "plev"
        
        if pressure_dim is None:
            raise ValueError("O dataset ERA5 não possui dimensão de níveis de pressão (pressure_level, level ou plev).")

        # Filtra apenas variáveis meteorológicas relevantes (exclui metadados)
        variaveis_meteorologicas = []
        variaveis_excluir = ['number', 'expver', 'step', 'surface']  # Variáveis de metadados
        
        for var in ds_era5.data_vars:
            # Verifica se a variável tem dimensão temporal e de pressão
            var_dims = ds_era5[var].dims
            time_dim_era5 = 'valid_time' if 'valid_time' in var_dims else 'time' if 'time' in var_dims else None
            
            if (time_dim_era5 and pressure_dim in var_dims and 
                not any(excl in var.lower() for excl in variaveis_excluir)):
                variaveis_meteorologicas.append(var)
        
        if not variaveis_meteorologicas:
            raise ValueError("Nenhuma variável meteorológica válida encontrada no ERA5")
        
        print(f"Variáveis meteorológicas encontradas: {variaveis_meteorologicas}")
        
        lista_df = []
        niveis_pressao = ds_era5[pressure_dim].values
        
        for nivel in niveis_pressao:
            ponto_nivel = ds_era5.isel(latitude=lat_idx_era5, longitude=lon_idx_era5).sel({pressure_dim: nivel})
            
            dados_nivel = {}
            for var in variaveis_meteorologicas:
                try:
                    dados_nivel[var] = ponto_nivel[var].values
                except Exception as e:
                    print(f"Aviso: Erro ao extrair {var} no nível {nivel}: {e}")
                    continue
            
            if dados_nivel:  # Só adiciona se há dados válidos
                df_nivel = pd.DataFrame(dados_nivel)
                time_dim_era5 = 'valid_time' if 'valid_time' in ds_era5.dims else 'time'
                df_nivel[time_dim_era5] = ds_era5[time_dim_era5].values
                df_nivel[pressure_dim] = nivel
                lista_df.append(df_nivel)
        
        if not lista_df:
            raise ValueError("Nenhum dado válido foi extraído do ERA5")
        
        df_empilhado = pd.concat(lista_df, ignore_index=True)
        df_era5_csv = df_empilhado.set_index([time_dim_era5, pressure_dim]).unstack(level=pressure_dim)
        df_era5_csv.columns = [f"{var}_{press}" for var, press in df_era5_csv.columns]
        df_era5_csv = df_era5_csv.reset_index()

        # --- Agregação ERA5-Land ---
        print("Agregando dados ERA5-Land...")
        df_land_clean = df_land.dropna()
        df_land_clean["date"] = pd.to_datetime(df_land_clean[time_dim]).dt.date
        
        # Define variáveis por tipo (ajuste conforme suas variáveis)
        variaveis_instantaneas_land = ["u10", "v10", "d2m", "t2m", "sp", "z"]
        variaveis_acumuladas_land = ["tp", "ssrd", "strd", "sf"]
        
        # Filtra apenas variáveis que existem no dataset
        vars_inst_existentes = [v for v in variaveis_instantaneas_land if v in df_land_clean.columns]
        vars_acum_existentes = [v for v in variaveis_acumuladas_land if v in df_land_clean.columns]
        
        df_land_final = pd.DataFrame()
        
        if vars_inst_existentes:
            df_instantaneas = df_land_clean.groupby("date")[vars_inst_existentes].mean()
            df_land_final = pd.concat([df_land_final, df_instantaneas], axis=1)
        
        if vars_acum_existentes:
            df_acumuladas = df_land_clean.groupby("date")[vars_acum_existentes].sum()
            df_land_final = pd.concat([df_land_final, df_acumuladas], axis=1)
        
        df_land_final = df_land_final.reset_index()
        
        # Conversões de unidades
        if "tp" in df_land_final.columns:
            df_land_final["tp"] = df_land_final["tp"] * 1000  # m para mm
        if "t2m" in df_land_final.columns:
            df_land_final["t2m"] = df_land_final["t2m"] - 273.15  # K para °C
        if "d2m" in df_land_final.columns:
            df_land_final["d2m"] = df_land_final["d2m"] - 273.15  # K para °C

        # --- Agregação ERA5 ---
        print("Agregando dados ERA5...")
        df_era5_clean = df_era5_csv.dropna()
        df_era5_clean.rename(columns={time_dim_era5: "date"}, inplace=True)
        df_era5_clean["date"] = pd.to_datetime(df_era5_clean["date"]).dt.date
        
        variaveis_era5 = [col for col in df_era5_clean.columns if col != 'date']
        df_era5_final = df_era5_clean.groupby("date")[variaveis_era5].mean().reset_index()
        
        # Conversão de temperatura (K para °C) para variáveis que começam com 't_'
        for col in df_era5_final.columns:
            if col.startswith('t_'):
                df_era5_final[col] = df_era5_final[col] - 273.15

        # --- Junção dos dados ERA ---
        print("Combinando dados ERA5 e ERA5-Land...")
        df_era5_final['date'] = pd.to_datetime(df_era5_final['date'])
        df_land_final['date'] = pd.to_datetime(df_land_final['date'])
        
        df_era5_merged = pd.merge(df_era5_final, df_land_final, on='date', how='inner')
        df_era5_merged = df_era5_merged.dropna()

        if df_era5_merged.empty:
            raise ValueError("Nenhum dado coincidente encontrado entre ERA5 e ERA5-Land")
        # Salva os dados combinados em CSV
        output_path_era = output_dir / f"dados_era5_{station_input_path.stem}_processado.csv"
        df_era5_merged.to_csv(output_path_era, index=False)
        print(f"Dados combinados ERA salvos em: {output_path_era}")
        print(f"Colunas no arquivo final: {list(df_era5_merged.columns)}")

    
        
        # Resumo final
        print(f"\n=== RESUMO DO PROCESSAMENTO ===")
        print(f"ERA5 + ERA5-Land: {len(df_era5_merged)} registros, {len(df_era5_merged.columns)} variáveis")
        print(f"Estação: {len(df_estacao_final)} registros")
        print(f"Período ERA5: {df_era5_merged['date'].min()} a {df_era5_merged['date'].max()}")
        print(f"Período Estação: {df_estacao_final['data'].min()} a {df_estacao_final['data'].max()}")
        print("\nProcessamento concluído com sucesso!")
        
        return "Processamento concluído com sucesso!", "success"
    
    except Exception as e:
        error_msg = f"Ocorreu um erro: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, "danger"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa dados ERA5/ERA5-Land e de estação meteorológica.")
    parser.add_argument("--era5_dir", type=str, required=True, help="Diretório contendo arquivos .nc do ERA5.")
    parser.add_argument("--era5_land_dir", type=str, required=True, help="Diretório contendo arquivos .nc do ERA5-Land.")
    parser.add_argument("--station_file", type=str, required=True, help="Caminho para o arquivo .csv da estação.")
    parser.add_argument("--lat", type=float, required=True, help="Latitude do ponto de interesse.")
    parser.add_argument("--lon", type=float, required=True, help="Longitude do ponto de interesse.")
    args = parser.parse_args()
    
    iniciar_processamento(args.era5_dir, args.era5_land_dir, args.station_file, args.lat, args.lon)