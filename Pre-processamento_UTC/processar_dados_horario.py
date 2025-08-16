import xarray as xr
import os
import pandas as pd
import argparse
from pathlib import Path
import warnings
from datetime import timezone, timedelta

warnings.filterwarnings('ignore')

def iniciar_processamento(era5_dir, era5_land_dir, station_file, lat, lon):
    """
    Função principal que orquestra o processamento dos dados
    ERA5 e da estação meteorológica mantendo horários específicos.
    """
    try:
        output_dir = Path("dados_horarios_csv")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Arquivos CSV serão salvos em: '{output_dir.resolve()}'")
        
        # --- 1. Processamento dos dados da estação ---
        print("\nIniciando processamento dos dados da estação...")
        
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
        
        # Remove colunas com NaN (todas as linhas são NaN)
        df_estacao = df_estacao.dropna(axis=1, how='all')
        print(f"Colunas após remover colunas com todos NaN: {list(df_estacao.columns)}")
        
        # Processa datetime e filtra por horários específicos
        if "Datetime" not in df_estacao.columns:
            raise ValueError("Coluna 'Datetime' não encontrada")
            
        df_estacao["Datetime"] = pd.to_datetime(df_estacao["Datetime"], errors='coerce')
        df_estacao = df_estacao.dropna(subset=['Datetime'])
        
        # Correção de timezone: dados da estação estão em horário local do Peru (UTC-5)
        # Converte para UTC adicionando 5 horas
        df_estacao["Datetime"] = df_estacao["Datetime"] + timedelta(hours=5)
        print("Correção de timezone aplicada: horário local Peru (UTC-5) convertido para UTC")
        
        # Filtra apenas horários 00:00, 06:00, 12:00, 18:00 (minutos = 00)
        horarios_desejados = [0, 6, 12, 18]
        df_estacao = df_estacao[
            (df_estacao['Datetime'].dt.hour.isin(horarios_desejados)) & 
            (df_estacao['Datetime'].dt.minute == 0)
        ]
        
        if df_estacao.empty:
            raise ValueError("Nenhum dado válido de estação nos horários especificados")
        
        # Renomeia coluna de data para 'data' antes da remoção de NaN
        if 'Datetime' in df_estacao.columns:
            df_estacao = df_estacao.rename(columns={'Datetime': 'data'})
        
        # Remove parênteses com unidades dos nomes das colunas
        df_estacao.columns = [col.split('(')[0].strip() for col in df_estacao.columns]
        
        # Filtra apenas colunas de temperatura e precipitação (além da coluna de data)
        colunas_interesse = ['data']
        colunas_disponiveis = df_estacao.columns.tolist()
        
        # Procura por colunas de temperatura (temp, temperature, t, etc)
        for col in colunas_disponiveis:
            col_lower = col.lower()
            if any(termo in col_lower for termo in ['temp', 'temperature', 't2m', 'air', 'celsius']):
                colunas_interesse.append(col)
                print(f"Coluna de temperatura identificada: {col}")
        
        # Procura por colunas de precipitação (precip, rain, precipitation, etc)
        for col in colunas_disponiveis:
            col_lower = col.lower()
            if any(termo in col_lower for termo in ['precip', 'rain', 'precipitation', 'pluvio', 'chuva', 'tp']):
                colunas_interesse.append(col)
                print(f"Coluna de precipitação identificada: {col}")
        
        # Remove duplicatas mantendo ordem
        colunas_interesse = list(dict.fromkeys(colunas_interesse))
        
        # Filtra o dataframe mantendo apenas as colunas de interesse
        colunas_existentes = [col for col in colunas_interesse if col in df_estacao.columns]
        df_estacao = df_estacao[colunas_existentes]
        
        print(f"Colunas mantidas nos dados da estação: {list(df_estacao.columns)}")
        
        # Remove células individuais com valores NaN (mantém estrutura mas com diferentes quantidades por coluna)
        # Não remove linhas inteiras, apenas transforma NaN em valores ausentes
        # Os dados serão salvos com células vazias onde havia NaN
        
        station_input_path = Path(station_file)
        station_output_name = f"{station_input_path.stem}_horario_processado.csv"
        output_path_station = output_dir / station_output_name
        
        df_estacao.to_csv(output_path_station, index=False)
        print(f"Dados da estação processados e salvos em: {output_path_station}")
        print(f"Período estação: {df_estacao['data'].min()} a {df_estacao['data'].max()}")
        print(f"Registros na estação: {len(df_estacao)}")

        # --- 2. Processamento dos dados ERA5 e ERA5-Land ---
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
        df_land['date'] = ds_era5_land[time_dim].values
        
        # Filtra horários específicos para ERA5-Land
        df_land['datetime'] = pd.to_datetime(df_land['date'])
        df_land = df_land[
            (df_land['datetime'].dt.hour.isin(horarios_desejados)) & 
            (df_land['datetime'].dt.minute == 0)
        ]
        
        if df_land.empty:
            raise ValueError("Nenhum dado ERA5-Land nos horários especificados")

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
            raise ValueError("O dataset ERA5 não possui dimensão de níveis de pressão")

        # Filtra apenas variáveis meteorológicas relevantes
        variaveis_meteorologicas = []
        variaveis_excluir = ['number', 'expver', 'step', 'surface']
        
        for var in ds_era5.data_vars:
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
            
            if dados_nivel:
                df_nivel = pd.DataFrame(dados_nivel)
                time_dim_era5 = 'valid_time' if 'valid_time' in ds_era5.dims else 'time'
                df_nivel['date'] = ds_era5[time_dim_era5].values
                df_nivel[pressure_dim] = nivel
                lista_df.append(df_nivel)
        
        if not lista_df:
            raise ValueError("Nenhum dado válido foi extraído do ERA5")
        
        df_empilhado = pd.concat(lista_df, ignore_index=True)
        
        # Filtra horários específicos para ERA5
        df_empilhado['datetime'] = pd.to_datetime(df_empilhado['date'])
        df_empilhado = df_empilhado[
            (df_empilhado['datetime'].dt.hour.isin(horarios_desejados)) & 
            (df_empilhado['datetime'].dt.minute == 0)
        ]
        
        if df_empilhado.empty:
            raise ValueError("Nenhum dado ERA5 nos horários especificados")
        
        # Remove a coluna datetime temporária antes do unstacking
        df_empilhado = df_empilhado.drop(columns=['datetime'])
        
        df_era5_csv = df_empilhado.set_index(['date', pressure_dim]).unstack(level=pressure_dim)
        df_era5_csv.columns = [f"{var}_{int(press)}" for var, press in df_era5_csv.columns]
        df_era5_csv = df_era5_csv.reset_index()

        # --- Combinação dos dados sem agregação ---
        print("Combinando dados ERA5 e ERA5-Land sem agregação...")
        
        # Prepara ERA5-Land para junção (mantém NaN como células vazias)
        df_land_clean = df_land.copy()
        df_land_clean = df_land_clean.rename(columns={'datetime': 'timestamp'})
        
        # Prepara ERA5 para junção (mantém NaN como células vazias)
        df_era5_clean = df_era5_csv.copy()
        df_era5_clean['timestamp'] = pd.to_datetime(df_era5_clean['date'])
        
        # Conversões de unidades antes da junção
        if "tp" in df_land_clean.columns:
            df_land_clean["tp"] = df_land_clean["tp"] * 1000  # m para mm
        if "t2m" in df_land_clean.columns:
            df_land_clean["t2m"] = df_land_clean["t2m"] - 273.15  # K para °C
        if "d2m" in df_land_clean.columns:
            df_land_clean["d2m"] = df_land_clean["d2m"] - 273.15  # K para °C
            
        # Conversão de temperatura para ERA5 (K para °C)
        for col in df_era5_clean.columns:
            if col.startswith('t_'):
                df_era5_clean[col] = df_era5_clean[col] - 273.15

        # Renomeia coluna timestamp para 'date' se existir
        if 'timestamp' in df_era5_clean.columns:
            df_era5_clean = df_era5_clean.rename(columns={'timestamp': 'date'})
        if 'timestamp' in df_land_clean.columns:
            df_land_clean = df_land_clean.rename(columns={'timestamp': 'date'})

        # Renomeia coluna 'valid_time' para 'date' se existir
        if 'valid_time' in df_era5_clean.columns:
            df_era5_clean = df_era5_clean.rename(columns={'valid_time': 'date'})
        if 'valid_time' in df_land_clean.columns:
            df_land_clean = df_land_clean.rename(columns={'valid_time': 'date'})

        # Remove colunas duplicadas 'date' mantendo apenas a primeira
        df_era5_clean = df_era5_clean.loc[:, ~df_era5_clean.columns.duplicated()]
        df_land_clean = df_land_clean.loc[:, ~df_land_clean.columns.duplicated()]

        # Garante que a coluna 'date' está no tipo datetime
        df_era5_clean['date'] = pd.to_datetime(df_era5_clean['date'])
        df_land_clean['date'] = pd.to_datetime(df_land_clean['date'])

        # Faz o merge
        df_era5_merged = pd.merge(df_era5_clean, df_land_clean, on='date', how='inner')

        if df_era5_merged.empty:
            raise ValueError("Nenhum dado coincidente encontrado entre ERA5 e ERA5-Land")
            
        # Limpa nomes das variáveis ERA5 removendo símbolos
        df_era5_merged.columns = [col.replace('(', '').replace(')', '').replace('/', '').replace('-', '_').replace(' ', '_').replace('.', '') for col in df_era5_merged.columns]
        
        # Salva os dados combinados em CSV
        output_path_era = output_dir / f"dados_era5_horario_{station_input_path.stem}_processado.csv"
        df_era5_merged.to_csv(output_path_era, index=False)
        print(f"Dados combinados ERA salvos em: {output_path_era}")
        print(f"Colunas no arquivo final: {list(df_era5_merged.columns)}")
        
        # Resumo final
        print(f"\n=== RESUMO DO PROCESSAMENTO ===")
        print(f"ERA5 + ERA5-Land: {len(df_era5_merged)} registros, {len(df_era5_merged.columns)} variáveis")
        print(f"Estação: {len(df_estacao)} registros, {len(df_estacao.columns)} variáveis")
        print(f"Período ERA5: {df_era5_merged['date'].min()} a {df_era5_merged['date'].max()}")
        print(f"Horários mantidos: 00:00, 06:00, 12:00, 18:00")
        print("\nProcessamento concluído com sucesso!")
        
        return "Processamento concluído com sucesso!", "success"
    
    except Exception as e:
        error_msg = f"Ocorreu um erro: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, "danger"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa dados ERA5/ERA5-Land e de estação meteorológica mantendo horários específicos.")
    parser.add_argument("--era5_dir", type=str, required=True, help="Diretório contendo arquivos .nc do ERA5.")
    parser.add_argument("--era5_land_dir", type=str, required=True, help="Diretório contendo arquivos .nc do ERA5-Land.")
    parser.add_argument("--station_file", type=str, required=True, help="Caminho para o arquivo .csv da estação.")
    parser.add_argument("--lat", type=float, required=True, help="Latitude do ponto de interesse.")
    parser.add_argument("--lon", type=float, required=True, help="Longitude do ponto de interesse.")
    args = parser.parse_args()
    
    iniciar_processamento(args.era5_dir, args.era5_land_dir, args.station_file, args.lat, args.lon)