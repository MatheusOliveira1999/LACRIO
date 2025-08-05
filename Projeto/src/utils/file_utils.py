"""
Utilitários para manipulação de arquivos
"""

import os
import pandas as pd
import xarray as xr
from typing import Dict, List, Any
from pathlib import Path


def create_project_directories():
    """Cria diretórios necessários do projeto"""
    from config.settings import REQUIRED_DIRS
    
    for directory in REQUIRED_DIRS:
        Path(directory).mkdir(parents=True, exist_ok=True)


def validate_input_files(era5_path: str, station_path: str, dem_path: str) -> Dict[str, bool]:
    """
    Valida se os arquivos de entrada são válidos
    
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
    
    # Validar ERA5
    try:
        if os.path.exists(era5_path) and os.path.getsize(era5_path) > 0:
            # Tentar ler o arquivo
            df_test = pd.read_csv(era5_path, nrows=5)
            if 'date' in df_test.columns and len(df_test) > 0:
                validation_results['era5'] = True
            else:
                validation_results['era5'] = False
        else:
            validation_results['era5'] = False
    except Exception:
        validation_results['era5'] = False
    
    # Validar Estação  
    try:
        if os.path.exists(station_path) and os.path.getsize(station_path) > 0:
            # Tentar ler o arquivo
            df_test = pd.read_csv(station_path, nrows=5)
            # Verificar se tem coluna de data (vários nomes possíveis)
            date_cols = ['date', 'data', 'Data', 'DATE']
            has_date = any(col in df_test.columns for col in date_cols)
            if has_date and len(df_test) > 0:
                validation_results['station'] = True
            else:
                validation_results['station'] = False
        else:
            validation_results['station'] = False
    except Exception:
        validation_results['station'] = False
    
    # Validar DEM
    try:
        if os.path.exists(dem_path) and os.path.getsize(dem_path) > 0:
            # Tentar abrir como NetCDF
            ds_test = xr.open_dataset(dem_path)
            # Verificar se tem variáveis
            if len(ds_test.variables) > 0:
                validation_results['dem'] = True
            else:
                validation_results['dem'] = False
            ds_test.close()
        else:
            validation_results['dem'] = False
    except Exception:
        validation_results['dem'] = False
    
    return validation_results


def get_file_info(filepath: str) -> Dict[str, Any]:
    """
    Obtém informações sobre um arquivo
    
    Parameters:
    -----------
    filepath : str
        Caminho para o arquivo
        
    Returns:
    --------
    dict : Informações do arquivo
    """
    if not os.path.exists(filepath):
        return {'exists': False}
    
    file_info = {
        'exists': True,
        'size_bytes': os.path.getsize(filepath),
        'size_mb': round(os.path.getsize(filepath) / (1024*1024), 2),
        'extension': os.path.splitext(filepath)[1].lower(),
        'basename': os.path.basename(filepath)
    }
    
    # Informações específicas por tipo
    try:
        if file_info['extension'] == '.csv':
            df = pd.read_csv(filepath, nrows=10)
            file_info.update({
                'type': 'CSV',
                'columns': list(df.columns),
                'sample_rows': len(df)
            })
        elif file_info['extension'] == '.nc':
            ds = xr.open_dataset(filepath)
            file_info.update({
                'type': 'NetCDF',
                'variables': list(ds.variables.keys()),
                'dimensions': list(ds.dimensions.keys())
            })
            ds.close()
    except Exception as e:
        file_info['error'] = str(e)
    
    return file_info


def clean_filename(filename: str) -> str:
    """
    Limpa nome do arquivo removendo caracteres problemáticos
    
    Parameters:
    -----------
    filename : str
        Nome do arquivo original
        
    Returns:
    --------
    str : Nome do arquivo limpo
    """
    import re
    
    # Remover caracteres especiais
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Remover underscores múltiplos
    filename = re.sub(r'_+', '_', filename)
    
    # Remover underscores no início e fim
    filename = filename.strip('_')
    
    return filename


def ensure_directory_exists(directory: str):
    """
    Garante que um diretório existe
    
    Parameters:
    -----------
    directory : str
        Caminho do diretório
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def list_files_in_directory(directory: str, extension: str = None) -> List[str]:
    """
    Lista arquivos em um diretório
    
    Parameters:
    -----------
    directory : str
        Caminho do diretório
    extension : str, optional
        Filtrar por extensão (ex: '.csv')
        
    Returns:
    --------
    list : Lista de arquivos encontrados
    """
    if not os.path.exists(directory):
        return []
    
    files = []
    for file in os.listdir(directory):
        filepath = os.path.join(directory, file)
        if os.path.isfile(filepath):
            if extension is None or file.lower().endswith(extension.lower()):
                files.append(filepath)
    
    return sorted(files)


def copy_file_to_directory(source_path: str, destination_dir: str, 
                          new_name: str = None) -> str:
    """
    Copia arquivo para um diretório
    
    Parameters:
    -----------
    source_path : str
        Caminho do arquivo origem
    destination_dir : str
        Diretório de destino
    new_name : str, optional
        Novo nome para o arquivo
        
    Returns:
    --------
    str : Caminho do arquivo copiado
    """
    import shutil
    
    ensure_directory_exists(destination_dir)
    
    if new_name:
        destination_path = os.path.join(destination_dir, new_name)
    else:
        destination_path = os.path.join(destination_dir, os.path.basename(source_path))
    
    shutil.copy2(source_path, destination_path)
    return destination_path


def delete_old_files(directory: str, max_age_days: int = 7):
    """
    Remove arquivos antigos de um diretório
    
    Parameters:
    -----------
    directory : str
        Diretório para limpar
    max_age_days : int
        Idade máxima em dias
    """
    import time
    
    if not os.path.exists(directory):
        return
    
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > max_age_seconds:
                try:
                    os.remove(filepath)
                    print(f"Arquivo antigo removido: {filename}")
                except Exception as e:
                    print(f"Erro ao remover {filename}: {str(e)}")


def get_directory_size(directory: str) -> Dict[str, Any]:
    """
    Calcula tamanho de um diretório
    
    Parameters:
    -----------
    directory : str
        Caminho do diretório
        
    Returns:
    --------
    dict : Informações de tamanho
    """
    if not os.path.exists(directory):
        return {'exists': False}
    
    total_size = 0
    file_count = 0
    
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
                file_count += 1
            except OSError:
                pass
    
    return {
        'exists': True,
        'total_size_bytes': total_size,
        'total_size_mb': round(total_size / (1024*1024), 2),
        'file_count': file_count
    }


if __name__ == "__main__":
    # Teste básico
    print("🧪 Testando utilitários de arquivo...")
    
    # Testar criação de diretórios
    create_project_directories()
    print("✅ Diretórios criados")
    
    # Testar informações de diretório
    info = get_directory_size('.')
    print(f"📁 Diretório atual: {info['file_count']} arquivos, {info['total_size_mb']} MB")