# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LACRIO is a comprehensive climate downscaling toolkit focused on improving numerical weather model and global reanalysis data. The repository contains multiple interconnected modules for meteorological data processing, analysis, and machine learning-based downscaling.

## High-Level Architecture

The repository is organized into four main components:

### 1. Pre-processamento_UTC/
**Meteorological data preprocessing system**
- Processes ERA5/ERA5-Land reanalysis data and local weather station observations
- Converts Peru local time (UTC-5) to UTC for temporal alignment
- Combines multiple data sources into unified CSV files

### 2. Pre_downscaling_comparison/
**Data quality analysis and validation system**
- Pre-downscaling data comparison between ERA5 and weather station observations
- Generates comprehensive reports with statistical analysis and visualizations
- CLI interface for automated processing
- Integrates with the main downscaling system

### 3. Projeto_modular_downscaling-tool2/
**Main ML-based downscaling system**
- Advanced statistical downscaling using 9 machine learning algorithms
- Feature engineering with 80+ derived variables including lags, moving averages, and interactions
- CLI interface for automated processing
- Ensemble learning combining multiple models

### 4. Dados_satelites/
**Satellite data integration (in development)**
- NASA GPM-IMERG precipitation data processing
- MODIS Land Surface Temperature integration
- Planned expansion to the downscaling system

## Common Development Commands

### Pre-processamento_UTC
```bash
# Process daily data with UTC conversion
python processar_dados.py \
    --era5_dir "/path/to/era5" \
    --era5_land_dir "/path/to/era5_land" \
    --station_file "/path/to/station.csv" \
    --lat -9.5 --lon -77.5

# Process hourly data
python processar_dados_horario.py \
    --era5_dir "/path/to/era5" \
    --era5_land_dir "/path/to/era5_land" \
    --station_file "/path/to/station.csv" \
    --lat -9.5 --lon -77.5

# Web interface for data processing
python app.py  # or app_dados_horarios.py for hourly data
```

### Pre_downscaling_comparison
```bash
# Basic data comparison
python run_comparison.py \
    --era5 era5_data.csv \
    --station station_data.csv \
    --station-name "Llanganuco"

# Advanced comparison with report generation
python run_comparison.py \
    --era5 era5_data.csv \
    --station station_data.csv \
    --station-name "Llanganuco" \
    --variables temperature precipitation \
    --generate-report \
    --output-dir results

# Dependencies installation
pip install -r requirements.txt
```

### Projeto_modular_downscaling-tool2
```bash
# Activate conda environment
conda activate temp-predict

# Basic downscaling execution
python run_downscaling.py \
    --era5 data.csv \
    --station station.csv \
    --dem elevation.nc \
    --lat -9.41 --lon -77.35

# Advanced execution with optimization
python run_downscaling.py \
    --era5 data.csv \
    --station station.csv \
    --dem elevation.nc \
    --lat -9.41 --lon -77.35 \
    --models RandomForest XGBoost MLP \
    --optimize \
    --variables temperature precipitation

# Test validation module
/home/matheus/miniconda3/envs/temp-predict/bin/python -c "
import sys; sys.path.append('.')
from analyses.validacao.validador_downscaling import ValidadorDownscaling
print('✅ Validation module imported successfully')
"
```

## Data Pipeline Flow

### Typical Workflow:
1. **Raw Data Processing** (Pre-processamento_UTC): Clean and align temporal data
2. **Quality Analysis** (Pre_downscaling_comparison): Validate data quality and generate reports
3. **Downscaling** (Projeto_modular_downscaling-tool2): Apply ML models for high-resolution predictions
4. **Satellite Integration** (Future): Incorporate remote sensing data

### Data Requirements:
- **ERA5/ERA5-Land**: NetCDF files with atmospheric and surface variables
- **Weather Stations**: CSV files with columns: `Datetime`, `Temperature (°C)`, `Precipitation (mm)`
- **DEM**: Digital Elevation Model in NetCDF format for topographic corrections
- **Coordinates**: Latitude/longitude of target location

## Key Implementation Details

### Temporal Handling
- All systems use UTC as the standard timezone
- Pre-processamento_UTC handles conversion from Peru local time (UTC-5)
- Temporal validation prevents data leakage in ML models
- Default train/test split: before/after 2018-01-01

### Feature Engineering
- Creates 80+ derived features including temporal lags (1,2,3,7,14 days)
- Moving averages (3,7,14,30 days) and statistical aggregations
- Topographic corrections using DEM data
- Automatic scaling and preprocessing

### Model Architecture
- 9 ML algorithms: Linear Regression, Ridge, Random Forest, Extra Trees, Gradient Boosting, XGBoost, SVR, MLP, Ensemble
- Temporal validation with chronological splits
- Hyperparameter optimization via GridSearch/RandomSearch
- Ensemble meta-learning combining top performers

### Output Structure
```
models/{location}/          # Trained ML models (.pkl files)
results/{location}/         # Performance reports and metrics
static/img/{location}/      # Generated visualizations and plots
uploads/                    # Input data staging area
```

## Environment Requirements

### Python Dependencies (varies by module):
- **Core**: `numpy`, `pandas`, `xarray`, `netCDF4`
- **ML**: `scikit-learn`, `xgboost`
- **Visualization**: `matplotlib`, `seaborn`
- **Web**: `Flask`, `Werkzeug` (for web interfaces)
- **Temporal**: `pytz` (for timezone handling)

### Data Formats:
- **Input**: CSV (weather stations), NetCDF (ERA5, DEM), NetCDF4 (satellites)
- **Output**: CSV (processed data), PKL (trained models), PNG (visualizations), MD/HTML (reports)

## Regional Context

This system is specifically designed for climate research in the Cordillera Blanca region of Peru, focusing on:
- High-altitude meteorological stations in tropical Andes
- Complex topographic effects on climate variables
- Conversion between global reanalysis data and local observations
- Agricultural and hydrological applications in data-sparse regions

When working with this codebase, ensure proper environment activation (especially `temp-predict` conda environment for the main downscaling tool) and verify that all required data files are available before executing processing pipelines.