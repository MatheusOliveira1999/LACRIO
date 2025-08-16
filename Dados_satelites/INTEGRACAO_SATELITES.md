# Projeto de Integração de Dados de Satélites no Sistema de Downscaling

## 📋 Visão Geral

Este projeto visa integrar dados de satélites (NASA GPM-IMERG e MODIS LST) ao sistema modular de downscaling climático existente, expandindo as capacidades do modelo para incluir observações de precipitação e temperatura de superfície derivadas de sensoriamento remoto.

## 🎯 Objetivos

### Objetivo Principal
Integrar dados de satélites da NASA e MODIS ao pipeline de downscaling para melhorar a precisão das predições climáticas através de múltiplas fontes de dados observacionais.

### Objetivos Específicos
1. **Processamento de Dados GPM-IMERG**: Integrar dados de precipitação diária da NASA (formato .nc4)
2. **Processamento de Dados MODIS LST**: Integrar dados de temperatura de superfície (formato .tiff)  
3. **Sincronização Temporal**: Alinhar séries temporais de diferentes fontes de dados
4. **Feature Engineering**: Criar features derivadas dos dados de satélites
5. **Validação Cruzada**: Comparar observações de satélites com dados de estações

## 📊 Dados Disponíveis

### 1. Dados NASA GPM-IMERG (Precipitação)
- **Localização**: `Dados_satelites/dados_nasa/`
- **Formato**: NetCDF4 (.nc4)
- **Variável**: Precipitação diária (mm/dia)
- **Resolução Temporal**: Diária
- **Resolução Espacial**: 0.1° x 0.1°
- **Período**: 1998-2025
- **Estrutura**: `3B-DAY.MS.MRG.3IMERG.YYYYMMDD-S000000-E235959.V07B.nc4`

### 2. Dados MODIS LST (Temperatura de Superfície)
- **Localização**: `Dados_satelites/dados_modis/`
- **Formato**: GeoTIFF (.tif)
- **Variável**: Land Surface Temperature (LST)
- **Resolução Temporal**: 8-diária
- **Resolução Espacial**: ~1 km
- **Período**: 2000-2024
- **Estrutura**: `MODIS_LST_YYYY-MM-DD.tif`

## 🏗️ Arquitetura da Integração

### Pipeline Proposto

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Dados NASA    │    │   Dados MODIS    │    │  Dados Estação  │
│  (GPM-IMERG)    │    │     (LST)        │    │   + ERA5        │
│     .nc4        │    │     .tiff        │    │     .csv        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              PRÉ-PROCESSAMENTO DE SATÉLITES                     │
│  • Extração espacial (lat/lon específicos)                     │
│  • Reprojeção e regrid                                         │
│  • Interpolação temporal                                       │
│  • Controle de qualidade                                       │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 SINCRONIZAÇÃO TEMPORAL                         │
│  • Alinhamento de séries temporais                             │
│  • Tratamento de valores faltantes                             │
│  • Padronização de timestamps                                  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│               FEATURE ENGINEERING SATELITAL                    │
│  • Lags temporais                                              │
│  • Médias móveis                                               │
│  • Anomalias sazonais                                          │
│  • Índices derivados                                           │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│              INTEGRAÇÃO AO MODELO EXISTENTE                    │
│  • Adição de features aos pipelines ML                         │
│  • Ensemble multi-fonte                                        │
│  • Validação cruzada                                           │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Implementação Técnica

### Fase 1: Módulo de Processamento de Satélites

#### 1.1 Classe `SatelliteDataProcessor`

```python
# Localização: src/core/satellite_processor.py

class SatelliteDataProcessor:
    """
    Processador de dados de satélites para integração 
    ao sistema de downscaling
    """
    
    def __init__(self, lat, lon, buffer_distance=0.1):
        self.lat = lat
        self.lon = lon
        self.buffer_distance = buffer_distance
        
    def process_gpm_imerg(self, data_dir, start_date, end_date):
        """Processa dados GPM-IMERG NetCDF4"""
        pass
        
    def process_modis_lst(self, data_dir, start_date, end_date):
        """Processa dados MODIS LST GeoTIFF"""
        pass
        
    def spatial_extraction(self, dataset, lat, lon):
        """Extrai dados para coordenadas específicas"""
        pass
        
    def temporal_interpolation(self, data, target_frequency='D'):
        """Interpola dados para frequência temporal desejada"""
        pass
```

#### 1.2 Estrutura de Arquivos

```
src/
├── core/
│   ├── data_processor.py          # Existente
│   ├── satellite_processor.py     # NOVO - Processamento satélites
│   └── data_integrator.py         # NOVO - Integração multi-fonte
├── ml/
│   ├── satellite_features.py      # NOVO - Features de satélites
│   └── ensemble_satellite.py      # NOVO - Ensemble multi-fonte
└── utils/
    ├── satellite_utils.py         # NOVO - Utilitários satélites
    └── quality_control.py         # NOVO - Controle qualidade
```

### Fase 2: Feature Engineering Satelital

#### 2.1 Features Derivadas - GPM-IMERG

- **Temporais**:
  - Precipitação acumulada (3, 7, 15, 30 dias)
  - Número de dias secos consecutivos
  - Intensidade máxima de precipitação
  - Anomalias sazonais

- **Estatísticas**:
  - Média, mediana, percentis (75, 90, 95)
  - Coeficiente de variação
  - Assimetria e curtose

#### 2.2 Features Derivadas - MODIS LST

- **Temporais**:
  - Temperatura média, mínima, máxima
  - Amplitude térmica diurna
  - Tendências temporais
  - Ciclos sazonais

- **Espaciais**:
  - Gradiente térmico
  - Rugosidade térmica
  - Correlação espacial

### Fase 3: Integração ao Pipeline Existente

#### 3.1 Modificações na Classe `WeatherDownscalingModel`

```python
# src/core/weather_model.py (modificações)

class WeatherDownscalingModel:
    def __init__(self, variable='temperature', use_satellites=False):
        self.use_satellites = use_satellites
        if use_satellites:
            self.satellite_processor = SatelliteDataProcessor()
    
    def load_and_merge_data(self, era5_path, station_path, dem_path, 
                          lat, lon, satellite_data_dirs=None):
        """
        Estende método existente para incluir dados de satélites
        """
        # Carregamento existente (ERA5, estação, DEM)
        data = self._load_existing_data(era5_path, station_path, dem_path, lat, lon)
        
        if self.use_satellites and satellite_data_dirs:
            # Carregar dados de satélites
            satellite_data = self._load_satellite_data(satellite_data_dirs, lat, lon)
            # Integrar com dados existentes
            data = self._integrate_satellite_data(data, satellite_data)
            
        return data
```

## 📈 Etapas de Desenvolvimento

### Sprint 1: Infraestrutura Base (2-3 dias)
- [x] Análise da estrutura atual do projeto
- [x] Identificação dos dados de satélites disponíveis
- [x] Especificação técnica da integração
- [ ] Criação da classe `SatelliteDataProcessor`
- [ ] Implementação do carregamento básico GPM-IMERG

### Sprint 2: Processamento GPM-IMERG (3-4 dias)
- [ ] Extração espacial para coordenadas específicas
- [ ] Interpolação temporal para frequência diária
- [ ] Controle de qualidade dos dados
- [ ] Validação cruzada com dados de estações
- [ ] Testes unitários

### Sprint 3: Processamento MODIS LST (3-4 dias)
- [ ] Leitura e processamento de arquivos GeoTIFF
- [ ] Reprojeção para sistema de coordenadas consistente
- [ ] Interpolação temporal (8-diária → diária)
- [ ] Correções atmosféricas e topográficas
- [ ] Testes de integração

### Sprint 4: Feature Engineering (2-3 dias)
- [ ] Implementação de features temporais
- [ ] Features estatísticas e índices derivados
- [ ] Detecção e correção de anomalias
- [ ] Normalização e escalonamento
- [ ] Seleção de features relevantes

### Sprint 5: Integração ao Pipeline ML (3-4 dias)
- [ ] Modificação das classes existentes
- [ ] Ensemble multi-fonte (ERA5 + Satélites + Estações)
- [ ] Validação temporal com dados de satélites
- [ ] Métricas de performance específicas
- [ ] Interface web atualizada

### Sprint 6: Validação e Otimização (2-3 dias)
- [ ] Validação cruzada extensiva
- [ ] Análise de melhoria da performance
- [ ] Otimização de parâmetros
- [ ] Documentação técnica
- [ ] Relatório de resultados

## 🧪 Pré-processamento Necessário

### 1. Dados GPM-IMERG

#### Desafios Identificados:
- **Volume**: ~730 arquivos (2 anos de dados diários)
- **Formato**: NetCDF4 com estrutura complexa
- **Resolução**: 0.1° pode exigir interpolação espacial
- **Coordenadas**: Sistema de coordenadas específico da NASA

#### Processamento Requerido:
```python
def preprocess_gpm_imerg():
    """
    1. Extrair variável de precipitação
    2. Reprojetar para coordenadas locais
    3. Extrair pixel mais próximo das coordenadas da estação
    4. Aplicar fatores de escala e offsets
    5. Controle de qualidade (valores físicos válidos)
    6. Agregação temporal se necessário
    """
```

### 2. Dados MODIS LST

#### Desafios Identificados:
- **Volume**: ~2,300 arquivos (16 anos, 8-diária)
- **Formato**: GeoTIFF com projeções variadas
- **Resolução**: ~1km pode exigir agregação espacial
- **Frequência**: 8-diária requer interpolação temporal

#### Processamento Requerido:
```python
def preprocess_modis_lst():
    """
    1. Leitura de arquivos GeoTIFF
    2. Conversão de DN para temperatura (Kelvin → Celsius)
    3. Reprojeção para WGS84
    4. Extração espacial para área de interesse
    5. Interpolação temporal para dados diários
    6. Detecção de nuvens e píxeis inválidos
    """
```

## 📊 Resultados Esperados

### Métricas de Melhoria

#### Para Precipitação:
- **Melhoria esperada no R²**: +10-15%
- **Redução do RMSE**: 10-20%
- **Melhoria na detecção de eventos extremos**: +15-25%
- **Redução de falsos alarmes**: 10-15%

#### Para Temperatura:
- **Melhoria esperada no R²**: +5-10%
- **Redução do RMSE**: 5-15%
- **Melhoria na captura de variabilidade espacial**: +10-20%
- **Redução de bias sazonal**: 5-10%

### Benefícios da Integração

1. **Multi-escala**: Combinação de observações locais e regionais
2. **Redundância**: Múltiplas fontes aumentam robustez
3. **Cobertura espacial**: Dados de satélites cobrem áreas sem estações
4. **Resolução temporal**: GPM-IMERG fornece dados diários consistentes
5. **Validação independente**: Satélites como fonte de validação cruzada

## 🔍 Controle de Qualidade

### Filtros de Qualidade - GPM-IMERG
- Remoção de valores fisicamente impossíveis (< 0 ou > 500 mm/dia)
- Filtros baseados em flags de qualidade da NASA
- Detecção de outliers estatísticos
- Consistência temporal (mudanças abruptas)

### Filtros de Qualidade - MODIS LST
- Remoção de píxeis com presença de nuvens
- Filtros de qualidade baseados em metadados
- Validação de range físico de temperaturas
- Suavização temporal para reduzir ruído

## 🌐 Interface Web Atualizada

### Novas Funcionalidades

1. **Upload de Dados Satelitais**:
   - Suporte para diretórios de arquivos NetCDF4
   - Upload de arquivos MODIS GeoTIFF
   - Validação automática de formatos

2. **Configurações de Processamento**:
   - Seleção de fontes de dados (ERA5, Satélites, Estações)
   - Parâmetros de extração espacial
   - Opções de interpolação temporal

3. **Visualizações Expandidas**:
   - Comparação multi-fonte
   - Mapas de cobertura espacial
   - Séries temporais integradas
   - Métricas de qualidade por fonte

## 📋 Dependências Adicionais

### Bibliotecas Python Necessárias

```python
# Adições ao requirements.txt
rasterio>=1.3.0        # Leitura de GeoTIFF
rioxarray>=0.13.0      # Integração xarray+rasterio
pyproj>=3.4.0          # Reprojeções cartográficas
cartopy>=0.21.0        # Visualização cartográfica
h5py>=3.7.0            # Leitura eficiente de HDF5/NetCDF4
h5netcdf>=1.0.0        # Backend NetCDF4 otimizado
```

## 🔧 Configuração de Ambiente

### Estrutura de Diretórios

```
Projeto_modular_downscaling-tool/
├── data/
│   ├── satellites/
│   │   ├── gpm_imerg/         # Links simbólicos para Dados_satelites/dados_nasa/
│   │   └── modis_lst/         # Links simbólicos para Dados_satelites/dados_modis/
│   ├── era5/
│   ├── stations/
│   └── dem/
├── src/
├── models/
└── results/
    └── satellite_integration/  # Resultados da integração
```

## 📈 Cronograma Detalhado

### Semana 1-2: Desenvolvimento Base
- **Dias 1-3**: Implementação `SatelliteDataProcessor`
- **Dias 4-7**: Processamento GPM-IMERG
- **Dias 8-10**: Testes e validação GPM
- **Dias 11-14**: Processamento MODIS LST

### Semana 3: Feature Engineering e Integração
- **Dias 15-17**: Feature engineering satelital
- **Dias 18-21**: Integração ao pipeline ML existente

### Semana 4: Validação e Otimização
- **Dias 22-25**: Testes integrados e validação
- **Dias 26-28**: Otimização e documentação final

## 🎯 Critérios de Sucesso

### Critérios Técnicos
- [ ] Carregamento sucessivo de 100% dos arquivos GPM-IMERG
- [ ] Carregamento sucessivo de 95%+ dos arquivos MODIS LST
- [ ] Sincronização temporal com <1% de dados perdidos
- [ ] Melhoria de performance em pelo menos uma métrica principal

### Critérios de Qualidade
- [ ] Cobertura de testes unitários >80%
- [ ] Documentação técnica completa
- [ ] Interface web funcional com novos recursos
- [ ] Tempo de processamento <2x do pipeline original

### Critérios Científicos
- [ ] Validação cruzada satisfatória entre fontes
- [ ] Análise estatística de melhoria significativa
- [ ] Relatório técnico com resultados e interpretações
- [ ] Recomendações para uso operacional

---

## 📝 Notas de Implementação

### Considerações Técnicas
1. **Memória**: Dados de satélites podem ser volumosos - implementar processamento em chunks
2. **I/O**: Otimizar leitura de arquivos com paralelização quando possível  
3. **Cache**: Implementar cache para dados processados para evitar reprocessamento
4. **Escalabilidade**: Design modular para fácil adição de novas fontes de satélites

### Próximas Expansões
- **GOES-16**: Dados de alta resolução temporal
- **Sentinel-2**: Índices de vegetação
- **SMAP**: Umidade do solo
- **GRACE**: Água subterrânea

Este documento serve como roadmap técnico para a integração completa de dados de satélites ao sistema de downscaling climático existente.