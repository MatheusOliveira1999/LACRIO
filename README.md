# LACRIO
Desenvolvimento de uma ferramenta automatizada para melhorar dados climáticos gerados por modelos numéricos e reanálise globais

## Visão Geral do Projeto

LACRIO é um toolkit abrangente de downscaling climático focado em melhorar dados de modelos meteorológicos numéricos e reanálise global. O repositório contém múltiplos módulos interconectados para processamento, análise e downscaling baseado em machine learning de dados meteorológicos.

## Arquitetura de Alto Nível

O repositório está organizado em quatro componentes principais:

### 1. Pre-processamento_UTC/
**Sistema de pré-processamento de dados meteorológicos**
- Processa dados de reanálise ERA5/ERA5-Land e observações de estações meteorológicas locais
- Converte horário local do Peru (UTC-5) para UTC para alinhamento temporal
- Combina múltiplas fontes de dados em arquivos CSV unificados

### 2. Pre_downscaling_comparison/
**Sistema de análise de qualidade e validação de dados**
- Comparação de dados pré-downscaling entre observações ERA5 e estações meteorológicas
- Gera relatórios abrangentes com análise estatística e visualizações
- Interface CLI para processamento automatizado
- Integra-se com o sistema principal de downscaling

### 3. Downscaling-tool/
**Sistema principal de downscaling baseado em ML**
- Downscaling estatístico avançado usando 9 algoritmos de machine learning
- Engenharia de features com 80+ variáveis derivadas incluindo lags, médias móveis e interações
- Interface CLI para processamento automatizado
- Ensemble learning combinando múltiplos modelos

### 4. Dados_satelites/
**Integração de dados de satélites (em desenvolvimento)**
- Processamento de dados de precipitação NASA GPM-IMERG
- Integração de temperatura de superfície MODIS
- Expansão planejada para o sistema de downscaling

## Comandos Comuns de Desenvolvimento

### Pre-processamento_UTC
```bash
# Processar dados diários com conversão UTC
python processar_dados.py \
    --era5_dir "/caminho/para/era5" \
    --era5_land_dir "/caminho/para/era5_land" \
    --station_file "/caminho/para/estacao.csv" \
    --lat -9.5 --lon -77.5

# Processar dados horários
python processar_dados_horario.py \
    --era5_dir "/caminho/para/era5" \
    --era5_land_dir "/caminho/para/era5_land" \
    --station_file "/caminho/para/estacao.csv" \
    --lat -9.5 --lon -77.5

# Interface web para processamento de dados
python app.py  # ou app_dados_horarios.py para dados horários
```

### Pre_downscaling_comparison
```bash
# Comparação básica de dados
python run_comparison.py \
    --era5 dados_era5.csv \
    --station dados_estacao.csv \
    --station-name "Llanganuco"

# Comparação avançada com relatório
python run_comparison.py \
    --era5 dados_era5.csv \
    --station dados_estacao.csv \
    --station-name "Llanganuco" \
    --variables temperature precipitation \
    --generate-report \
    --output-dir resultados

# Instalação de dependências
pip install -r requirements.txt
```

### Pre_downscaling_comparison
```bash
# Ativar ambiente conda
conda activate temp-predict

# Execução básica de downscaling
python run_downscaling.py \
    --era5 dados.csv \
    --station estacao.csv \
    --dem elevacao.nc \
    --lat -9.41 --lon -77.35

# Execução avançada com otimização
python run_downscaling.py \
    --era5 dados.csv \
    --station estacao.csv \
    --dem elevacao.nc \
    --lat -9.41 --lon -77.35 \
    --models RandomForest XGBoost MLP \
    --optimize \
    --variables temperature precipitation

# Testar módulo de validação
/home/matheus/miniconda3/envs/temp-predict/bin/python -c "
import sys; sys.path.append('.')
from analyses.validacao.validador_downscaling import ValidadorDownscaling
print('✅ Módulo de validação importado com sucesso')
"
```

## Fluxo do Pipeline de Dados

### Workflow Típico:
1. **Processamento de Dados Brutos** (Pre-processamento_UTC): Limpar e alinhar dados temporais
2. **Análise de Qualidade** (Pre_downscaling_comparison): Validar qualidade dos dados e gerar relatórios
3. **Downscaling** (Projeto_modular_downscaling-tool2): Aplicar modelos ML para predições de alta resolução
4. **Integração de Satélites** (Futuro): Incorporar dados de sensoriamento remoto

### Requisitos de Dados:
- **ERA5/ERA5-Land**: Arquivos NetCDF com variáveis atmosféricas e de superfície
- **Estações Meteorológicas**: Arquivos CSV com colunas: `Datetime`, `Temperature (°C)`, `Precipitation (mm)`
- **DEM**: Modelo Digital de Elevação em formato NetCDF para correções topográficas
- **Coordenadas**: Latitude/longitude da localização alvo

## Detalhes Importantes de Implementação

### Manipulação Temporal
- Todos os sistemas usam UTC como fuso horário padrão
- Pre-processamento_UTC trata conversão do horário local do Peru (UTC-5)
- Validação temporal previne vazamento de dados em modelos ML
- Divisão padrão treino/teste: antes/depois de 2018-01-01

### Engenharia de Features
- Cria 80+ features derivadas incluindo lags temporais (1,2,3,7,14 dias)
- Médias móveis (3,7,14,30 dias) e agregações estatísticas
- Correções topográficas usando dados DEM
- Escalonamento e pré-processamento automático

### Arquitetura de Modelos
- 9 algoritmos ML: Linear Regression, Ridge, Random Forest, Extra Trees, Gradient Boosting, XGBoost, SVR, MLP, Ensemble
- Validação temporal com divisões cronológicas
- Otimização de hiperparâmetros via GridSearch/RandomSearch
- Meta-learning ensemble combinando os melhores performers

### Estrutura de Saída
```
models/{localização}/          # Modelos ML treinados (arquivos .pkl)
results/{localização}/         # Relatórios de performance e métricas
static/img/{localização}/      # Visualizações e gráficos gerados
uploads/                       # Área de staging de dados de entrada
```

## Requisitos de Ambiente

### Dependências Python (varia por módulo):
- **Core**: `numpy`, `pandas`, `xarray`, `netCDF4`
- **ML**: `scikit-learn`, `xgboost`
- **Visualização**: `matplotlib`, `seaborn`
- **Web**: `Flask`, `Werkzeug` (para interfaces web)
- **Temporal**: `pytz` (para manipulação de fuso horário)

### Formatos de Dados:
- **Entrada**: CSV (estações meteorológicas), NetCDF (ERA5, DEM), NetCDF4 (satélites)
- **Saída**: CSV (dados processados), PKL (modelos treinados), PNG (visualizações), MD/HTML (relatórios)

## Contexto Regional

Este sistema é especificamente projetado para pesquisa climática na região da Cordilheira Branca do Peru, focando em:
- Estações meteorológicas de alta altitude nos Andes tropicais
- Efeitos topográficos complexos nas variáveis climáticas
- Conversão entre dados de reanálise global e observações locais
- Aplicações agrícolas e hidrológicas em regiões com poucos dados

Ao trabalhar com este código, certifique-se de ativar o ambiente adequado (especialmente o ambiente conda `temp-predict` para a ferramenta principal de downscaling) e verificar se todos os arquivos de dados necessários estão disponíveis antes de executar os pipelines de processamento.