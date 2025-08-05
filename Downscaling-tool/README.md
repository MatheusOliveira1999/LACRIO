

# 🌦️ Sistema Completo de Downscaling Climático com Machine Learning

Um sistema avançado para aprimorar dados climáticos de baixa resolução usando técnicas de Machine Learning, especialmente desenvolvido para regiões montanhosas.

## 📋 Índice

- [Características](#-características)
- [Requisitos](#-requisitos)
- [Instalação](#-instalação)
- [Uso](#-uso)
  - [Interface Web](#interface-web)
  - [Modo Standalone](#modo-standalone)
- [Estrutura dos Dados](#-estrutura-dos-dados)
- [Modelos Disponíveis](#-modelos-disponíveis)
- [Configurações Avançadas](#-configurações-avançadas)
- [Resultados e Visualizações](#-resultados-e-visualizações)
- [Exemplos](#-exemplos)
- [Troubleshooting](#-troubleshooting)
- [Contribuição](#-contribuição)
- [Licença](#-licença)

## 🚀 Características

### Principais Funcionalidades
- **Downscaling de múltiplas variáveis**: Temperatura e precipitação
- **8+ algoritmos de ML**: Random Forest, XGBoost, MLP, SVM, Gradient Boosting, etc.
- **Interface web intuitiva**: Sistema completo baseado em Flask
- **Otimização automática**: Grid Search e Random Search para hiperparâmetros
- **Feature engineering avançado**: 100+ features meteorológicas e topográficas
- **Validação temporal**: Time Series Split para dados climáticos
- **Ensemble learning**: Combinação dos melhores modelos
- **Visualizações completas**: 15+ tipos de gráficos e análises

### Características Técnicas
- Processamento de dados ERA5/ERA5-Land
- Integração com modelos de elevação digital (DEM)
- Correções topográficas para regiões montanhosas
- Métricas específicas para cada variável climática
- Sistema de relatórios automatizado
- Exportação de modelos treinados

## 📦 Requisitos

### Dependências Python
```
numpy >= 1.21.0
pandas >= 1.3.0
xarray >= 0.19.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
flask >= 2.0.0
joblib >= 1.1.0
scipy >= 1.7.0
werkzeug >= 2.0.0
```

### Dependências Opcionais
```
xgboost >= 1.5.0  # Para algoritmo XGBoost
optuna >= 3.0.0   # Para otimização avançada
```

### Requisitos de Sistema
- Python 3.8+
- RAM: 8GB mínimo (16GB recomendado)
- Armazenamento: 2GB livres
- Sistema: Windows/Linux/macOS
1. Clone o repositório
2. Crie um ambiente virtual com python 3.11:
   ```bash
   conda create -n temp-predict-py311 python=3.11
3. Ative o ambiente virtual 
4. Instale as dependências 
    ```bash
    pip install -r requirements.txt
## 🔧 Instalação

### 1. Clonar/Baixar o Arquivo

### 2. Crie um ambiente virtual com python 3.11:
```bash
   conda create -n temp-predict-py311 python=3.11
   ```

### 3. Crie um ambiente virtual com python 3.11:

```bash
    pip install -r requirements.txt
```


## 🎯 Uso

### Interface Web

**1. Iniciar o servidor:**
```bash
python sistema_downscaling.py web
```

**2. Acessar:** http://localhost:5000

**3. Seguir os passos na interface:**
- Upload dos arquivos (ERA5, estação, DEM)
- Configurar parâmetros
- Selecionar modelos e variáveis
- Configurar otimização
- Processar dados

### Modo Standalone

```python
from sistema_downscaling import run_weather_downscaling

# Executar downscaling
results = run_weather_downscaling(
    era5_path="dados_era5.csv",
    station_path="dados_estacao.csv", 
    dem_path="elevacao.nc",
    variables=['temperature', 'precipitation'],
    lat=-9.41,
    lon=-77.35,
    split_date='2018-01-01',
    selected_models=['RandomForest', 'XGBoost', 'MLP'],
    optimize_models=True,
    use_ensemble=True
)
```

## 📊 Estrutura dos Dados

### Arquivo ERA5/ERA5-Land (CSV)
```csv
date,t2m,tp,sp,u10,v10,d2m,r,ssrd,strd,z
2015-01-01,15.2,0.0,85000,2.1,-1.5,12.3,0.85,18500000,28000000,850000
2015-01-02,16.1,2.3,84900,1.8,-2.1,13.1,0.78,17800000,27500000,849500
...
```

**Variáveis necessárias:**
- `date`: Data (YYYY-MM-DD)
- `t2m`: Temperatura 2m (°C)
- `tp`: Precipitação total (mm)
- `sp`: Pressão superficial (Pa)
- `u10`, `v10`: Componentes do vento (m/s)
- `d2m`: Temperatura ponto de orvalho (°C)
- `r`: Umidade relativa (0-1)
- `ssrd`: Radiação solar descendente (J/m²)
- `strd`: Radiação térmica descendente (J/m²)
- `z`: Geopotencial (m²/s²)

### Arquivo da Estação (CSV)
```csv
data,Temperatura,Precipitação
2015-01-01,18.5,0.0
2015-01-02,19.2,5.2
...
```

**Formatos aceitos:**
- Nomes de colunas: `data/Data/DATE` para data
- Temperatura: `Temperatura/temperatura/Temperature/TEMP`
- Precipitação: `Precipitação/precipitacao/Precipitation/PREC/Rain`

### Arquivo DEM (NetCDF)
Modelo digital de elevação contendo:
- Variável de elevação: `HGT/elevation/dem/z/height`
- Coordenadas: `lat`, `lon`

## 🤖 Modelos Disponíveis

### Algoritmos Implementados

| Modelo | Tipo | Otimização | Tempo* |
|--------|------|------------|--------|
| **Linear Regression** | Linear | ✅ | ~1 min |
| **Ridge** | Linear regularizado | ✅ | ~3 min |
| **Random Forest** | Ensemble | ✅ | ~15 min |
| **Extra Trees** | Ensemble | ✅ | ~15 min |
| **Gradient Boosting** | Boosting | ✅ | ~20 min |
| **XGBoost** | Boosting avançado | ✅ | ~10 min |
| **SVM (SVR)** | Kernel | ✅ | ~8 min |
| **MLP** | Rede Neural | ✅ | ~12 min |

*Tempo aproximado com otimização rápida para 1 variável

### Ensemble Learning
O sistema combina automaticamente os 3 melhores modelos usando `VotingRegressor`.

## ⚙️ Configurações Avançadas

### Modos de Otimização

```python
config = {
    'optimize_hyperparams': True,
    'fast_optimization': True,          # Grid reduzido
    'use_random_search': False,         # RandomizedSearchCV
    'random_search_iter': 30,           # Iterações do Random Search
    'optimize_all_models': True,        # Otimizar todos vs apenas alguns
    'cv_folds': 3,                     # Folds para validação cruzada
    'feature_selection': True,          # Seleção automática de features
    'max_features': 100,               # Máximo de features
    'ensemble_size': 3                 # Tamanho do ensemble
}
```

### Feature Engineering

O sistema cria automaticamente 100+ features:

**Temporais:**
- Ciclos sazonais (sin/cos)
- Tendências lineares
- Dia do ano, mês, estação

**Lag Features:**
- Valores anteriores (1, 2, 3, 7, 14 dias)
- Médias móveis (3, 7, 14, 30 dias)
- Desvios padrão móveis

**Interações:**
- Velocidade do vento (u² + v²)
- Déficit de pressão de vapor
- Efeitos orográficos

**Topográficas:**
- Correção altimétrica (lapse rate)
- Efeito orográfico na precipitação
- Amplitude térmica estimada

## 📈 Resultados e Visualizações

### Métricas de Avaliação

**Para todas as variáveis:**
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² (Coeficiente de determinação)
- Correlação de Pearson
- Bias médio
- Skill Score (vs climatologia)

**Específicas para precipitação:**
- Accuracy de detecção de chuva
- POD (Probability of Detection)
- FAR (False Alarm Rate)
- R² apenas para dias chuvosos

### Gráficos Gerados

1. **Comparação de modelos** - Barras com métricas
2. **Observado vs Predito** - Scatter plots
3. **Séries temporais** - Evolução no tempo
4. **Distribuição de resíduos** - Histogramas e Q-Q plots
5. **Importância de features** - Ranking de variáveis
6. **Análise temporal** - Padrões sazonais e mensais
7. **Heatmaps** - Correlações e métricas normalizadas

### Arquivos de Saída

```
📁 results/
   ├── relatorio_downscaling_temperature.txt
   └── relatorio_downscaling_precipitation.txt

📁 static/img/
   ├── resultados_downscaling_temperature.png
   ├── comparacao_modelos_temperature.png
   ├── feature_importance_temperature_randomforest.png
   └── analise_temporal_temperature.png

📁 models/
   ├── randomforest_temperature.pkl
   └── xgboost_precipitation.pkl
```

## 💡 Exemplos

### Exemplo 1: Processamento Básico

```python
import sistema_downscaling as sd

# Configuração simples
results = sd.run_weather_downscaling(
    era5_path="era5_data.csv",
    station_path="station_data.csv",
    dem_path="elevation.nc",
    variables=['temperature'],
    selected_models=['RandomForest', 'XGBoost'],
    optimize_models=False  # Mais rápido
)

# Acessar melhor modelo
best_model = results['temperature']['model']
print(f"Melhor modelo: {results['temperature']['best_model_name']}")
```

### Exemplo 2: Configuração Avançada

```python
# Configuração para pesquisa científica
config = {
    'optimize_hyperparams': True,
    'fast_optimization': False,        # Otimização completa
    'cv_folds': 5,                    # Mais validação
    'feature_selection': True,
    'ensemble_size': 5                # Ensemble maior
}

model = sd.WeatherDownscalingModel('temperature', config)
model.load_and_merge_data(era5_path, station_path, dem_path, lat, lon)
model.create_features()
model.prepare_data('2018-01-01')
model.train_models(optimize=True)
model.create_ensemble()
model.generate_report()
```

### Exemplo 3: Predição em Novos Dados

```python
# Carregar modelo salvo
model = sd.WeatherDownscalingModel('temperature')
trained_model = model.load_model('models/randomforest_temperature.pkl')

# Fazer predições
predictions = model.predict_new_data('novos_dados.csv')
print(predictions.head())
```

## 🔧 Troubleshooting

### Problemas Comuns

**1. Erro de convergência (MLP):**
```
Solução: Ativar 'fast_optimization' ou reduzir max_iter
```

**2. Memória insuficiente:**
```
Solução: Reduzir max_features ou usar menos modelos simultaneamente
```

**3. XGBoost não disponível:**
```bash
pip install xgboost
```

**4. Dados insuficientes:**
```
Mínimo: 365 registros (1 ano)
Recomendado: 1095+ registros (3+ anos)
```

**5. Features com escala muito grande:**
```
O sistema normaliza automaticamente, mas verifique:
- Pressão deve estar em Pa (não hPa)
- Radiação em J/m² (não W/m²)
```

### Otimização de Performance

**Para datasets grandes (>10 anos):**
```python
config = {
    'fast_optimization': True,
    'cv_folds': 3,
    'max_features': 50,
    'ensemble_size': 3
}
```

**Para máxima precisão:**
```python
config = {
    'fast_optimization': False,
    'cv_folds': 5,
    'use_random_search': True,
    'random_search_iter': 100
}
```

## 📚 Referências Científicas

O sistema implementa técnicas baseadas em:

1. **Downscaling estatístico:** Wilby & Wigley (1997)
2. **Machine Learning em climatologia:** Sachindra et al. (2018)
3. **Ensemble methods:** Breiman (2001)
4. **Correções topográficas:** Barry (2008)
5. **Validação temporal:** Bergmeir & Benítez (2012)


## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.


**🌟 Se este projeto foi útil, considere dar uma estrela!**