# Sistema de Downscaling Climático com Machine Learning

Um sistema avançado para downscaling estatístico de dados climáticos usando técnicas de Machine Learning, com interface web interativa e processamento automático de dados meteorológicos.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Características](#características)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Uso](#uso)
- [Arquitetura](#arquitetura)
- [Modelos Disponíveis](#modelos-disponíveis)
- [Resultados](#resultados)
- [Licença](#licença)

## 🌍 Visão Geral

Este sistema implementa técnicas de **downscaling estatístico** para aumentar a resolução espacial e temporal de dados climáticos usando modelos de Machine Learning. O projeto processa dados meteorológicos de reanálise (ERA5) e observações de estações locais para gerar predições de alta resolução para temperatura e precipitação.

### Principais Funcionalidades

- **Downscaling de Temperatura**: Predições precisas com R² até 0.537
- **Downscaling de Precipitação**: Modelagem especializada para dados não-lineares
- **Interface Web Interativa**: Flask-based com upload de arquivos e visualização
- **Múltiplos Algoritmos**: 9 modelos diferentes incluindo ensemble
- **Otimização Automática**: Hyperparameter tuning e seleção de features
- **Relatórios Detalhados**: Análise completa de performance e métricas

## ✨ Características

### 🤖 Machine Learning
- **9 Algoritmos**: Linear Regression, Ridge, Random Forest, Extra Trees, Gradient Boosting, XGBoost, SVR, MLP, Ensemble
- **Feature Engineering**: 80+ features derivadas incluindo lags, médias móveis e interações
- **Otimização Automática**: GridSearch e RandomSearch para hyperparameters
- **Ensemble Learning**: Combinação dos melhores modelos para máxima precisão

### 📊 Processamento de Dados
- **Múltiplas Fontes**: ERA5 (reanálise), dados de estação, DEM (topografia)
- **Feature Engineering Avançado**: Variáveis temporais, espaciais e híbridas
- **Validação Temporal**: Split chronológico respeitando a natureza temporal
- **Tratamento de Outliers**: Preprocessing robusto com scaling adaptativo

### 🌐 Interface Web
- **Upload Intuitivo**: Interface drag-and-drop para arquivos
- **Configuração Flexível**: Parâmetros ajustáveis via interface
- **Visualização Rica**: Gráficos interativos de resultados e métricas
- **Download de Resultados**: Export completo em ZIP

### 📈 Visualização
- **Análise Temporal**: Gráficos de séries temporais comparativas
- **Feature Importance**: Ranking de importância das variáveis
- **Métricas Detalhadas**: RMSE, R², MAE, correlação, skill score
- **Comparação de Modelos**: Performance side-by-side

## 🏗 Estrutura do Projeto

```
projeto/
├── src/                          # Código fonte principal
│   ├── core/                     # Componentes centrais
│   │   ├── data_processor.py     # Processamento de dados
│   │   └── weather_model.py      # Modelo principal
│   ├── ml/                       # Machine Learning
│   │   ├── ml_models.py          # Definição dos modelos
│   │   ├── feature_engineering.py # Engenharia de features
│   │   └── model_optimizer.py    # Otimização de hiperparâmetros
│   ├── utils/                    # Utilitários
│   │   ├── utils.py              # Funções auxiliares
│   │   └── report_generator.py   # Geração de relatórios
│   ├── visualization/            # Visualização
│   │   └── visualization.py      # Gráficos e plots
│   └── web/                      # Interface web
│       ├── app.py                # Aplicação Flask
│       ├── templates_generator.py # Geração de templates
│       └── templates/
│           └── index.html        # Interface principal
├── models/                       # Modelos treinados (.pkl)
├── static/                       # Arquivos estáticos
│   └── img/                      # Gráficos gerados
├── results/                      # Relatórios de resultados
├── uploads/                      # Arquivos enviados
├── templates/                    # Templates adicionais
└── requirements.txt              # Dependências
```

## 🔧 Requisitos

### Dependências Principais
```
numpy>=1.21.0
pandas>=1.3.0
xarray>=0.19.0
netCDF4>=1.5.7
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
joblib>=1.0.0
Flask>=2.0.0
Werkzeug>=2.0.0
```

### Dados Necessários
- **ERA5**: Dados de reanálise (temperatura, pressão, vento, etc.)
- **Estação Local**: Observações in-situ de temperatura/precipitação
- **DEM**: Modelo digital de elevação (formato NetCDF)

## 🚀 Instalação

1. **Clone o repositório**:
```bash
git clone https://github.com/seu-usuario/projeto-downscaling.git
cd projeto-downscaling
```

2. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

3. **Configure o ambiente**:
```bash
# Criar diretórios necessários
mkdir -p uploads models results static/img
```

## 📖 Uso

### Modo Web (Recomendado)

1. **Inicie o servidor**:
```bash
python src/web/app.py web
```

2. **Acesse a interface**:
```
http://localhost:5000
```

3. **Upload dos arquivos**:
   - Arquivo ERA5 (CSV)
   - Dados da estação (CSV)
   - Arquivo DEM (NetCDF)

4. **Configure parâmetros**:
   - Latitude/Longitude
   - Data de divisão treino/teste
   - Modelos a usar
   - Configurações de otimização

5. **Execute e baixe resultados**

### Modo Programático

```python
from src.utils.utils import run_downscaling_pipeline_advanced

# Configurar arquivos
files = {
    'era5': 'caminho/era5_data.csv',
    'station': 'caminho/station_data.csv', 
    'dem': 'caminho/elevation.nc'
}

# Parâmetros
params = {
    'latitude': -9.41,
    'longitude': -77.35,
    'split_date': '2018-01-01'
}

# Executar downscaling
results = run_downscaling_pipeline_advanced(
    files, params,
    selected_models=['RandomForest', 'XGBoost', 'MLP'],
    variables=['temperature', 'precipitation'],
    use_ensemble=True,
    optimize_params=True
)
```

## 🏛 Arquitetura

### Pipeline de Processamento

1. **Carregamento de Dados** (`data_processor.py`)
   - Leitura de múltiplas fontes
   - Sincronização temporal
   - Validação de qualidade

2. **Engenharia de Features** (`feature_engineering.py`)
   - Variáveis lag (1, 2, 3, 7, 14 dias)
   - Médias móveis (3, 7, 14, 30 dias)
   - Interações entre variáveis
   - Correções topográficas

3. **Modelagem** (`weather_model.py`)
   - Treinamento de múltiplos algoritmos
   - Validação temporal
   - Seleção de features
   - Criação de ensemble

4. **Visualização** (`visualization.py`)
   - Gráficos temporais
   - Análise de importância
   - Mapas de performance

### Componentes Principais

#### `WeatherDownscalingModel`
Classe principal que orquestra todo o processo:
```python
model = WeatherDownscalingModel(variable='temperature')
model.load_and_merge_data(era5_path, station_path, dem_path, lat, lon)
model.create_features()
model.prepare_data(split_date='2018-01-01')
model.train_models(optimize=True)
model.create_ensemble()
```

#### `ModelBuilder`
Configura os 9 algoritmos de ML com parâmetros otimizados:
- Modelos lineares com regularização
- Ensemble methods (RF, ET, GB)
- Algoritmos avançados (XGBoost, SVM, MLP)

#### `FeatureEngineer`
Cria 80+ features derivadas:
- **Temporais**: lags, médias móveis, tendências
- **Estatísticas**: min, max, std, percentis
- **Híbridas**: interações entre variáveis

## 🤖 Modelos Disponíveis

| Modelo | Tipo | Características | Melhor Para |
|--------|------|----------------|-------------|
| **Linear Regression** | Linear | Rápido, interpretável | Relações lineares simples |
| **Ridge** | Linear + Regularização | Controle de overfitting | Muitas features correlacionadas |
| **Random Forest** | Ensemble de Árvores | Robusto, feature importance | Dados não-lineares |
| **Extra Trees** | Ensemble Randomizado | Menos overfitting | Datasets pequenos |
| **Gradient Boosting** | Boosting Sequencial | Alta precisão | Padrões complexos |
| **XGBoost** | Boosting Otimizado | State-of-the-art | Competições ML |
| **SVR** | Kernel Methods | Não-linear, robusto | Dados com ruído |
| **MLP** | Rede Neural | Aprende representações | Padrões muito complexos |
| **Ensemble** | Combinação | Máxima estabilidade | Produção |

### Performance Típica (Temperatura)

| Modelo | RMSE (°C) | R² | Correlação |
|--------|-----------|-----|------------|
| **Ensemble** | 0.821 | 0.537 | 0.743 |
| Ridge | 0.836 | 0.519 | 0.732 |
| Linear Regression | 0.841 | 0.514 | 0.729 |
| Random Forest | 0.882 | 0.466 | 0.685 |
| XGBoost | 0.888 | 0.457 | 0.677 |


## 📊 Resultados

### Métricas de Avaliação

#### Para Temperatura
- **RMSE**: Root Mean Square Error (°C)
- **R²**: Coeficiente de determinação
- **MAE**: Mean Absolute Error (°C)
- **Correlação**: Correlação de Pearson
- **Skill Score**: Melhoria sobre climatologia

#### Para Precipitação
- **Métricas básicas**: RMSE, R², MAE
- **Detecção de chuva**: Accuracy, POD, FAR
- **R² para dias chuvosos**: Performance específica

### Exemplo de Relatório

```
🏆 MELHOR MODELO: Ensemble
   • RMSE: 0.821 °C
   • R²: 0.537
   • Melhoria sobre climatologia: 53.7%

📈 INTERPRETAÇÃO:
   • Qualidade da predição: Moderada (R² = 0.537)
   • O modelo é 53.7% melhor que a climatologia
```

### Features Mais Importantes

1. **t2m (Temperatura 2m)**: Variável principal
2. **t2m_lag_1**: Temperatura do dia anterior
3. **sp (Pressão)**: Condições atmosféricas
4. **t2m_ma_7**: Média móvel 7 dias
5. **z_500 (Geopotencial)**: Padrões de circulação

## 🔬 Casos de Uso

### Aplicações Científicas
- **Climatologia Regional**: Estudos de mudanças climáticas
- **Hidrologia**: Modelagem de bacias hidrográficas
- **Agricultura**: Zoneamento agroclimático
- **Gestão de Recursos**: Planejamento hídrico

### Aplicações Operacionais
- **Previsão Sazonal**: Projeções de médio prazo
- **Estudos de Impacto**: Avaliação de vulnerabilidades
- **Planejamento Urbano**: Adaptação às mudanças climáticas

## 🛠 Desenvolvimento

### Estrutura de Classes

```python
# Core
DataProcessor      # Carregamento e preprocessamento
WeatherModel       # Orquestração principal
FeatureEngineer    # Criação de features

# ML
ModelBuilder       # Configuração de modelos
ModelOptimizer     # Otimização de hiperparâmetros

# Utils
ReportGenerator    # Geração de relatórios
Visualization      # Gráficos e plots

# Web
Flask App          # Interface web
TemplateGenerator  # HTML dinâmico
```

### Extensibilidade

#### Adicionar Novo Modelo
```python
# Em ml_models.py
models['NovoModelo'] = Pipeline([
    ('scaler', StandardScaler()),
    ('model', NovoAlgoritmo(**params))
])
```

#### Nova Feature
```python
# Em feature_engineering.py
def create_nova_feature(self, data):
    data['nova_feature'] = data['var1'] * data['var2']
    return data
```

## 📈 Roadmap

### Versão Atual (v1.0)
- ✅ Downscaling básico temperatura/precipitação
- ✅ Interface web completa
- ✅ 9 algoritmos de ML
- ✅ Ensemble learning
- ✅ Otimização automática

### Próximas Versões

#### v1.1 - Otimização
- [ ] Integração com Optuna
- [ ] Paralelização de modelos
- [ ] Cache de resultados
- [ ] Validação cruzada temporal

#### v1.2 - Expansão
- [ ] Suporte a mais variáveis (umidade, vento)
- [ ] Modelos deep learning (LSTM, CNN)
- [ ] API REST completa
- [ ] Dashboard analytics

#### v2.0 - Produção
- [ ] Containerização Docker
- [ ] Deploy em cloud
- [ ] Monitoramento de performance
- [ ] Versionamento de modelos

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.


## 🙏 Agradecimentos

- **ERA5 Reanalysis**: Dados do European Centre for Medium-Range Weather Forecasts
- **Scikit-learn**: Framework de Machine Learning
- **Flask**: Framework web para Python
- **Comunidade Open Source**: Pelas bibliotecas e ferramentas utilizadas

---

**📍 Status do Projeto**: ✅ Ativo | 🔄 Em desenvolvimento | 📊 Produção ready