# Projeto de Comparação de Dados Climáticos Pré-Downscaling

## 📋 Visão Geral

Este projeto visa criar um sistema de **comparação e análise de dados climáticos** antes do processo de downscaling. O sistema analisa os dados originais do ERA5 e das estações meteorológicas, gerando gráficos comparativos do período completo e relatórios detalhados para subsidiar o processo de downscaling.

## 🎯 Objetivos

- **Análise Comparativa**: Comparar dados ERA5 vs. observações das estações meteorológicas
- **Visualização Temporal**: Gráficos do período completo de dados disponíveis
- **Relatório Detalhado**: Análise estatística e qualitativa dos dados
- **Interface Web**: Sistema web para upload, processamento e visualização
- **Integração**: Executar antes do sistema de downscaling existente

## 🏗️ Arquitetura do Sistema

### Estrutura de Diretórios
```
Pre_downscaling_comparison/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Carregamento de dados ERA5 e estação
│   │   └── data_comparator.py      # Análise e comparação dos dados
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── statistical_analysis.py # Análises estatísticas
│   │   └── temporal_analysis.py    # Análises temporais
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── comparative_plots.py    # Gráficos comparativos
│   │   └── report_charts.py        # Gráficos para relatórios
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── report_generator.py     # Geração de relatórios
│   │   └── templates/
│   │       ├── report_template.md  # Template do relatório
│   │       └── summary_template.md # Template resumo
│   ├── web/
│   │   ├── __init__.py
│   │   ├── app.py                  # Aplicação Flask
│   │   ├── routes.py               # Rotas da aplicação
│   │   └── templates/
│   │       ├── index.html          # Interface principal
│   │       ├── results.html        # Página de resultados
│   │       └── report.html         # Visualização do relatório
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Configurações do sistema
│       ├── validators.py           # Validação de dados
│       └── helpers.py              # Funções auxiliares
├── static/
│   ├── css/
│   │   └── styles.css              # Estilos da aplicação
│   ├── js/
│   │   └── main.js                 # JavaScript da aplicação
│   └── img/                        # Imagens geradas
├── results/
│   ├── reports/                    # Relatórios gerados
│   ├── plots/                      # Gráficos salvos
│   └── summaries/                  # Resumos estatísticos
├── uploads/                        # Arquivos enviados
├── tests/                          # Testes unitários
├── requirements.txt                # Dependências
├── config.yaml                     # Configuração principal
└── README.md                       # Documentação
```

## 🔧 Componentes Principais

### 1. Carregamento de Dados (`data_loader.py`)
```python
class DataLoader:
    """Carrega e sincroniza dados ERA5 e estações meteorológicas"""
    
    def load_era5_data(self, filepath: str) -> pd.DataFrame:
        """Carrega dados ERA5"""
        
    def load_station_data(self, filepath: str) -> pd.DataFrame:
        """Carrega dados da estação"""
        
    def synchronize_datasets(self, era5_df: pd.DataFrame, 
                           station_df: pd.DataFrame) -> tuple:
        """Sincroniza datasets por período comum"""
        
    def validate_data_quality(self, df: pd.DataFrame) -> dict:
        """Valida qualidade dos dados"""
```

### 2. Comparação de Dados (`data_comparator.py`)
```python
class DataComparator:
    """Compara dados ERA5 vs. estações meteorológicas"""
    
    def compare_variables(self, era5_df: pd.DataFrame, 
                         station_df: pd.DataFrame) -> dict:
        """Compara variáveis climáticas"""
        
    def calculate_bias_metrics(self, observed: pd.Series, 
                              predicted: pd.Series) -> dict:
        """Calcula métricas de bias"""
        
    def seasonal_comparison(self, era5_df: pd.DataFrame, 
                           station_df: pd.DataFrame) -> dict:
        """Análise sazonal comparativa"""
        
    def extreme_events_analysis(self, era5_df: pd.DataFrame, 
                               station_df: pd.DataFrame) -> dict:
        """Análise de eventos extremos"""
```

### 3. Análise Estatística (`statistical_analysis.py`)
```python
class StatisticalAnalysis:
    """Análises estatísticas dos dados"""
    
    def correlation_analysis(self, era5_data: pd.Series, 
                           station_data: pd.Series) -> dict:
        """Análise de correlação"""
        
    def bias_analysis(self, era5_data: pd.Series, 
                     station_data: pd.Series) -> dict:
        """Análise de bias"""
        
    def distribution_analysis(self, era5_data: pd.Series, 
                            station_data: pd.Series) -> dict:
        """Análise de distribuições"""
        
    def trend_analysis(self, era5_data: pd.Series, 
                      station_data: pd.Series) -> dict:
        """Análise de tendências"""
```

### 4. Análise Temporal (`temporal_analysis.py`)
```python
class TemporalAnalysis:
    """Análises temporais específicas"""
    
    def daily_cycle_analysis(self, hourly_data: pd.DataFrame) -> dict:
        """Análise do ciclo diário"""
        
    def seasonal_cycle_analysis(self, daily_data: pd.DataFrame) -> dict:
        """Análise do ciclo sazonal"""
        
    def interannual_variability(self, yearly_data: pd.DataFrame) -> dict:
        """Análise de variabilidade interanual"""
        
    def missing_data_analysis(self, data: pd.DataFrame) -> dict:
        """Análise de dados faltantes"""
```

### 5. Visualização (`comparative_plots.py`)
```python
class ComparativePlots:
    """Geração de gráficos comparativos"""
    
    def time_series_comparison(self, era5_data: pd.Series, 
                              station_data: pd.Series, 
                              variable: str) -> None:
        """Série temporal comparativa do período completo"""
        
    def scatter_plot_comparison(self, era5_data: pd.Series, 
                               station_data: pd.Series, 
                               variable: str) -> None:
        """Gráfico de dispersão ERA5 vs. Estação"""
        
    def monthly_boxplots(self, era5_data: pd.Series, 
                        station_data: pd.Series, 
                        variable: str) -> None:
        """Box plots mensais comparativos"""
        
    def bias_analysis_plots(self, bias_data: pd.Series, 
                           variable: str) -> None:
        """Gráficos de análise de bias"""
        
    def seasonal_comparison_plots(self, seasonal_data: dict, 
                                 variable: str) -> None:
        """Gráficos de comparação sazonal"""
        
    def distribution_comparison(self, era5_data: pd.Series, 
                               station_data: pd.Series, 
                               variable: str) -> None:
        """Comparação de distribuições"""
```

### 6. Geração de Relatórios (`report_generator.py`)
```python
class ReportGenerator:
    """Geração de relatórios detalhados"""
    
    def generate_full_report(self, analysis_results: dict, 
                           output_path: str) -> str:
        """Gera relatório completo em Markdown"""
        
    def generate_summary_report(self, analysis_results: dict) -> str:
        """Gera resumo executivo"""
        
    def generate_html_report(self, markdown_content: str) -> str:
        """Converte relatório para HTML"""
        
    def create_comparison_table(self, statistics: dict) -> str:
        """Cria tabela comparativa de estatísticas"""
```

### 7. Interface Web (`app.py`)
```python
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

@app.route('/')
def index():
    """Página principal de upload"""
    
@app.route('/upload', methods=['POST'])
def upload_files():
    """Upload de arquivos ERA5 e estação"""
    
@app.route('/process', methods=['POST'])
def process_comparison():
    """Processa comparação de dados"""
    
@app.route('/results')
def show_results():
    """Exibe resultados da análise"""
    
@app.route('/download_report')
def download_report():
    """Download do relatório completo"""
```

## 📊 Análises a Implementar

### 1. **Análise de Série Temporal Completa**
- Gráfico do período completo (ERA5 vs. Estação)
- Identificação de tendências e sazonalidade
- Análise de consistência temporal
- Detecção de outliers e anomalias

### 2. **Análise Estatística Descritiva**
- Estatísticas básicas (média, mediana, desvio padrão)
- Análise de correlação (Pearson, Spearman)
- Teste de normalidade (Shapiro-Wilk, Kolmogorov-Smirnov)
- Análise de bias (sistemático e aleatório)

### 3. **Análise de Distribuições**
- Histogramas comparativos
- Q-Q plots
- Análise de percentis
- Teste de aderência de distribuições

### 4. **Análise Sazonal**
- Comparação por estações do ano
- Ciclo diário (se dados horários)
- Variabilidade mensal
- Padrões sazonais de bias

### 5. **Análise de Eventos Extremos**
- Identificação de percentis extremos (95%, 99%)
- Comparação de valores máximos e mínimos
- Análise de eventos de precipitação intensa
- Ondas de calor e frio

### 6. **Análise de Qualidade dos Dados**
- Detecção de dados faltantes
- Identificação de valores suspeitos
- Análise de consistência
- Recomendações de pré-processamento

## 📋 Template do Relatório

### Estrutura do Relatório Markdown:

```markdown
# Relatório de Comparação Pré-Downscaling

## 1. Resumo Executivo
- Período analisado
- Variáveis comparadas
- Principais achados
- Recomendações

## 2. Descrição dos Dados
### 2.1 Dados ERA5
- Período: [data_inicio] a [data_fim]
- Variáveis: [lista_variaveis]
- Resolução temporal: [resolucao]
- Estatísticas básicas

### 2.2 Dados da Estação
- Estação: [nome_estacao]
- Localização: [lat], [lon]
- Período: [data_inicio] a [data_fim]
- Variáveis: [lista_variaveis]
- Estatísticas básicas

## 3. Análise Comparativa

### 3.1 Temperatura
#### Estatísticas Básicas
[tabela_estatisticas_temperatura]

#### Análise de Correlação
- Correlação de Pearson: [valor]
- Correlação de Spearman: [valor]
- Significância estatística: [p_value]

#### Análise de Bias
- Bias médio: [valor] °C
- RMSE: [valor] °C
- MAE: [valor] °C
- Coeficiente de variação: [valor]

#### Análise Sazonal
[resultados_analise_sazonal]

### 3.2 Precipitação
[similar_structure_for_precipitation]

## 4. Análise Temporal
### 4.1 Série Temporal Completa
[descricao_serie_temporal]

### 4.2 Variabilidade Interanual
[analise_variabilidade]

### 4.3 Tendências
[analise_tendencias]

## 5. Análise de Eventos Extremos
### 5.1 Temperatura
[analise_extremos_temperatura]

### 5.2 Precipitação
[analise_extremos_precipitacao]

## 6. Qualidade dos Dados
### 6.1 Dados Faltantes
[analise_dados_faltantes]

### 6.2 Outliers
[analise_outliers]

### 6.3 Consistência
[analise_consistencia]

## 7. Recomendações para Downscaling
### 7.1 Pré-processamento
[recomendacoes_preprocessamento]

### 7.2 Variáveis Prioritárias
[variaveis_prioritarias]

### 7.3 Período de Treinamento
[recomendacao_periodo_treino]

### 7.4 Considerações Especiais
[consideracoes_especiais]

## 8. Conclusões
[conclusoes_principais]

## 9. Anexos
### 9.1 Gráficos Adicionais
[lista_graficos]

### 9.2 Tabelas Estatísticas
[tabelas_detalhadas]
```

## 🌐 Interface Web

### Página Principal (`index.html`)
```html
<!DOCTYPE html>
<html>
<head>
    <title>Comparação Pré-Downscaling</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
</head>
<body>
    <div class="container">
        <h1>🌍 Sistema de Comparação Pré-Downscaling</h1>
        
        <div class="upload-section">
            <h2>📁 Upload de Dados</h2>
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="file-input-group">
                    <label for="era5File">Dados ERA5 (CSV):</label>
                    <input type="file" id="era5File" name="era5" accept=".csv" required>
                </div>
                
                <div class="file-input-group">
                    <label for="stationFile">Dados da Estação (CSV):</label>
                    <input type="file" id="stationFile" name="station" accept=".csv" required>
                </div>
                
                <div class="parameters-section">
                    <h3>⚙️ Parâmetros de Análise</h3>
                    
                    <div class="input-group">
                        <label for="stationName">Nome da Estação:</label>
                        <input type="text" id="stationName" name="station_name" required>
                    </div>
                    
                    <div class="input-group">
                        <label for="latitude">Latitude:</label>
                        <input type="number" id="latitude" name="latitude" step="0.01" required>
                    </div>
                    
                    <div class="input-group">
                        <label for="longitude">Longitude:</label>
                        <input type="number" id="longitude" name="longitude" step="0.01" required>
                    </div>
                    
                    <div class="checkbox-group">
                        <h4>Variáveis a Analisar:</h4>
                        <label><input type="checkbox" name="variables" value="temperature" checked> Temperatura</label>
                        <label><input type="checkbox" name="variables" value="precipitation" checked> Precipitação</label>
                        <label><input type="checkbox" name="variables" value="humidity"> Umidade</label>
                        <label><input type="checkbox" name="variables" value="pressure"> Pressão</label>
                    </div>
                    
                    <div class="checkbox-group">
                        <h4>Análises a Incluir:</h4>
                        <label><input type="checkbox" name="analyses" value="correlation" checked> Análise de Correlação</label>
                        <label><input type="checkbox" name="analyses" value="bias" checked> Análise de Bias</label>
                        <label><input type="checkbox" name="analyses" value="seasonal" checked> Análise Sazonal</label>
                        <label><input type="checkbox" name="analyses" value="extremes" checked> Eventos Extremos</label>
                        <label><input type="checkbox" name="analyses" value="trends" checked> Análise de Tendências</label>
                    </div>
                </div>
                
                <button type="submit" class="btn-primary">🚀 Iniciar Análise</button>
            </form>
        </div>
        
        <div id="progressSection" class="progress-section" style="display: none;">
            <h2>📊 Processando Análise...</h2>
            <div class="progress-bar">
                <div id="progressBar" class="progress-fill"></div>
            </div>
            <p id="progressText">Iniciando processamento...</p>
        </div>
        
        <div id="resultsSection" class="results-section" style="display: none;">
            <h2>📈 Resultados da Análise</h2>
            <div id="resultsContent"></div>
            <div class="action-buttons">
                <button id="viewReportBtn" class="btn-secondary">📋 Ver Relatório Completo</button>
                <button id="downloadBtn" class="btn-primary">⬇️ Download Resultados</button>
                <button id="proceedDownscalingBtn" class="btn-success">➡️ Prosseguir para Downscaling</button>
            </div>
        </div>
    </div>
    
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
```

## 🔧 Configuração (`config.yaml`)
```yaml
# Configuração do Sistema de Comparação Pré-Downscaling

app:
  name: "Sistema de Comparação Pré-Downscaling"
  version: "1.0.0"
  debug: true
  port: 5001
  max_file_size: 100  # MB

paths:
  uploads: "./uploads"
  results: "./results"
  plots: "./static/img"
  reports: "./results/reports"
  templates: "./src/reporting/templates"

analysis:
  variables:
    temperature:
      units: "°C"
      min_valid: -50
      max_valid: 60
      name: "Temperatura"
    precipitation:
      units: "mm"
      min_valid: 0
      max_valid: 1000
      name: "Precipitação"

statistics:
  correlation_methods: ["pearson", "spearman"]
  extreme_percentiles: [95, 99, 99.9]
  seasonal_months:
    DJF: [12, 1, 2]  # Verão (Peru)
    MAM: [3, 4, 5]   # Outono
    JJA: [6, 7, 8]   # Inverno
    SON: [9, 10, 11] # Primavera

plotting:
  figsize: [12, 8]
  dpi: 300
  style: "seaborn-v0_8"
  colors:
    era5: "#1f77b4"
    station: "#ff7f0e"
    comparison: "#2ca02c"
  save_formats: ["png", "pdf"]

validation:
  min_data_points: 100
  max_missing_percentage: 30
  outlier_method: "iqr"  # iqr, zscore, isolation_forest
  outlier_threshold: 3

reporting:
  format: "markdown"
  include_plots: true
  detailed_statistics: true
  recommendations: true

integration:
  downscaling_tool_path: "../Projeto_modular_downscaling-tool"
  auto_proceed: false
  copy_files: true
```

## 📦 Dependências (`requirements.txt`)
```
# Core dependencies
numpy>=1.21.0
pandas>=1.5.0
scipy>=1.9.0
scikit-learn>=1.1.0

# Data processing
xarray>=2022.6.0
netCDF4>=1.6.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0

# Web framework
Flask>=2.2.0
Werkzeug>=2.2.0

# Report generation
markdown>=3.4.0
jinja2>=3.1.0

# Configuration
PyYAML>=6.0
python-dotenv>=0.20.0

# Utilities
tqdm>=4.64.0
click>=8.1.0

# Development (optional)
pytest>=7.1.0
pytest-cov>=3.0.0
black>=22.6.0
flake8>=5.0.0
```

## 🚀 Fluxo de Execução

### 1. **Preparação**
```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar estrutura de diretórios
python setup.py init
```

### 2. **Execução via Web**
```bash
# Iniciar servidor web
python src/web/app.py

# Acessar interface
# http://localhost:5001
```

### 3. **Execução Programática**
```python
from src.core.data_loader import DataLoader
from src.core.data_comparator import DataComparator
from src.reporting.report_generator import ReportGenerator

# Carregar dados
loader = DataLoader()
era5_data = loader.load_era5_data("path/to/era5.csv")
station_data = loader.load_station_data("path/to/station.csv")

# Sincronizar datasets
era5_sync, station_sync = loader.synchronize_datasets(era5_data, station_data)

# Executar comparação
comparator = DataComparator()
results = comparator.compare_variables(era5_sync, station_sync)

# Gerar relatório
reporter = ReportGenerator()
report = reporter.generate_full_report(results, "output/report.md")
```

### 4. **Integração com Downscaling**
```python
# Após análise de comparação, preparar dados para downscaling
from src.utils.helpers import prepare_for_downscaling

# Copiar arquivos processados para pasta do projeto de downscaling
downscaling_files = prepare_for_downscaling(
    era5_path="uploads/era5_data.csv",
    station_path="uploads/station_data.csv",
    results=comparison_results,
    output_dir="../Projeto_modular_downscaling-tool/uploads"
)

# Executar downscaling automaticamente (opcional)
if auto_proceed:
    import sys
    sys.path.append("../Projeto_modular_downscaling-tool")
    from src.utils.utils import run_downscaling_pipeline_advanced
    
    downscaling_results = run_downscaling_pipeline_advanced(
        downscaling_files,
        parameters,
        selected_models=['RandomForest', 'XGBoost'],
        variables=['temperature', 'precipitation']
    )
```

## 🎯 Métricas e Indicadores

### Principais Métricas Calculadas:

1. **Correlação**
   - Pearson (linear)
   - Spearman (monotônica)
   - Kendall (tau)

2. **Bias e Erro**
   - Bias médio
   - RMSE (Root Mean Square Error)
   - MAE (Mean Absolute Error)
   - MAPE (Mean Absolute Percentage Error)

3. **Distribuição**
   - Teste de normalidade
   - Skewness e Kurtosis
   - Percentis (5, 25, 50, 75, 95)

4. **Variabilidade**
   - Coeficiente de variação
   - Desvio padrão
   - Amplitude interquartil

5. **Eventos Extremos**
   - Percentis extremos (95%, 99%)
   - Frequência de excedência
   - Máximos e mínimos absolutos

## 🔄 Integração com Sistema Existente

### Preparação dos Dados:
1. **Análise de Qualidade**: Identificação de problemas nos dados
2. **Recomendações**: Sugestões de pré-processamento
3. **Sincronização**: Garantir períodos comuns entre datasets
4. **Formato Padronizado**: Preparar dados no formato esperado pelo downscaling

### Outputs para Downscaling:
- Dados limpos e sincronizados
- Relatório de qualidade dos dados
- Recomendações de período de treinamento
- Identificação de variáveis mais confiáveis

### Interface de Continuidade:
```python
# Botão na interface web para prosseguir automaticamente
@app.route('/proceed_to_downscaling', methods=['POST'])
def proceed_to_downscaling():
    """Transfere dados e parâmetros para o sistema de downscaling"""
    
    # Copiar arquivos processados
    # Preparar configuração
    # Redirecionar ou chamar sistema de downscaling
    # Retornar status de integração
```

## 📈 Exemplo de Saída do Relatório

### Resumo Executivo:
```
🌍 RELATÓRIO DE COMPARAÇÃO PRÉ-DOWNSCALING

📊 PERÍODO ANALISADO: 2010-01-01 a 2020-12-31 (11 anos)
📍 ESTAÇÃO: Llanganuco (LAT: -9.41, LON: -77.35)
🌡️ VARIÁVEIS: Temperatura e Precipitação

✅ PRINCIPAIS ACHADOS:
• Correlação temperatura ERA5 vs. estação: 0.85 (forte)
• Bias médio temperatura: +1.2°C (ERA5 superestima)
• Correlação precipitação: 0.67 (moderada)
• Bias precipitação: -15% (ERA5 subestima)

🎯 RECOMENDAÇÕES:
• Aplicar correção de bias para temperatura
• Considerar transformação logarítmica para precipitação
• Período de treino recomendado: 2010-2017
• Período de teste: 2018-2020
```

Este projeto servirá como uma etapa fundamental de preparação e validação antes do processo de downscaling, fornecendo insights valiosos sobre a qualidade e características dos dados que serão utilizados no modelo de Machine Learning.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Analisar estrutura do projeto modular de downscaling", "status": "completed"}, {"id": "2", "content": "Examinar c\u00f3digo da aplica\u00e7\u00e3o web existente", "status": "completed"}, {"id": "3", "content": "Entender processamento de dados do ERA5", "status": "completed"}, {"id": "4", "content": "Criar CLAUDE.md para projeto de compara\u00e7\u00e3o", "status": "completed"}]