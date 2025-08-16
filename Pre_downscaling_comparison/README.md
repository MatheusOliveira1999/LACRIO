# Sistema de Comparação Pré-Downscaling

Sistema completo para análise comparativa de dados climáticos ERA5 vs. estações meteorológicas, desenvolvido para executar **antes** do processo de downscaling estatístico.

## 🎯 Objetivo

Analisar e comparar dados climáticos de reanálise (ERA5) com observações de estações meteorológicas para:
- Identificar qualidade e consistência dos dados
- Gerar gráficos comparativos do período completo
- Produzir relatórios detalhados com recomendações
- Preparar dados para o processo de downscaling

## ✨ Características Principais

### 📊 **Análises Implementadas**
- **Correlação**: Pearson, Spearman, Kendall
- **Bias**: Sistemático, aleatório, sazonal
- **Distribuições**: Testes de normalidade, Q-Q plots
- **Eventos Extremos**: Percentis, detecção de extremos
- **Análise Temporal**: Tendências, variabilidade interanual
- **Análise Sazonal**: Comparações por estações do ano

### 📈 **Visualizações**
- Séries temporais comparativas do período completo
- Gráficos de dispersão ERA5 vs. Estação
- Análise de bias temporal e sazonal
- Comparações de distribuições
- Box plots mensais e sazonais
- Dashboard resumo com métricas principais

### 📋 **Relatórios**
- Relatório completo em Markdown e HTML
- Resumo executivo
- Recomendações para downscaling
- Export em JSON para análises posteriores

### 🌐 **Interface Web**
- Upload intuitivo de arquivos
- Configuração de parâmetros via interface
- Visualização de resultados em tempo real
- Download completo de resultados
- Integração automática com sistema de downscaling

## 🚀 Instalação e Uso

### 1. **Instalação**

```bash
# Clonar o repositório (ou copiar a pasta)
cd Pre_downscaling_comparison

# Instalar dependências
pip install -r requirements.txt
```

### 2. **Execução via Interface Web**

```bash
# Iniciar servidor
python src/web/app.py

# Acessar interface
# http://localhost:5001
```

### 3. **Execução Programática**

```python
from src.core.data_loader import DataLoader
from src.core.data_comparator import DataComparator
from src.visualization.comparative_plots import ComparativePlots
from src.reporting.report_generator import ReportGenerator

# Carregar e sincronizar dados
loader = DataLoader()
era5_df = loader.load_era5_data("era5_data.csv")
station_df = loader.load_station_data("station_data.csv")
era5_sync, station_sync = loader.synchronize_datasets(era5_df, station_df)

# Executar comparação
comparator = DataComparator()
results = comparator.compare_variables(era5_sync, station_sync)

# Gerar visualizações
plotter = ComparativePlots()
for var in ['temperature', 'precipitation']:
    plotter.time_series_comparison(
        era5_sync[var], station_sync[var], era5_sync['date'], var
    )

# Gerar relatório
reporter = ReportGenerator()
report_path = reporter.generate_full_report(results)
```

## 📁 Estrutura do Projeto

```
Pre_downscaling_comparison/
├── src/
│   ├── core/                     # Módulos centrais
│   │   ├── data_loader.py        # Carregamento de dados
│   │   └── data_comparator.py    # Comparação de dados
│   ├── analysis/                 # Análises estatísticas
│   │   ├── statistical_analysis.py
│   │   └── temporal_analysis.py
│   ├── visualization/            # Visualizações
│   │   └── comparative_plots.py
│   ├── reporting/                # Geração de relatórios
│   │   └── report_generator.py
│   ├── web/                      # Interface web
│   │   ├── app.py
│   │   └── templates/
│   └── utils/                    # Utilitários
├── static/                       # Arquivos estáticos
│   └── img/                      # Gráficos gerados
├── results/                      # Resultados
│   ├── reports/                  # Relatórios
│   ├── plots/                    # Gráficos salvos
│   └── summaries/                # Resumos
├── uploads/                      # Arquivos enviados
├── config.yaml                   # Configuração principal
├── requirements.txt              # Dependências
└── README.md                     # Documentação
```

## ⚙️ Configuração

O arquivo `config.yaml` permite personalizar:

```yaml
# Variáveis a analisar
analysis:
  variables:
    temperature:
      units: "°C"
      min_valid: -50
      max_valid: 60

# Parâmetros estatísticos
statistics:
  correlation_methods: ["pearson", "spearman"]
  extreme_percentiles: [95, 99, 99.9]

# Configurações de visualização
plotting:
  figsize: [12, 8]
  dpi: 300
  colors:
    era5: "#1f77b4"
    station: "#ff7f0e"
```

## 📊 Tipos de Análise

### 1. **Análise de Correlação**
- Correlações de Pearson, Spearman e Kendall
- Intervalos de confiança
- Correlações por quantis e estações

### 2. **Análise de Bias**
- Bias sistemático e aleatório
- Bias por magnitude da observação
- Variação sazonal do bias

### 3. **Análise de Distribuições**
- Testes de normalidade
- Comparação de distribuições (KS test)
- Q-Q plots e funções cumulativas

### 4. **Análise Temporal**
- Tendências de longo prazo
- Variabilidade interanual
- Análise de dados faltantes

### 5. **Eventos Extremos**
- Detecção de percentis extremos
- Taxa de detecção de eventos
- Comparação de valores máximos/mínimos

## 📋 Formato dos Dados

### Arquivos de Entrada

**ERA5 (CSV):**
```csv
date,t2m,tp,sp,d2m,u10,v10
2010-01-01,15.2,0.0,1013.2,12.1,2.3,-1.1
2010-01-02,16.1,2.4,1012.8,13.0,1.8,-0.8
...
```

**Estação (CSV):**
```csv
date,temperature,precipitation,humidity,pressure
2010-01-01,15.8,0.0,78.2,1012.5
2010-01-02,16.9,1.8,82.1,1011.9
...
```

### Saídas Geradas

- **Gráficos**: PNG de alta resolução (300 DPI)
- **Relatórios**: Markdown, HTML e JSON
- **Dados**: CSV sincronizados para downscaling
- **Configuração**: JSON para sistema de downscaling

## 🔧 Integração com Downscaling

O sistema se integra automaticamente com o projeto de downscaling:

1. **Análise pré-downscaling** → Validação de dados
2. **Relatório de qualidade** → Recomendações
3. **Preparação de dados** → Cópia para projeto downscaling
4. **Configuração automática** → Parâmetros otimizados
5. **Execução downscaling** → Processo ML

### Botão "Prosseguir para Downscaling"

```python
# Copia arquivos validados
# Gera configuração otimizada
# Prepara parâmetros recomendados
{
    "files": {...},
    "params": {...},
    "comparison_summary": {...}
}
```

## 📈 Métricas de Qualidade

### Correlação
- **> 0.8**: Excelente - Proceder diretamente
- **0.6 - 0.8**: Boa - Pré-processamento mínimo
- **0.4 - 0.6**: Moderada - Pré-processamento necessário
- **< 0.4**: Pobre - Revisar dados ou métodos

### Bias
- **Próximo de 0**: Sem correção necessária
- **Bias sistemático**: Aplicar correção de bias
- **Bias sazonal**: Correção por estação

### Recomendações Automáticas
- Período de treinamento otimizado
- Variáveis prioritárias
- Métodos de pré-processamento
- Configurações de modelo

## 🌍 Interface Web

### Funcionalidades
- **Upload**: Drag-and-drop ou seleção de arquivos
- **Configuração**: Parâmetros via formulário
- **Processamento**: Barra de progresso em tempo real
- **Resultados**: Dashboard interativo
- **Download**: Pacote completo ZIP
- **Integração**: Botão para downscaling

### Páginas
- `/` - Upload e configuração
- `/results` - Visualização de resultados
- `/download_results` - Download completo
- `/proceed_downscaling` - Preparação para ML

## 🧪 Exemplo de Uso Completo

```python
# 1. Configuração inicial
from src.web.app import main

# Executar interface web
main()

# 2. Upload via interface:
# - era5_data.csv
# - station_data.csv
# - Parâmetros: lat=-9.41, lon=-77.35

# 3. Processamento automático:
# - Carregamento e sincronização
# - Análise comparativa completa
# - Geração de visualizações
# - Criação de relatórios

# 4. Resultados:
# - 15+ gráficos comparativos
# - Relatório HTML detalhado
# - Recomendações específicas
# - Dados preparados para downscaling

# 5. Integração:
# - Clique em "Prosseguir para Downscaling"
# - Arquivos copiados automaticamente
# - Configuração salva
# - Pronto para ML
```

## 📊 Saídas do Sistema

### Gráficos Gerados
1. **time_series_comparison_[var].png** - Série temporal completa
2. **scatter_comparison_[var].png** - Dispersão ERA5 vs Estação
3. **bias_analysis_[var].png** - Análise temporal do bias
4. **seasonal_comparison_[var].png** - Comparação sazonal
5. **distribution_comparison_[var].png** - Comparação de distribuições
6. **summary_dashboard.png** - Dashboard resumo

### Relatórios
1. **relatorio_comparacao_[timestamp].md** - Relatório completo
2. **resumo_executivo_[timestamp].md** - Resumo executivo
3. **relatorio_comparacao_[timestamp].html** - Versão HTML
4. **resultados_comparacao_[timestamp].json** - Export JSON

### Arquivos de Integração
1. **pre_comparison_config.json** - Configuração para downscaling
2. **[arquivos_csv_sincronizados]** - Dados limpos
3. **summary.json** - Resumo para automação

## 🔍 Exemplo de Relatório

```markdown
# Relatório de Comparação Pré-Downscaling

## Resumo Executivo
• Período: 2010-01-01 a 2020-12-31 (11 anos)
• Estação: Llanganuco (-9.41, -77.35)
• Variáveis: Temperatura, Precipitação

## Principais Achados
• Correlação temperatura: 0.85 (forte)
• Bias médio temperatura: +1.2°C
• Correlação precipitação: 0.67 (moderada)
• Bias precipitação: -15%

## Recomendações
✅ Temperatura: Excelente para downscaling
🔄 Precipitação: Aplicar correção de bias
📅 Período treino: 2010-2017
📅 Período teste: 2018-2020
```

## 🚀 Próximos Passos

Após executar a comparação pré-downscaling:

1. **Revisar relatório** - Analisar qualidade dos dados
2. **Aplicar recomendações** - Pré-processar se necessário  
3. **Prosseguir para downscaling** - Usar botão de integração
4. **Executar ML** - Sistema de downscaling automático
5. **Validar resultados** - Comparar com dados independentes

---

**🌍 Sistema desenvolvido para análise climática no Peru - Cordilheira Branca**

**📧 Suporte**: Sistema integrado ao projeto de downscaling LACRIO

**🔄 Versão**: 1.0.0 - Compatível com projeto modular de downscaling