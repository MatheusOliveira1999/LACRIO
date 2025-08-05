# 🌦️ Sistema de Downscaling Climático com IA

Um sistema completo para melhoramento de dados climáticos utilizando técnicas de downscaling por meio de Inteligência Artificial, especialmente desenvolvido para regiões montanhosas.

## 📋 Características

- **Múltiplas Variáveis**: Temperatura e Precipitação
- **Dados de Entrada**: ERA5/ERA5-Land, estações meteorológicas in-situ, modelos de elevação digital
- **Algoritmos ML**: 9 modelos diferentes (Linear, Árvores, Ensemble, Redes Neurais, XGBoost)
- **Otimização Automática**: Hiperparâmetros otimizados automaticamente
- **Modelos Ensemble**: Combinação dos melhores modelos
- **Interface Web**: Interface amigável para uso
- **Visualizações**: Gráficos e análises detalhadas
- **Features Especializadas**: Criadas especificamente para regiões montanhosas

## 🏗️ Estrutura do Projeto

```
downscaling_project/
├── main.py                     # Execução via linha de comando
├── run_web.py                  # Servidor web
├── requirements.txt            # Dependências
├── config/
│   └── settings.py            # Configurações globais
├── src/
│   ├── models/                # Modelos principais
│   ├── data/                  # Carregamento e processamento
│   ├── ml/                    # Algoritmos e otimização
│   ├── visualization/         # Gráficos e visualizações
│   ├── utils/                 # Utilitários
│   └── web/                   # Interface web
├── static/img/                # Gráficos gerados
├── uploads/                   # Arquivos de upload
├── models/                    # Modelos salvos
└── results/                   # Relatórios
```

## 🚀 Instalação

### 1. Clonar o repositório
```bash
git clone <repository-url>
cd downscaling_project
```

### 2. Criar ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Instalar dependências
```bash
pip install -r requirements.txt
```

### 4. (Opcional) Instalar XGBoost
```bash
pip install xgboost
```

## 💻 Uso

### Interface Web (Recomendado)

```bash
python run_web.py
```

Acesse: http://localhost:5000

### Linha de Comando

#### Uso básico:
```bash
python main.py --era5 dados_era5.csv --station dados_estacao.csv --dem modelo_elevacao.nc
```

#### Apenas temperatura:
```bash
python main.py --era5 dados_era5.csv --station dados_estacao.csv --dem modelo_elevacao.nc \
               --variables temperature
```

#### Modelos específicos:
```bash
python main.py --era5 dados_era5.csv --station dados_estacao.csv --dem modelo_elevacao.nc \
               --models RandomForest XGBoost --optimize
```

#### Análise completa:
```bash
python main.py --era5 dados_era5.csv --station dados_estacao.csv --dem modelo_elevacao.nc \
               --optimize --ensemble --save-plots --save-models
```

#### Modo rápido (para testes):
```bash
python main.py --era5 dados_era5.csv --station dados_estacao.csv --dem modelo_elevacao.nc \
               --fast
```

### Opções de Linha de Comando

#### Obrigatórios:
- `--era5`: Arquivo CSV com dados ERA5/ERA5-Land
- `--station`: Arquivo CSV com dados da estação
- `--dem`: Arquivo NetCDF com modelo de elevação

#### Localização:
- `--lat`: Latitude da estação (padrão: -9.41)
- `--lon`: Longitude da estação (padrão: -77.35)

#### Processamento:
- `--variables`: Variáveis a processar (`temperature`, `precipitation`)
- `--models`: Modelos específicos para treinar
- `--split-date`: Data de divisão treino/teste (padrão: 2018-01-01)

#### Treinamento:
- `--optimize`: Otimizar hiperparâmetros
- `--no-optimize`: Não otimizar (mais rápido)
- `--ensemble`: Criar modelo ensemble
- `--no-ensemble`: Não criar ensemble

#### Saídas:
- `--save-models`: Salvar modelos treinados
- `--save-plots`: Gerar e salvar gráficos
- `--no-plots`: Não gerar gráficos

#### Outros:
- `--fast`: Modo rápido (sem otimização, sem gráficos)
- `--verbose`: Saída detalhada
- `--config`: Arquivo JSON com configurações

## 📊 Formato dos Dados

### Dados ERA5/ERA5-Land (CSV)
```csv
date,t2m,tp,sp,u10,v10,d2m,r,ssrd,strd,z
2020-01-01,15.2,0.0,101325,2.1,-1.5,8.3,65.2,15000000,25000000,500000
...
```

### Dados da Estação (CSV)
```csv
data,Temperatura,Precipitação
2020-01-01,14.8,0.0
...
```

### Modelo de Elevação (NetCDF)
Arquivo NetCDF com variável de elevação (`HGT`, `elevation`, `dem`, `z`, ou `height`)

## 🤖 Modelos Disponíveis

1. **LinearRegression** - Regressão linear simples
2. **Ridge** - Regressão Ridge (regularização L2)
3. **Lasso** - Regressão Lasso (regularização L1)
4. **DecisionTree** - Árvore de decisão
5. **RandomForest** - Floresta aleatória
6. **ExtraTrees** - Árvores extremamente aleatórias
7. **GradientBoosting** - Gradient boosting
8. **XGBoost** - XGBoost (se instalado)
9. **SVR** - Support Vector Regression
10. **MLP** - Rede neural multicamadas

## 📈 Features Criadas

### Temporais
- Componentes sazonais e cíclicas
- Tendências temporais
- Informações de data/hora

### Lag
- Valores defasados (1, 2, 3, 7, 14 dias)
- Diferentes lags para temperatura e precipitação

### Estatísticas Móveis
- Médias móveis (3, 7, 14, 30 dias)
- Desvios padrão móveis
- Mínimos e máximos móveis
- Percentis móveis

### Interações
- Velocidade e direção do vento
- Interações entre variáveis meteorológicas
- Índices derivados

### Topográficas (para regiões montanhosas)
- Correção de temperatura por altitude
- Efeito orográfico na precipitação
- Features específicas para montanhas

## 📊 Métricas de Avaliação

### Básicas
- **RMSE**: Raiz do erro quadrático médio
- **MAE**: Erro absoluto médio
- **R²**: Coeficiente de determinação
- **Correlação**: Correlação de Pearson
- **Bias**: Viés médio
- **Skill Score**: Comparação com climatologia

### Específicas para Precipitação
- **Acurácia de Detecção**: % de acertos na detecção de chuva
- **POD**: Probabilidade de detecção
- **FAR**: Taxa de falso alarme
- **R² para dias chuvosos**: R² apenas para dias com precipitação

## 📁 Saídas do Sistema

### Modelos Salvos
- Arquivos `.pkl` com modelos treinados
- Arquivo JSON com informações e métricas

### Gráficos
- Observado vs Predito
- Séries temporais
- Análise de resíduos
- Importância das features
- Análise temporal (mensal/sazonal)
- Comparação entre modelos

### Relatórios
- Relatório completo em texto
- Métricas detalhadas por modelo
- Análise de performance

## ⚙️ Configuração Avançada

Criar arquivo `config.json`:

```json
{
  "optimize_hyperparams": true,
  "feature_selection": true,
  "max_features": 80,
  "ensemble_size": 3,
  "fast_optimization": true,
  "cv_folds": 3,
  "random_state": 42
}
```

Usar com:
```bash
python main.py --config config.json [outros argumentos]
```

## 🐛 Solução de Problemas

### Erro de memória
- Use `--fast` para modo rápido
- Reduza `max_features` na configuração
- Use `--no-optimize` para economizar memória

### XGBoost não disponível
```bash
pip install xgboost
```

### Arquivos não encontrados
- Verifique os caminhos dos arquivos
- Certifique-se que os arquivos não estão vazios
- Verifique as permissões de leitura

### Dados insuficientes
- Mínimo de 365 registros necessários
- Certifique-se que há sobreposição temporal entre ERA5 e estação

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 📞 Suporte

Para dúvidas e suporte:
- Abra uma issue no GitHub
- Consulte a documentação completa
- Verifique os exemplos na pasta `examples/`

## 🙏 Agradecimentos

- ERA5/ERA5-Land data from Copernicus Climate Change Service (C3S)
- Scikit-learn community
- XGBoost developers
- Flask framework