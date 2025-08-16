# Sistema de Processamento de Dados Meteorológicos ERA5/ERA5-Land

## 📋 Descrição

Este sistema processa e combina dados meteorológicos de três fontes:
- **ERA5**: Dados de reanálise atmosférica com múltiplos níveis de pressão
- **ERA5-Land**: Dados de reanálise de superfície com alta resolução
- **Estação Meteorológica**: Dados observacionais locais

O sistema converte automaticamente dados de estações meteorológicas do horário local do Peru (UTC-5) para UTC e gera arquivos CSV agregados diariamente prontos para análise.

## 🚀 Funcionalidades

- Interface de linha de comando (CLI) para processamento automatizado
- Conversão automática de horário local Peru (UTC-5) para UTC
- Processamento automatizado de arquivos NetCDF (ERA5/ERA5-Land)
- Leitura inteligente de arquivos CSV com detecção automática de encoding
- Extração de dados para coordenadas específicas (latitude/longitude)
- Agregação diária de variáveis meteorológicas
- Conversão automática de unidades (temperatura K→°C, precipitação m→mm)
- Combinação de dados de múltiplas fontes em arquivos CSV unificados


## 🔧 Pré-requisitos

### Dependências Python

```bash
pip install xarray
pip install pandas
pip install netCDF4
pip install dask
pip install pytz
```

### Dados Necessários

1. **Arquivos ERA5** (.nc): Dados atmosféricos com níveis de pressão
2. **Arquivos ERA5-Land** (.nc): Dados de superfície
3. **Arquivo da Estação** (.csv): Deve conter as colunas:
   - `Datetime`: Data/hora das observações
   - `Temperature (°C)`: Temperatura
   - `Precipitation (mm)`: Precipitação

## 💻 Instalação e Uso

### 1. Clone ou baixe os arquivos do projeto

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

Ou instale individualmente:
```bash
pip install xarray pandas netCDF4 dask pytz
```

### 3. Execute o processamento via linha de comando

#### Para dados diários:
```bash
python processar_dados.py \
    --era5_dir "/caminho/para/era5" \
    --era5_land_dir "/caminho/para/era5_land" \
    --station_file "/caminho/para/estacao.csv" \
    --lat -9.5 \
    --lon -77.5
```

#### Para dados horários:
```bash
python processar_dados_horario.py \
    --era5_dir "/caminho/para/era5" \
    --era5_land_dir "/caminho/para/era5_land" \
    --station_file "/caminho/para/estacao.csv" \
    --lat -9.5 \
    --lon -77.5
```

#### Parâmetros obrigatórios:
- **--era5_dir**: Caminho para a pasta com arquivos .nc do ERA5
- **--era5_land_dir**: Caminho para a pasta com arquivos .nc do ERA5-Land
- **--station_file**: Caminho completo para o arquivo CSV da estação
- **--lat**: Coordenada latitude do ponto de interesse
- **--lon**: Coordenada longitude do ponto de interesse

## 📊 Saída do Processamento

O sistema gera dois arquivos CSV:
- Para dados diários: na pasta `dados_csv/`
- Para dados horários: na pasta `dados_horarios_csv/`

### 1. Dados da Estação Processados
- Nome: `[nome_original]_processado.csv`
- Colunas: `data`, `Precipitação`, `Temperatura`
- Agregação: Valores diários (média para temperatura, soma para precipitação)
- **Conversão de fuso horário**: Dados convertidos do horário local Peru (UTC-5) para UTC

### 2. Dados ERA5/ERA5-Land Combinados
- Nome: `dados_era5_[nome_estação]_processado.csv`
- Colunas: Múltiplas variáveis meteorológicas incluindo:
  - Variáveis de superfície do ERA5-Land (t2m, tp, u10, v10, etc.)
  - Variáveis atmosféricas do ERA5 em diferentes níveis de pressão
- Agregação: Valores diários médios (ou soma para variáveis acumuladas)

## 🔍 Detalhes Técnicos

### Variáveis Processadas

**ERA5-Land (superfície):**
- Variáveis instantâneas: u10, v10, d2m, t2m, sp, z (médias diárias)
- Variáveis acumuladas: tp, ssrd, strd, sf (somas diárias)

**ERA5 (níveis de pressão):**
- Todas as variáveis meteorológicas disponíveis nos níveis de pressão
- Exclui automaticamente metadados (number, expver, step, surface)

### Conversões de Unidades
- Temperatura: Kelvin → Celsius
- Precipitação: metros → milímetros

### Conversão de Fuso Horário
- Dados da estação: Horário local Peru (UTC-5) → UTC
- Usado biblioteca pytz com timezone 'America/Lima'
- Tratamento automático de horários ambíguos e inexistentes

### Tratamento de Erros
- Detecção automática de encoding para arquivos CSV
- Validação de colunas obrigatórias
- Tratamento de valores ausentes (NaN)
- Mensagens de erro detalhadas para debugging

## ⚠️ Observações Importantes

1. **Performance**: O processamento pode levar vários minutos dependendo do tamanho dos dados
2. **Memória**: Datasets grandes podem requerer bastante memória RAM
3. **Formato de Data**: O sistema espera datas no formato padrão (YYYY-MM-DD HH:MM:SS)
4. **Coordenadas**: O sistema seleciona automaticamente o ponto mais próximo das coordenadas fornecidas

## 🐛 Solução de Problemas

### Erro: "Arquivo não encontrado"
- Verifique se os caminhos estão corretos e completos
- Use barras normais (/) ou duplas barras invertidas (\\) no Windows

### Erro: "Colunas necessárias não encontradas"
- Verifique se o arquivo CSV da estação tem as colunas exatas: `Datetime`, `Temperature (°C)`, `Precipitation (mm)`

### Erro: "Nenhum arquivo .nc encontrado"
- Certifique-se de que os diretórios ERA5/ERA5-Land contêm arquivos NetCDF (.nc)

### Processo muito demorado
- Normal para grandes datasets
- Considere processar períodos menores de dados
- Monitore o uso de memória RAM

## 📝 Exemplo de Uso

### Processamento de dados diários com conversão UTC:
```bash
python processar_dados.py \
    --era5_dir "/dados/era5" \
    --era5_land_dir "/dados/era5_land" \
    --station_file "/dados/estacao_Peru.csv" \
    --lat -9.5 \
    --lon -77.5
```

### Processamento de dados horários (mantendo horários específicos):
```bash
python processar_dados_horario.py \
    --era5_dir "/dados/era5" \
    --era5_land_dir "/dados/era5_land" \
    --station_file "/dados/estacao_Peru.csv" \
    --lat -9.5 \
    --lon -77.5
```

## 📄 Licença

Este projeto é fornecido como está, para fins educacionais e de pesquisa.


**Nota**: Para questões específicas sobre os dados ERA5/ERA5-Land, consulte a documentação oficial do ECMWF (European Centre for Medium-Range Weather Forecasts).