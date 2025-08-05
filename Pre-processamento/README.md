# Sistema de Processamento de Dados Meteorológicos ERA5/ERA5-Land

## 📋 Descrição

Este sistema processa e combina dados meteorológicos de três fontes:
- **ERA5**: Dados de reanálise atmosférica com múltiplos níveis de pressão
- **ERA5-Land**: Dados de reanálise de superfície com alta resolução
- **Estação Meteorológica**: Dados observacionais locais

O sistema oferece uma interface web simples para configurar e executar o processamento, gerando arquivos CSV agregados diariamente prontos para análise.

## 🚀 Funcionalidades

- Interface web intuitiva para configuração dos parâmetros
- Processamento automatizado de arquivos NetCDF (ERA5/ERA5-Land)
- Leitura inteligente de arquivos CSV com detecção automática de encoding
- Extração de dados para coordenadas específicas (latitude/longitude)
- Agregação diária de variáveis meteorológicas
- Conversão automática de unidades (temperatura K→°C, precipitação m→mm)
- Combinação de dados de múltiplas fontes em arquivos CSV unificados


## 🔧 Pré-requisitos

### Dependências Python

```bash
pip install flask
pip install xarray
pip install pandas
pip install netCDF4
pip install dask
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
pip install flask xarray pandas netCDF4 dask
```

### 3. Execute a aplicação
```bash
python app.py
```

### 4. Acesse a interface web
Abra seu navegador e acesse: `http://127.0.0.1:5000`

### 5. Configure os parâmetros:
- **Diretório ERA5**: Caminho para a pasta com arquivos .nc do ERA5
- **Diretório ERA5-Land**: Caminho para a pasta com arquivos .nc do ERA5-Land
- **Arquivo da Estação**: Caminho completo para o arquivo CSV da estação
- **Latitude**: Coordenada latitude do ponto de interesse
- **Longitude**: Coordenada longitude do ponto de interesse

### 6. Clique em "Processar Dados"

## 📊 Saída do Processamento

O sistema gera dois arquivos CSV na pasta `dados_csv/`:

### 1. Dados da Estação Processados
- Nome: `[nome_original]_processado.csv`
- Colunas: `data`, `Precipitação`, `Temperatura`
- Agregação: Valores diários (média para temperatura, soma para precipitação)

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

```python
# Uso via linha de comando (sem interface web)
python processar_dados.py \
    --era5_dir "/dados/era5" \
    --era5_land_dir "/dados/era5_land" \
    --station_file "/dados/estacao_RS.csv" \
    --lat -29.7 \
    --lon -53.7
```

## 📄 Licença

Este projeto é fornecido como está, para fins educacionais e de pesquisa.


**Nota**: Para questões específicas sobre os dados ERA5/ERA5-Land, consulte a documentação oficial do ECMWF (European Centre for Medium-Range Weather Forecasts).