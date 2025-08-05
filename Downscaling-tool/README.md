# Ferramenta de Previsão de Temperatura com ERA5

## Instalação

1. Clone o repositório
2. Crie um ambiente virtual com python 3.11:
   ```bash
   conda create -n temp-predict-py311 python=3.11
3. Ative o ambiente virtual 
4. Instale as dependências 
    ```bash
    pip install -r requirements.txt

## Execução

1. Rode o codigo
    ```bash
    python sistema_downscaling.py web

2. Abra o navegador em: http://localhost:5000

## Uso 

1. Carregue os arquivos necessários (ERA5, Estação, DEM)
2. Configure os parâmetros
3. Execute o processamento
4. Visualize e exporte os resultados