from flask import Flask, render_template, request, flash, redirect, url_for
from processar_dados_horario import iniciar_processamento

# Inicializa a aplicação Flask
app = Flask(__name__)
# Chave secreta necessária para usar 'flash messages'
app.secret_key = 'chave-secreta-dados-horarios-2024'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Coleta os dados do formulário enviado
        era5_dir = request.form.get('era5_dir')
        era5_land_dir = request.form.get('era5_land_dir')
        station_file = request.form.get('station_file')
        lat = request.form.get('lat')
        lon = request.form.get('lon')

        # Validação simples para garantir que os campos não estão vazios
        if not all([era5_dir, era5_land_dir, station_file, lat, lon]):
            flash("Todos os campos são obrigatórios!", "danger")
            return redirect(url_for('index'))

        try:
            # Converte lat/lon para float
            lat_float = float(lat)
            lon_float = float(lon)

            # Informa o usuário que o processo começou
            flash("Processamento de dados horários iniciado. Correção UTC aplicada aos dados da estação (Peru UTC-5 → UTC). Filtrando apenas temperatura e precipitação. Mantendo horários 00:00, 06:00, 12:00 e 18:00. Isso pode levar vários minutos...", "info")
            
            # Chama a função de processamento
            # ATENÇÃO: Em uma aplicação real, processos longos devem ser
            # executados em background para não travar o servidor.
            # Para este caso de uso, executaremos diretamente.
            mensagem, categoria = iniciar_processamento(era5_dir, era5_land_dir, station_file, lat_float, lon_float)
            
            # Informa o usuário sobre o resultado
            flash(mensagem, categoria)

        except ValueError:
            flash("Latitude e Longitude devem ser números válidos.", "danger")
        except Exception as e:
            flash(f"Ocorreu um erro inesperado: {e}", "danger")

        return redirect(url_for('index'))

    # Se o método for GET, apenas renderiza a página
    return render_template('index_horarios.html')

if __name__ == '__main__':
    # Executa a aplicação em modo de debug
    # Acesse em http://127.0.0.1:5001 no seu navegador
    app.run(debug=False, port=5001)