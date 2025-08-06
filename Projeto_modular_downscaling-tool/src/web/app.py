"""
app.py
Aplicação principal Flask para interface web
"""

import os
import io
import sys
import zipfile
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Adicionar o diretório raiz do projeto ao sys.path para imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Importar módulos locais
from src.utils.utils import run_weather_downscaling, run_downscaling_pipeline_advanced  
from src.web.templates_generator import create_html_template

# Configuração do Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Definir caminhos absolutos para as pastas na raiz do projeto
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app.config['UPLOAD_FOLDER'] = os.path.join(project_root, 'uploads')
app.config['MODELS_FOLDER'] = os.path.join(project_root, 'models')
app.config['RESULTS_FOLDER'] = os.path.join(project_root, 'results')
app.config['STATIC_IMG_FOLDER'] = os.path.join(project_root, 'static', 'img')

# Criar pastas necessárias
for folder in [app.config['UPLOAD_FOLDER'], app.config['MODELS_FOLDER'], 
               app.config['RESULTS_FOLDER'], app.config['STATIC_IMG_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Sessão atual
current_session = {}

@app.route('/')
def index():
    """Página inicial"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Upload de arquivos"""
    try:
        files_info = {}

        # Processar arquivos
        for file_type in ['era5', 'station', 'dem']:
            if file_type in request.files:
                file = request.files[file_type]
                if file.filename != '':
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    files_info[file_type] = filepath

        # Parâmetros
        latitude = float(request.form.get('latitude', -9.41))
        longitude = float(request.form.get('longitude', -77.35))
        split_date = request.form.get('split_date', '2018-01-01')

        # Validar arquivos necessários
        required_files = ['era5', 'station', 'dem']
        missing_files = [f for f in required_files if f not in files_info]
        if missing_files:
            return jsonify({
                'status': 'error', 
                'message': f'Arquivos faltantes: {missing_files}'
            }), 400

        # Informações dos dados
        data_info = {
            'status': 'uploaded',
            'files': list(files_info.keys()),
            'latitude': latitude,
            'longitude': longitude,
            'split_date': split_date
        }

        # Armazenar na sessão
        current_session['files'] = files_info
        current_session['params'] = {
            'latitude': latitude,
            'longitude': longitude,
            'split_date': split_date
        }
        current_session['data_info'] = data_info

        return jsonify({'status': 'success', 'data_info': data_info})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/process', methods=['POST'])
def process_models():
    """Processar modelos com configurações avançadas"""
    try:
        if 'files' not in current_session:
            return jsonify({'status': 'error', 'message': 'Nenhum arquivo foi carregado'}), 400

        # Obter configurações do request
        data = request.json or {}
        selected_models = data.get('models', ['RandomForest', 'XGBoost', 'MLP'])
        variables = data.get('variables', ['temperature', 'precipitation'])
        use_ensemble = data.get('use_ensemble', True)
        optimize_params = data.get('optimize_params', True)
        
        # Configurações de otimização
        opt_config = data.get('optimization_config', {})
        
        # Executar processamento com as novas configurações
        results = run_downscaling_pipeline_advanced(
            current_session['files'],
            current_session['params'],
            selected_models,
            variables,
            use_ensemble,
            optimize_params,
            opt_config
        )

        current_session['results'] = results

        return jsonify({
            'status': 'success', 
            'results': results
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/download_results')
def download_results():
    """Download dos resultados"""
    try:
        if 'results' not in current_session:
            return jsonify({'status': 'error', 'message': 'Nenhum resultado disponível'}), 400

        # Criar arquivo ZIP com resultados
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Adicionar relatórios da pasta 'results'
            results_dir = app.config['RESULTS_FOLDER']
            for variable in current_session['results']:
                report_name = f'relatorio_downscaling_{variable}.txt'
                report_path = os.path.join(results_dir, report_name)
                if os.path.exists(report_path):
                    zip_file.write(report_path, report_name)
            
            # Adicionar gráficos da pasta 'static/img'
            img_dir = app.config['STATIC_IMG_FOLDER']
            for variable in current_session['results']:
                for graph_type in ['resultados_downscaling', 'feature_importance', 'analise_temporal']:
                    # Procurar por todos os arquivos que começam com o padrão
                    if os.path.exists(img_dir):
                        for filename in os.listdir(img_dir):
                            if filename.startswith(f'{graph_type}_{variable}'):
                                graph_path = os.path.join(img_dir, filename)
                                if os.path.exists(graph_path):
                                    zip_file.write(graph_path, os.path.join('img', filename))
        
        zip_buffer.seek(0)
        
        return send_file(
            io.BytesIO(zip_buffer.read()),
            as_attachment=True,
            download_name='resultados_downscaling.zip',
            mimetype='application/zip'
        )

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# Função principal para modo standalone
def main():
    """Função principal para executar o sistema"""
    
    if len(sys.argv) > 1 and sys.argv[1] == 'web':
        # Modo web
        create_html_template()
        print("🌍 Iniciando servidor web...")
        print("Acesse: http://localhost:5000")
        app.run(debug=True, port=5000)
        
    else:
        # Modo standalone - exemplo
        print("🚀 Executando exemplo de downscaling climático...")
        
        # Caminhos dos arquivos (ajustar conforme necessário)
        era5_path = "dados_era5_era5land.csv"
        station_path = "dados_estacao_cuchillacocha.csv"
        dem_path = "Quilcayhuanca_static.nc"
        
        # Verificar se arquivos existem
        files_exist = all(os.path.exists(f) for f in [era5_path, station_path, dem_path])
        
        if not files_exist:
            print("❌ Arquivos de exemplo não encontrados.")
            print("💡 Para usar o modo web: python app.py web")
            print("💡 Para standalone: ajuste os caminhos dos arquivos no código")
            
        else:
            # Executar downscaling
            results = run_weather_downscaling(
                era5_path=era5_path,
                station_path=station_path,
                dem_path=dem_path,
                variables=['temperature', 'precipitation'],
                lat=-9.41, lon=-77.35,
                split_date='2018-01-01',
                optimize_models=True,
                use_ensemble=True,
                save_results=True,
                generate_plots=True
            )
            
            print("\n🎉 Downscaling concluído!")
            print("📁 Verifique os arquivos gerados na pasta atual")

if __name__ == "__main__":
    main()