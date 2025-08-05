"""
Rotas da aplicação web
"""

from flask import render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import io
import zipfile
import json
from typing import Dict, Any

# Imports locais
from ..models.downscaling_model import WeatherDownscalingModel
from ..visualization.plots import PlotGenerator
from ..utils.file_utils import validate_input_files
from config.settings import UPLOAD_FOLDER, RESULTS_FOLDER, IMG_FOLDER


# Sessão atual (em produção, usar Redis ou banco de dados)
current_session = {}


def register_routes(app):
    """
    Registra todas as rotas da aplicação
    
    Parameters:
    -----------
    app : Flask
        Aplicação Flask
    """
    
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
                        filepath = os.path.join(UPLOAD_FOLDER, filename)
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
            
            # Validar arquivos
            validation_result = validate_input_files(
                files_info.get('era5', ''),
                files_info.get('station', ''),
                files_info.get('dem', '')
            )
            
            if not all(validation_result.values()):
                invalid_files = [f for f, valid in validation_result.items() if not valid]
                return jsonify({
                    'status': 'error',
                    'message': f'Arquivos inválidos: {invalid_files}'
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
                return jsonify({
                    'status': 'error',
                    'message': 'Nenhum arquivo foi carregado'
                }), 400
            
            # Obter configurações do request
            data = request.json or {}
            selected_models = data.get('models', ['RandomForest', 'XGBoost', 'MLP'])
            variables = data.get('variables', ['temperature', 'precipitation'])
            use_ensemble = data.get('use_ensemble', True)
            optimize_params = data.get('optimize_params', True)
            
            # Configurações de otimização
            opt_config = data.get('optimization_config', {})
            
            # Executar processamento
            results = run_downscaling_pipeline_web(
                current_session['files'],
                current_session['params'],
                selected_models,
                variables,
                use_ensemble,
                optimize_params,
                opt_config
            )
            
            # Contar gráficos gerados
            total_plots = 0
            for var in results:
                if isinstance(results[var], dict) and 'generated_plots' in results[var]:
                    plots = results[var]['generated_plots']
                    if 'individual_models' in plots:
                        total_plots += len(plots['individual_models'])
                    if 'temporal_individual' in plots:
                        total_plots += len(plots['temporal_individual'])
                    total_plots += 3  # comparison + summary + feature_importance
            
            current_session['results'] = results
            
            return jsonify({
                'status': 'success',
                'results': results,
                'total_plots_generated': total_plots
            })
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
    
    @app.route('/download_results')
    def download_results():
        """Download dos resultados"""
        try:
            if 'results' not in current_session:
                return jsonify({
                    'status': 'error',
                    'message': 'Nenhum resultado disponível'
                }), 400
            
            # Criar arquivo ZIP com resultados
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Adicionar relatórios
                for variable in current_session['results']:
                    report_name = f'relatorio_downscaling_{variable}.txt'
                    report_path = os.path.join(RESULTS_FOLDER, report_name)
                    if os.path.exists(report_path):
                        zip_file.write(report_path, report_name)
                
                # Adicionar gráficos
                for variable in current_session['results']:
                    for graph_type in ['resultados', 'feature_importance', 'analise_temporal', 'comparacao_modelos']:
                        graph_name = f'{graph_type}_{variable}.png'
                        graph_path = os.path.join(IMG_FOLDER, graph_name)
                        if os.path.exists(graph_path):
                            zip_file.write(graph_path, os.path.join('graficos', graph_name))
            
            zip_buffer.seek(0)
            
            return send_file(
                io.BytesIO(zip_buffer.read()),
                as_attachment=True,
                download_name='resultados_downscaling.zip',
                mimetype='application/zip'
            )
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
    
    @app.route('/health')
    def health_check():
        """Health check da aplicação"""
        return jsonify({
            'status': 'healthy',
            'message': 'Sistema de downscaling operacional'
        })
    
    @app.route('/models')
    def list_models():
        """Lista modelos disponíveis"""
        from ..ml.model_factory import ModelFactory
        
        factory = ModelFactory()
        available_models = factory.get_available_models()
        
        return jsonify({
            'available_models': available_models,
            'temperature_recommended': factory.recommend_models_for_variable('temperature'),
            'precipitation_recommended': factory.recommend_models_for_variable('precipitation')
        })


def run_downscaling_pipeline_web(files: Dict[str, str], params: Dict[str, Any],
                                selected_models: list, variables: list,
                                use_ensemble: bool, optimize_params: bool,
                                opt_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pipeline principal para interface web
    
    Parameters:
    -----------
    files : dict
        Caminhos dos arquivos
    params : dict
        Parâmetros da análise
    selected_models : list
        Modelos selecionados
    variables : list
        Variáveis climáticas
    use_ensemble : bool
        Usar ensemble
    optimize_params : bool
        Otimizar hiperparâmetros
    opt_config : dict
        Configurações de otimização
        
    Returns:
    --------
    dict : Resultados da análise
    """
    
    lat = params['latitude']
    lon = params['longitude']
    split_date = params['split_date']
    
    era5_path = files['era5']
    station_path = files['station']
    dem_path = files['dem']
    
    results_all = {}
    
    for variable in variables:
        print(f"\n{'='*60}")
        print(f"PROCESSANDO {variable.upper()}")
        print(f"{'='*60}")
        
        try:
            # Configuração do modelo
            config = {
                'optimize_hyperparams': optimize_params,
                'feature_selection': opt_config.get('feature_selection', True),
                'max_features': 80,
                'ensemble_size': 3,
                'optimize_all_models': opt_config.get('optimize_all_models', True),
                'fast_optimization': opt_config.get('fast_optimization', True),
                'use_random_search': opt_config.get('use_random_search', False),
                'random_search_iter': opt_config.get('random_search_iter', 30),
                'cv_folds': opt_config.get('cv_folds', 3),
                'show_top_params': False,
                'verbose': False
            }
            
            # Criar modelo
            model = WeatherDownscalingModel(variable=variable, config=config)
            
            # Pipeline completo
            model.load_and_merge_data(era5_path, station_path, dem_path, lat, lon)
            model.create_features()
            model.prepare_data(split_date)
            model.train_models(selected_models=selected_models, optimize=optimize_params)
            
            # Criar ensemble se solicitado
            if use_ensemble and len(model.results) >= 2:
                model.create_ensemble()
            
            # Gerar visualizações
            try:
                plot_generator = PlotGenerator(model)
                generate_individual = opt_config.get('generate_individual_plots', True)
                
                if generate_individual:
                    plot_generator.plot_all_models_results(save_plots=True)
                    plot_generator.plot_temporal_analysis_all_models(save_plots=True)
                
                plot_generator.plot_models_comparison(save_plot=True)
                plot_generator.plot_results(save_plots=True)
                plot_generator.plot_feature_importance(save_plot=True)
                plot_generator.plot_temporal_analysis(save_plot=True)
                
            except Exception as e:
                print(f"⚠️ Erro ao gerar gráficos: {str(e)}")
            
            # Gerar relatório
            try:
                from ..utils.report_generator import generate_report
                generate_report(model, variable)
            except Exception as e:
                print(f"⚠️ Erro ao gerar relatório: {str(e)}")
            
            # Salvar modelos
            model.save_models()
            
            # Coletar resultados
            plots_info = {
                'comparison': f'comparacao_modelos_{variable}.png',
                'summary': f'resultados_downscaling_{variable}.png'
            }
            
            generate_individual = opt_config.get('generate_individual_plots', True)
            if generate_individual:
                plots_info.update({
                    'individual_models': [f'resultados_{variable}_{m.lower().replace(" ", "_")}.png' 
                                        for m in model.results.keys()],
                    'temporal_individual': [f'analise_temporal_{variable}_{m.lower().replace(" ", "_")}.png' 
                                          for m in model.results.keys()]
                })
            
            results_all[variable] = {
                'models': {
                    model_name: {
                        'metrics': data['metrics'],
                        'model_type': type(data['model']).__name__
                    }
                    for model_name, data in model.results.items()
                },
                'best_model': min(model.results.items(), 
                                key=lambda x: x[1]['metrics']['RMSE'])[0],
                'data_info': {
                    'total_records': len(model.data),
                    'train_records': len(model.X_train),
                    'test_records': len(model.X_test),
                    'features_count': len(model.features)
                },
                'generated_plots': plots_info,
                'optimization_config': config
            }
            
            print(f"\n✅ {variable.upper()} processado com sucesso!")
            
        except Exception as e:
            print(f"\n❌ Erro ao processar {variable}: {str(e)}")
            results_all[variable] = {'error': str(e)}
    
    return results_all