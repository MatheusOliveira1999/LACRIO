"""
Aplicação Flask para interface web
"""

from flask import Flask
import os
from config.settings import UPLOAD_FOLDER, MAX_CONTENT_LENGTH


def create_app():
    """
    Factory function para criar aplicação Flask
    
    Returns:
    --------
    Flask : Aplicação Flask configurada
    """
    app = Flask(__name__)
    
    # Configurações
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    
    # Registrar blueprints/rotas
    from .routes import register_routes
    register_routes(app)
    
    # Handler para erros de upload muito grande
    @app.errorhandler(413)
    def too_large(e):
        return {"error": "Arquivo muito grande. Máximo: 100MB"}, 413
    
    # Handler para erros gerais
    @app.errorhandler(500)
    def internal_error(e):
        return {"error": "Erro interno do servidor"}, 500
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)