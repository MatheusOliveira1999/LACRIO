#!/usr/bin/env python3
"""
Servidor web para o sistema de downscaling climático
"""

import os
import sys

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.web.app import create_app
from config.settings import create_directories


def main():
    """Função principal para executar o servidor web"""
    
    print("🌍 SISTEMA DE DOWNSCALING CLIMÁTICO - INTERFACE WEB")
    print("="*60)
    
    # Criar diretórios necessários
    print("📁 Criando diretórios necessários...")
    create_directories()
    
    # Criar aplicação Flask
    print("🔧 Inicializando aplicação web...")
    app = create_app()
    
    # Configurações do servidor
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    print(f"🚀 Iniciando servidor web...")
    print(f"📍 Endereço: http://{host}:{port}")
    print(f"🔧 Modo debug: {debug}")
    print("="*60)
    print("💡 Pressione Ctrl+C para parar o servidor")
    print("="*60)
    
    try:
        # Iniciar servidor
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n⏹️ Servidor parado pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro ao iniciar servidor: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())