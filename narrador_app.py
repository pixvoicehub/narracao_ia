# narrador_app.py
# =========================================================================
# - API DE NARRAÇÃO POR IA v2.0 (O ATOR) -
#
# OBJETIVO: Receber texto e voz via JSON, chamar a API do Google TTS,
# converter o áudio para WAV e retornar o arquivo de áudio.
# =========================================================================

import os
import io
import logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import google.generativeai as genai

# Configuração do Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialização da Aplicação Flask
application = Flask(__name__)
CORS(application, origins="*", expose_headers=['X-Model-Used'])

# =========================================================================
# - CONFIGURAÇÕES -
# =========================================================================

# 1. Obter a chave da API a partir das variáveis de ambiente
api_key = os.environ.get("GEMINI_API_KEY")

# 2. Configurações de modelo TTS
ALLOWED_TTS_MODELS = ['models/gemini-2.0-flash-exp', 'models/gemini-1.5-pro-002'] # Modelos mais estáveis para TTS
DEFAULT_TTS_MODEL = 'models/gemini-2.0-flash-exp' # Usando um modelo mais estável como padrão

# 3. Configurar a chave da API para o SDK do Google
if api_key:
    genai.configure(api_key=api_key)
else:
    logger.error("Chave da API GEMINI_API_KEY não encontrada nas variáveis de ambiente.")

# =========================================================================
# - ROTAS DA API -
# =========================================================================

@application.route('/')
def home():
    """Rota raiz para uma verificação simples de status."""
    return "Serviço Ator de IA (narrador-python-api) está online."

@application.route('/health')
def health_check():
    """Endpoint para verificação de saúde do serviço."""
    return jsonify({"status": "ok"}), 200

@application.route('/api/generate-audio', methods=['POST'])
def generate_audio_endpoint():
    """
    Endpoint principal para geração de áudio.
    Recebe JSON com 'text' e 'voice'.
    Retorna um arquivo de áudio WAV.
    """
    try:
        # 1. Validação da Chave da API
        if not api_key:
            logger.error("Chave da API GEMINI_API_KEY ausente.")
            return jsonify({"error": "Configuração do servidor incompleta: Chave da API ausente."}), 500

        # 2. Validação do Payload JSON
        data = request.get_json()
        if not data:
            logger.warning("Requisição inválida: corpo JSON ausente.")
            return jsonify({"error": "Requisição inválida, corpo JSON ausente."}), 400

        text_to_narrate = data.get('text')
        voice_name = data.get('voice')

        if not text_to_narrate:
            logger.warning("Texto ausente ou vazio na requisição.")
            return jsonify({"error": "O texto não pode estar vazio."}), 400

        if not voice_name:
             logger.warning("Voz ausente na requisição.")
             return jsonify({"error": "O nome da voz não pode estar vazio."}), 400

        # 3. Configuração do Modelo TTS (usando o padrão por enquanto)
        # O modelo é escolhido internamente pela API com base na voz.
        # Vamos usar o modelo padrão definido.
        tts_model_to_use = DEFAULT_TTS_MODEL
        logger.info(f"Usando modelo TTS: {tts_model_to_use}")

        # --- CHAMADA PRINCIPAL PARA A API DO GOOGLE TTS ---
        logger.info("Chamando API text_to_speech do Google...")
        try:
            # A chamada mais direta possível, usando a função 'text_to_speech'
            # que é a mais moderna e lida com toda a configuração internamente.
            response = genai.text_to_speech(
                model=tts_model_to_use,
                text=text_to_narrate,
                voice=voice_name, # Passa o nome da voz diretamente
            )

            # Verificar se a resposta contém os dados de áudio esperados
            if not hasattr(response, 'audio') or not hasattr(response.audio, 'data'):
                # Log da resposta completa para depuração, caso falhe
                logger.error(f"Resposta da API sem áudio: {response}")
                return jsonify({"error": "A API não retornou dados de áudio válidos."}), 500

            audio_data = response.audio.data
            logger.info("Áudio recebido com sucesso da API do Google.")

        except Exception as tts_api_error:
            logger.error(f"Erro ao chamar a API text_to_speech do Google: {tts_api_error}", exc_info=True)
            # Retornar uma mensagem mais específica do erro da API
            return jsonify({"error": f"Falha na comunicação com a API de TTS: {str(tts_api_error)}"}), 502 # 502 para erro upstream


        # 4. Retorno da Resposta
        # A API moderna já retorna o áudio no formato correto.
        # Vamos servir o áudio diretamente como um arquivo WAV.
        logger.info("Preparando resposta com áudio WAV...")
        return send_file(
            io.BytesIO(audio_data),
            mimetype='audio/wav',
            as_attachment=True,
            download_name='narracao.wav'
        )

    except Exception as e:
        # Captura qualquer outro erro inesperado
        logger.error(f"Ocorreu um erro crítico na API de Narração: {e}", exc_info=True)
        return jsonify({"error": "Ocorreu um erro interno no servidor."}), 500

# =========================================================================
# - PONTO DE ENTRADA PARA O GUNICORN -
# =========================================================================
# O Render usa o Gunicorn para rodar a aplicação WSGI.
# Ele procura por uma variável chamada 'application' por padrão.
# Como nosso objeto Flask se chama 'application', está tudo certo.

if __name__ == '__main__':
    # Isso é usado apenas para desenvolvimento local com `python narrador_app.py`
    application.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
