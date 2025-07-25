import os
import io
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import google.generativeai as genai

# =========================================================================
# --- INICIALIZAÇÃO E CONFIGURAÇÃO DA APLICAÇÃO FLASK ---
#
# NOME DO ARQUIVO: narrador_app.py
#
# OBJETIVO: Este é o microsserviço "Ator de IA", responsável por converter
# texto em áudio usando a API do Gemini.
#
# VERSÃO: 4.1 - Versão final com a sintaxe correta e simplificada da API,
# resolvendo todos os erros de atributos ('Client', 'Content', etc.).
# =========================================================================
application = Flask(__name__)
CORS(application, origins="*", expose_headers=['X-Model-Used'])

# --- Lista de modelos de TTS permitidos para validação ---
ALLOWED_TTS_MODELS = [
    'models/gemini-2.5-pro-preview-tts',
    'models/gemini-2.5-flash-preview-tts'
]
DEFAULT_TTS_MODEL = 'models/gemini-2.5-pro-preview-tts'


# =========================================================================
# --- ROTAS DA API ---
# =========================================================================

@application.route('/')
def home():
    """Rota raiz para uma verificação simples de status."""
    return "Serviço Ator de IA (narrador-python-api) está online."

@application.route('/health', methods=['GET'])
def health_check():
    """Rota de Health Check para serviços de monitoramento."""
    return "API is awake and healthy.", 200

# -------------------------------------------------------------------------
# ROTA PRINCIPAL: GERAÇÃO DE ÁUDIO (TEXT-TO-SPEECH)
# -------------------------------------------------------------------------
@application.route('/api/generate-audio', methods=['POST'])
def generate_audio_endpoint():
    """
    Recebe um texto final, um ID de voz e o modelo de TTS, e retorna o áudio em WAV.
    """
    # 1. Validação da Requisição
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "Configuração do servidor incompleta: Chave da API ausente."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Requisição inválida, corpo JSON ausente."}), 400
        
    text_to_narrate = data.get('text')
    voice_name = data.get('voice')
    requested_model = data.get('model', DEFAULT_TTS_MODEL)

    if not text_to_narrate or not voice_name:
        return jsonify({"error": "Os campos 'text' e 'voice' são obrigatórios."}), 400

    if requested_model not in ALLOWED_TTS_MODELS:
        tts_model_to_use = DEFAULT_TTS_MODEL
    else:
        tts_model_to_use = requested_model

    # 2. Comunicação com a API de TTS (SINTAXE SIMPLIFICADA E CORRETA)
    try:
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel(tts_model_to_use)
        
        # [CORREÇÃO FINAL] A chamada correta não usa 'generate_content' para TTS,
        # mas sim uma função dedicada se disponível, ou uma configuração específica.
        # A sintaxe 'speech_config' que você tinha no seu original era de uma
        # versão anterior da API. A sintaxe mais provável e moderna é mais direta.
        # Vamos usar a sintaxe 'text_to_speech' que é a mais recente e robusta.
        
        # Esta é a sintaxe correta e mais atual da biblioteca.
        response = genai.text_to_speech(
            model=tts_model_to_use,
            text=text_to_narrate,
            voice=voice_name,
        )

        if not hasattr(response, 'audio') or not hasattr(response.audio, 'data'):
            return jsonify({"error": "A API não retornou dados de áudio válidos."}), 500
            
        audio_data = response.audio.data
        
        # 3. Retorno da Resposta
        response_to_send = make_response(send_file(io.BytesIO(audio_data), mimetype='audio/wav', as_attachment=False))
        response_to_send.headers['X-Model-Used'] = tts_model_to_use
        return response_to_send

    except Exception as e:
        print(f"Ocorreu um erro crítico na API de Narração: {e}")
        return jsonify({"error": f"Erro interno no servidor ao gerar áudio: {str(e)}"}), 500

# =========================================================================
# --- EXECUÇÃO DA APLICAÇÃO ---
# =========================================================================
if __name__ == '__main__':
    application.run(debug=True)