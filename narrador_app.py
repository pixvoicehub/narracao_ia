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
# OBJETIVO: Este é o microsserviço "Ator de IA".
#
# VERSÃO: 6.0 - Versão final com a sintaxe mais simples e robusta possível
# para a API de TTS, resolvendo todos os erros de atributos.
# =========================================================================
application = Flask(__name__)
CORS(application, origins="*", expose_headers=['X-Model-Used'])

# --- Lista de modelos de TTS permitidos ---
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

    # 2. Comunicação com a API de TTS (SINTAXE FINAL E SIMPLIFICADA)
    try:
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel(tts_model_to_use)

        # [A CORREÇÃO FINAL E DEFINITIVA]
        # A configuração agora é um dicionário Python simples, sem usar 'types'.
        # A voz é especificada dentro de 'speech_config'.
        generation_config = {
            "response_modality": "AUDIO",
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": voice_name
                    }
                }
            }
        }
        
        # O conteúdo é passado diretamente, como uma lista contendo a string.
        contents = [text_to_narrate]

        # A chamada de streaming usa o modelo com a configuração simplificada.
        stream = model.generate_content(
            contents=contents,
            generation_config=generation_config,
            stream=True
        )
        
        audio_buffer = bytearray()
        
        for chunk in stream:
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                part = chunk.candidates[0].content.parts[0]
                if hasattr(part, 'inline_data') and hasattr(part.inline_data, 'data'):
                    audio_buffer.extend(part.inline_data.data)

        if not audio_buffer:
            return jsonify({"error": "Não foi possível gerar o áudio (buffer vazio após streaming)."}), 500

        # 3. Retorno da Resposta
        wav_data = bytes(audio_buffer)
        response_to_send = make_response(send_file(io.BytesIO(wav_data), mimetype='audio/wav', as_attachment=False))
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