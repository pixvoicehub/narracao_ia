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
# OBJETIVO: Este é o microsserviço "Ator de IA". Sua responsabilidade
# é receber um texto final, um ID de voz e o nome do modelo de TTS a ser
# usado (Pro ou Flash), e retornar o arquivo de áudio WAV correspondente.
#
# VERSÃO: 3.1 - Utiliza a função 'text_to_speech' com a sintaxe correta.
# =========================================================================
application = Flask(__name__)
CORS(application, origins="*", expose_headers=['X-Model-Used'])

# --- Lista de modelos de TTS permitidos para validação de segurança ---
ALLOWED_TTS_MODELS = [
    'models/gemini-2.5-pro-preview-tts',
    'models/gemini-2.5-flash-preview-tts'
]
# Define o modelo padrão de maior qualidade caso nenhum seja especificado
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
    # ----------------------------
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

    # 2. Comunicação com a API de TTS (SINTAXE CORRETA)
    # ----------------------------------------------------
    try:
        genai.configure(api_key=api_key)
        
        # [CORREÇÃO] A chamada para TTS é feita diretamente através da função 'text_to_speech'.
        # Ela recebe o modelo, o texto e a voz como parâmetros diretos.
        response = genai.text_to_speech(
            model=tts_model_to_use,
            text=text_to_narrate,
            voice=voice_name
        )
        
        # A resposta desta função já contém os dados de áudio.
        if not hasattr(response, 'audio') or not hasattr(response.audio, 'data'):
            return jsonify({"error": "A API não retornou dados de áudio válidos. Verifique o texto e o nome da voz."}), 500
            
        audio_data = response.audio.data
        
        # 3. Formatação e Retorno da Resposta
        # -----------------------------------
        response_to_send = make_response(send_file(io.BytesIO(audio_data), mimetype='audio/wav', as_attachment=False))
        response_to_send.headers['X-Model-Used'] = tts_model_to_use
        return response_to_send

    except Exception as e:
        # Log detalhado do erro no servidor para depuração
        print(f"Ocorreu um erro crítico na API de Narração: {e}")
        # Retorna uma mensagem de erro genérica para o cliente
        return jsonify({"error": f"Erro interno no servidor ao gerar áudio: {str(e)}"}), 500

# =========================================================================
# --- EXECUÇÃO DA APLICAÇÃO ---
# =========================================================================
if __name__ == '__main__':
    application.run(debug=True)