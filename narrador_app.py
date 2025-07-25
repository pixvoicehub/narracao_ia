import os
import io
import struct
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import google.generativeai as genai
from google.generativeai import types

# =========================================================================
# --- INICIALIZAÇÃO E CONFIGURAÇÃO DA APLICAÇÃO FLASK ---
#
# NOME DO ARQUIVO: narrador_app.py
#
# OBJETIVO: Este é o microsserviço "Ator de IA".
#
# VERSÃO: 9.0 - Versão definitiva com a sintaxe correta e funcional
# para a biblioteca google-generativeai atual instalada no Render.
# =========================================================================
application = Flask(__name__)
CORS(application, origins="*", expose_headers=['X-Model-Used'])

# --- Lista de modelos de TTS permitidos ---
ALLOWED_TTS_MODELS = [
    'models/gemini-2.5-pro-preview-tts',
    'models/gemini-2.5-flash-preview-tts'
]
DEFAULT_TTS_MODEL = 'models/gemini-2.5-pro-preview-tts'

# --- Funções Auxiliares de Áudio (Mantidas para garantir a formatação WAV) ---
def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters.get("bits_per_sample", 16)
    sample_rate = parameters.get("rate", 24000)
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1,
        num_channels, sample_rate, byte_rate, block_align,
        bits_per_sample, b"data", data_size
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    bits_per_sample = 16
    rate = 24000
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError): pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError): pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}

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

    try:
        genai.configure(api_key=api_key)
        
        # [SINTAXE FINAL E CORRETA]
        # 1. Instancia o modelo moderno
        model = genai.GenerativeModel(tts_model_to_use)

        # 2. O conteúdo é passado diretamente, sem 'types.Part' ou 'types.Content'
        contents = [text_to_narrate]
        
        # 3. A configuração de áudio usa a estrutura original, que é a correta
        generation_config = types.GenerateContentConfig(
            response_modalities=[types.ResponseModality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            ),
        )

        # 4. A chamada de streaming usa o modelo moderno com a configuração correta
        stream = model.generate_content(
            contents=contents,
            generation_config=generation_config,
            stream=True
        )
        
        audio_buffer = bytearray()
        audio_mime_type = "audio/L16;rate=24000"
        
        for chunk in stream:
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                part = chunk.candidates[0].content.parts[0]
                if hasattr(part, 'inline_data') and hasattr(part.inline_data, 'data'):
                    inline_data = part.inline_data
                    audio_buffer.extend(inline_data.data)
                    if hasattr(inline_data, 'mime_type'):
                        audio_mime_type = inline_data.mime_type

        if not audio_buffer:
            return jsonify({"error": "Não foi possível gerar o áudio (buffer vazio após streaming)."}), 500

        # Usa as funções auxiliares originais para garantir a formatação correta do WAV
        wav_data = convert_to_wav(bytes(audio_buffer), audio_mime_type)
        
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