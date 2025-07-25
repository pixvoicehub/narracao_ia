import os
import struct
import io
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import google.generativeai as genai
from google.generativeai import types

# =========================================================================
# --- INICIALIZAÇÃO E CONFIGURAÇÃO DA APLICAÇÃO FLASK ---
#
# NOME SUGERIDO PARA O ARQUIVO: narrador_app.py
#
# OBJETIVO: Este é o microsserviço "Ator de IA". Sua responsabilidade
# é receber um texto final (já humanizado pelo Diretor de IA), um ID de
# voz e o nome do modelo de TTS a ser usado (Pro ou Flash), e retornar
# o arquivo de áudio WAV correspondente.
# =========================================================================
application = Flask(__name__)
# Habilita CORS para permitir requisições do seu orquestrador PHP
CORS(app, origins="*", expose_headers=['X-Model-Used'])

# [NOVO] Lista de modelos de TTS permitidos para validação de segurança
ALLOWED_TTS_MODELS = [
    'models/gemini-2.5-pro-preview-tts',
    'models/gemini-2.5-flash-preview-tts'
]
# Define o modelo padrão de maior qualidade caso nenhum seja especificado
DEFAULT_TTS_MODEL = 'models/gemini-2.5-pro-preview-tts'

# =========================================================================
# --- FUNÇÕES AUXILIARES DE ÁUDIO ---
# (Estas funções permanecem inalteradas)
# =========================================================================

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Converte os dados de áudio brutos (L16) em um formato de arquivo WAV completo com cabeçalho."""
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
    """Extrai a taxa de amostragem e a profundidade de bits do mime_type."""
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

@app.route('/')
def home():
    """Rota raiz para uma verificação simples de status."""
    return "Serviço Ator de IA (narrador-python-api) está online."

@app.route('/health', methods=['GET'])
def health_check():
    """Rota de Health Check para serviços de monitoramento."""
    return "API is awake and healthy.", 200

# -------------------------------------------------------------------------
# ROTA PRINCIPAL: GERAÇÃO DE ÁUDIO (TEXT-TO-SPEECH)
# -------------------------------------------------------------------------
@app.route('/api/generate-audio', methods=['POST'])
def generate_audio_endpoint():
    """
    Recebe um texto final, um ID de voz e o modelo de TTS a ser usado,
    e retorna o áudio correspondente em formato WAV.
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
    # [NOVO] Recebe o modelo de TTS escolhido pelo usuário a partir do PHP
    requested_model = data.get('model', DEFAULT_TTS_MODEL)

    if not text_to_narrate or not voice_name:
        return jsonify({"error": "Os campos 'text' e 'voice' são obrigatórios."}), 400

    # [NOVO] Valida se o modelo recebido é um dos permitidos para evitar erros e injeção
    if requested_model not in ALLOWED_TTS_MODELS:
        tts_model_to_use = DEFAULT_TTS_MODEL
    else:
        tts_model_to_use = requested_model

    # 2. Geração de Áudio com o Modelo Dinâmico
    # ----------------------------------------
    try:
        genai.configure(api_key=api_key)
        client = genai.Client(api_key=api_key)
        
        contents_for_tts = [types.Content(role="user", parts=[types.Part.from_text(text=text_to_narrate)])]
        
        tts_config = types.GenerateContentConfig(
            response_modalities=[types.ResponseModality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            ),
        )
        
        # [ALTERADO] A variável 'tts_model_to_use' é usada dinamicamente na chamada da API,
        # permitindo a escolha entre os modelos Pro e Flash.
        stream = client.models.generate_content_stream(
            model=tts_model_to_use, 
            contents=contents_for_tts, 
            config=tts_config
        )
        
        audio_buffer = bytearray()
        audio_mime_type = "audio/L16;rate=24000"

        for chunk in stream:
            if (chunk.candidates and chunk.candidates[0].content and
                chunk.candidates[0].content.parts and chunk.candidates[0].content.parts[0].inline_data and
                chunk.candidates[0].content.parts[0].inline_data.data):
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                audio_buffer.extend(inline_data.data)
                audio_mime_type = inline_data.mime_type

        if not audio_buffer:
            return jsonify({"error": "Não foi possível gerar o áudio (buffer vazio)."}), 500

        # 3. Formatação e Retorno da Resposta
        # -----------------------------------
        wav_data = convert_to_wav(bytes(audio_buffer), audio_mime_type)
        
        response = make_response(send_file(io.BytesIO(wav_data), mimetype='audio/wav', as_attachment=False))
        response.headers['X-Model-Used'] = tts_model_to_use # Informa qual modelo foi realmente usado
        return response

    except Exception as e:
        print(f"Ocorreu um erro crítico na API de Narração: {e}")
        return jsonify({"error": f"Erro interno no servidor ao gerar áudio: {str(e)}"}), 500

# =========================================================================
# --- EXECUÇÃO DA APLICAÇÃO ---
# =========================================================================
if __name__ == '__main__':
    application.run(debug=True)