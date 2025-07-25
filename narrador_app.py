import os
import struct
import io
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
# [CORREÇÃO 1] A importação correta para a biblioteca moderna
import google.generativeai as genai
from google.generativeai import types

# =========================================================================
# --- INICIALIZAÇÃO E CONFIGURAÇÃO DA APLICAÇÃO FLASK ---
#
# NOME DO ARQUIVO: narrador_app.py
#
# OBJETIVO: Este é o microsserviço "Ator de IA", baseado na lógica original
# funcional, mas adaptado para a sintaxe da biblioteca atual do Render.
#
# VERSÃO: 20.0 - Versão Final e Estável.
# =========================================================================
application = Flask(__name__) # [CORREÇÃO 2] Renomeado de 'app' para 'application' para o Render
CORS(application, origins="*", expose_headers=['X-Model-Used'])

# --- Funções Auxiliares de Áudio (do seu código original) ---
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
    return "Backend do Gerador de Narração está online."

@application.route('/health', methods=['GET'])
def health_check():
    return "API is awake and healthy.", 200

@application.route('/api/generate-audio', methods=['POST'])
def generate_audio_endpoint():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "Configuração do servidor incompleta: Chave da API ausente."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Requisição inválida, corpo JSON ausente."}), 400
        
    text_to_narrate = data.get('text')
    voice_name = data.get('voice')
    # Recebe o modelo do PHP, mas trava no Pro para estabilidade, como combinado.
    model_to_use = "models/gemini-2.5-pro-preview-tts"

    if not text_to_narrate or not voice_name:
        return jsonify({"error": "Os campos 'text' e 'voice' são obrigatórios."}), 400

    try:
        genai.configure(api_key=api_key)
        
        # [TRADUÇÃO 1] Em vez de 'genai.Client()', a API moderna usa 'genai.GenerativeModel()'.
        model = genai.GenerativeModel(model_to_use)

        # [TRADUÇÃO 2] A estrutura de 'contents' com 'types.Part' é a correta para a configuração abaixo.
        contents = [types.Part.from_text(text_to_narrate)]
        
        # [TRADUÇÃO 3] A estrutura de 'GenerateContentConfig' do seu código original é a correta
        # para especificar que queremos APENAS áudio, resolvendo o erro "TEXT not supported".
        generation_config = types.GenerateContentConfig(
            response_modalities=[types.ResponseModality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            ),
        )
        
        audio_buffer = bytearray()
        audio_mime_type = "audio/L16;rate=24000"
        
        # [TRADUÇÃO 4] Em vez de 'client.models.generate_content_stream', a chamada moderna é 'model.generate_content'.
        stream = model.generate_content(
            contents=contents,
            generation_config=generation_config,
            stream=True
        )
        
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

        wav_data = convert_to_wav(bytes(audio_buffer), audio_mime_type)
        
        response = make_response(send_file(io.BytesIO(wav_data), mimetype='audio/wav', as_attachment=False))
        response.headers['X-Model-Used'] = "Pro"
        return response

    except Exception as e:
        print(f"Ocorreu um erro crítico na API: {e}")
        return jsonify({"error": f"Erro interno no servidor ao gerar áudio: {str(e)}"}), 500

# =========================================================================
# --- EXECUÇÃO DA APLICAÇÃO ---
# =========================================================================
if __name__ == '__main__':
    application.run(debug=True)