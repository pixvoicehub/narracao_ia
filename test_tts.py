# Arquivo: test_tts.py
import os
import sys
import google.generativeai as genai
from google.generativeai import types

# =========================================================================
# --- SCRIPT DE TESTE DE SINTAXE PARA GEMINI TTS ---
# Este script tentará várias sintaxes para gerar áudio.
# O resultado no log do Render nos dirá qual delas funciona.
# =========================================================================

print("--- INICIANDO TESTE DE SINTAXE TTS ---")

# --- Configuração Inicial ---
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("!!! ERRO FATAL: A variável de ambiente GEMINI_API_KEY não foi encontrada.")
        sys.exit(1) # Encerra o script com erro
    
    genai.configure(api_key=api_key)
    print("OK: API Key configurada com sucesso.")

    model_to_use = 'models/gemini-2.5-pro-preview-tts'
    text_to_narrate = "Teste de áudio bem-sucedido."
    voice_name = "default-pt-br-1"
except Exception as e:
    print(f"!!! ERRO FATAL na configuração: {e}")
    sys.exit(1)

# --- TENTATIVA 1: Sintaxe mais moderna ('text_to_speech') ---
try:
    print("\n--- TENTATIVA 1: Usando genai.text_to_speech() ---")
    response = genai.text_to_speech(model=model_to_use, text=text_to_narrate, voice=voice_name)
    if hasattr(response, 'audio') and hasattr(response.audio, 'data'):
        print(">>> SUCESSO NA TENTATIVA 1! A sintaxe correta é 'genai.text_to_speech()'.")
        sys.exit(0) # Encerra o script com sucesso
    else:
        print(">>> FALHA NA TENTATIVA 1: A API respondeu, mas sem dados de áudio.")
except Exception as e:
    print(f">>> FALHA NA TENTATIVA 1: Ocorreu um erro: {e}")


# --- TENTATIVA 2: Sintaxe híbrida (modelo moderno + config antiga) ---
try:
    print("\n--- TENTATIVA 2: Usando a estrutura de 'types.GenerateContentConfig' ---")
    model = genai.GenerativeModel(model_to_use)
    contents = [types.Part.from_text(text_to_narrate)]
    generation_config = types.GenerateContentConfig(
        response_modalities=[types.ResponseModality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        ),
    )
    stream = model.generate_content(contents=contents, generation_config=generation_config, stream=True)
    
    # Apenas verificamos se o stream foi criado
    if stream:
        print(">>> SUCESSO NA TENTATIVA 2! A sintaxe correta é a híbrida com 'types.GenerateContentConfig'.")
        sys.exit(0)
    else:
        print(">>> FALHA NA TENTATIVA 2: O stream não foi criado.")
except Exception as e:
    print(f">>> FALHA NA TENTATIVA 2: Ocorreu um erro: {e}")


# --- TENTATIVA 3: Sintaxe mais simples (modelo moderno + dicionário simples) ---
try:
    print("\n--- TENTATIVA 3: Usando a estrutura de dicionário simples ---")
    model = genai.GenerativeModel(model_to_use)
    contents = [text_to_narrate]
    generation_config = {
        "response_modality": "AUDIO",
        "speech_config": { "voice_config": { "prebuilt_voice_config": { "voice_name": voice_name } } }
    }
    stream = model.generate_content(contents=contents, generation_config=generation_config, stream=True)
    
    if stream:
        print(">>> SUCESSO NA TENTATIVA 3! A sintaxe correta é a híbrida com dicionário.")
        sys.exit(0)
    else:
        print(">>> FALHA NA TENTATIVA 3: O stream não foi criado.")
except Exception as e:
    print(f">>> FALHA NA TENTATIVA 3: Ocorreu um erro: {e}")

print("\n--- FIM DO TESTE: Nenhuma das sintaxes testadas funcionou. O erro é outro. ---")
sys.exit(1)