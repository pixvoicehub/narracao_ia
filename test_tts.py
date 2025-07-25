# Arquivo: test_tts.py
import os
import google.generativeai as genai

print("--- INICIANDO TESTE DE SINTAXE TTS ---")

try:
    # 1. Pega a chave de API do ambiente, como o Render faz.
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERRO: A variável de ambiente GEMINI_API_KEY não foi encontrada.")
        exit()
    
    genai.configure(api_key=api_key)
    print("API Key configurada com sucesso.")

    # 2. Define os parâmetros do teste.
    model_to_use = 'models/gemini-2.5-pro-preview-tts'
    text_to_narrate = "Se este áudio for gerado, a sintaxe está correta."
    voice_name = "default-pt-br-1" # Uma voz padrão para teste

    print(f"Modelo: {model_to_use}")
    print(f"Voz: {voice_name}")
    print("Tentando gerar áudio...")

    # 3. [O TESTE] Esta é a sintaxe mais provável e robusta.
    model = genai.GenerativeModel(model_to_use)
    
    # A configuração é um dicionário simples, como o erro sugeriu.
    generation_config = {
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {
                    "voice_name": voice_name
                }
            }
        }
    }
    
    # O conteúdo é a string de texto pura.
    response = model.generate_content(
        text_to_narrate,
        generation_config=generation_config
    )

    # 4. Verifica o resultado e salva o arquivo.
    if hasattr(response, 'audio') and hasattr(response.audio, 'data'):
        audio_data = response.audio.data
        with open("audio_de_teste.wav", "wb") as f:
            f.write(audio_data)
        print("\n--- SUCESSO! ---")
        print("O áudio 'audio_de_teste.wav' foi gerado com sucesso na pasta do projeto.")
        print("A sintaxe correta foi descoberta. Agora podemos aplicá-la ao narrador_app.py.")
    else:
        print("\n--- FALHA ---")
        print("A API respondeu, mas não retornou dados de áudio.")
        print("Resposta recebida:", response)

except Exception as e:
    print("\n--- ERRO CRÍTICO DURANTE O TESTE ---")
    print(f"O teste falhou com a seguinte exceção: {e}")