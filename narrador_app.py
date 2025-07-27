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
# Habilita CORS para todas as origens, permitindo que o frontend (dashboard.php) acesse esta API.
# Expor cabeçalhos é importante para que o frontend possa ver headers como X-Model-Used.
CORS(application, origins="*", expose_headers=['X-Model-Used'])

# =========================================================================
# - CONFIGURAÇÕES -
# =========================================================================

# 1. Obter a chave da API a partir das variáveis de ambiente do Render
api_key = os.environ.get("GEMINI_API_KEY")

# 2. Configuração do Modelo TTS
# OS ÚNICOS MODELOS QUE ATUALMENTE SUPORTAM TTS SÃO:
# 'gemini-2.5-pro-preview-tts' E 'gemini-2.5-flash-preview-tts'.
# O código deve usar um destes.
# Se você quiser alternar entre eles, pode adicionar uma variável de ambiente
# para definir qual usar, ou simplesmente deixar um como padrão e saber que
# o outro pode ser habilitado posteriormente.

# Definimos aqui o modelo que estamos usando ATUALMENTE.
# Para fins de teste e futuro, podemos manter um que seja mais robusto ou o que estiver funcionando.
# Se gemini-2.5-pro-preview-tts é o que você usa agora e funciona, vamos mantê-lo.
# Se quiser habilitar o flash para teste futuro, poderíamos adicionar uma lógica.

# MODELO ATUALMENTE EM USO (e que você sabe que funciona para TTS)
CURRENT_TTS_MODEL = 'gemini-2.5-pro-preview-tts'

# Se você quiser que o código possa selecionar entre os modelos TTS (com base em uma variável de ambiente, por exemplo):
# MODEL_TO_USE_FOR_TTS = os.environ.get("TTS_MODEL", 'gemini-2.5-pro-preview-tts') # Padrão para o Pro

# Para este exemplo, vamos manter o que está funcionando e adicionar o outro como uma opção comentada.
MODEL_TO_USE_FOR_TTS = CURRENT_TTS_MODEL # Usando o modelo Pro que você mencionou que funciona.

# Se você quiser testar o flash preview, descomente a linha abaixo e certifique-se de que ele está
# funcionando corretamente com a API do Google.
# MODEL_TO_USE_FOR_TTS = 'gemini-2.5-flash-preview-tts'

logger.info(f"Modelo TTS configurado para uso: {MODEL_TO_USE_FOR_TTS}")

# 3. Configurar a chave da API para o SDK do Google
if api_key:
    genai.configure(api_key=api_key)
    logger.info("Chave da API Gemini configurada.")
else:
    logger.error("Chave da API GEMINI_API_KEY não encontrada nas variáveis de ambiente. TTS pode falhar.")

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
    health_status = {"status": "ok"}
    if not api_key:
        health_status["warning"] = "GEMINI_API_KEY não configurada. TTS pode não funcionar."
    elif MODEL_TO_USE_FOR_TTS not in ['gemini-2.5-pro-preview-tts', 'gemini-2.5-flash-preview-tts']:
         # Adicionando uma verificação se o modelo selecionado é um dos conhecidos TTS
         health_status["warning"] = f"Modelo TTS '{MODEL_TO_USE_FOR_TTS}' não é um modelo TTS conhecido. Verifique a configuração."
         
    return jsonify(health_status), 200

@application.route('/api/generate-audio', methods=['POST'])
def generate_audio_endpoint():
    """
    Endpoint principal para geração de áudio.
    Recebe JSON com 'text' e 'voice'.
    Retorna um arquivo de áudio WAV.
    """
    try:
        # 1. Validação da Chave da API e Configuração
        if not api_key:
            logger.error("Chave da API GEMINI_API_KEY ausente.")
            return jsonify({"error": "Configuração do servidor incompleta: Chave da API ausente."}), 500

        # Reconfigura o genai aqui para garantir que use a API key carregada.
        # Isso é bom caso a chave seja carregada dinamicamente ou após o init.
        genai.configure(api_key=api_key)

        # 2. Validação do Payload JSON
        data = request.get_json()
        if not data:
            logger.warning("Requisição inválida: corpo JSON ausente.")
            return jsonify({"error": "Requisição inválida, corpo JSON ausente."}), 400

        text_to_narrate = data.get('text')
        voice_name = data.get('voice') # Ex: "en-US-Wavenet-F"

        if not text_to_narrate:
            logger.warning("Texto ausente ou vazio na requisição.")
            return jsonify({"error": "O texto não pode estar vazio."}), 400

        if not voice_name:
             logger.warning("Voz ausente na requisição.")
             return jsonify({"error": "O nome da voz não pode estar vazio."}), 400
        
        # Validação adicional: Verificar se o modelo escolhido é um modelo TTS válido
        if MODEL_TO_USE_FOR_TTS not in ['gemini-2.5-pro-preview-tts', 'gemini-2.5-flash-preview-tts']:
            logger.error(f"Modelo TTS selecionado '{MODEL_TO_USE_FOR_TTS}' não é um modelo TTS suportado pelo Google AI.")
            return jsonify({"error": f"Modelo TTS '{MODEL_TO_USE_FOR_TTS}' não é suportado. Por favor, use 'gemini-2.5-pro-preview-tts' ou 'gemini-2.5-flash-preview-tts'."}), 400


        # 3. Configuração e Chamada para a API do Google TTS
        logger.info(f"Chamando API Google TTS com modelo: {MODEL_TO_USE_FOR_TTS}...")
        try:
            # A forma correta de chamar TTS é através de genai.generate_content
            # com as configurações de áudio especificadas no generation_config.
            
            response = genai.generate_content(
                model=MODEL_TO_USE_FOR_TTS, # Usa o modelo configurado
                contents=[{
                    "role": "user",
                    "parts": [{"text": text_to_narrate}],
                }],
                generation_config={
                    "response_mime_type": "audio/wav", # Solicita áudio no formato WAV
                    "audio_encoding": "LINEAR16", # Codificação comum para WAV
                    "sample_rate_hertz": 24000, # Frequência de amostragem (comum para TTS)
                    "voice_config": { # Configuração da voz
                         "voice_name": voice_name # Passa o nome da voz
                    }
                }
            )

            # Verificar se a resposta contém os dados de áudio esperados
            if hasattr(response, 'audio') and hasattr(response.audio, 'data'):
                audio_data = response.audio.data
                logger.info("Áudio recebido com sucesso da API do Google.")
            else:
                # Se não tiver a estrutura esperada, logar a resposta completa para depuração
                logger.error(f"Resposta inesperada da API de TTS do modelo '{MODEL_TO_USE_FOR_TTS}'.")
                logger.error(f"Resposta recebida: {response}")
                return jsonify({"error": f"A API do modelo '{MODEL_TO_USE_FOR_TTS}' não retornou dados de áudio válidos. Verifique a documentação do Google para este modelo."}), 500

        except AttributeError as ae:
            # Captura erros relacionados a atributos inexistentes (como 'text_to_speech' que não existe)
            # ou se o modelo selecionado realmente não suportar a funcionalidade TTS.
            logger.error(f"AttributeError ao chamar a API TTS com o modelo '{MODEL_TO_USE_FOR_TTS}': {ae}", exc_info=True)
            return jsonify({"error": f"Erro de atributo ao usar o modelo '{MODEL_TO_USE_FOR_TTS}'. Verifique se este modelo suporta TTS e se a chamada está correta. Detalhe: {str(ae)}"}), 502

        except Exception as tts_api_error:
            # Captura outros erros da API do Google (conexão, autenticação, etc.)
            logger.error(f"Erro inesperado ao chamar a API do Google TTS com o modelo '{MODEL_TO_USE_FOR_TTS}': {tts_api_error}", exc_info=True)
            return jsonify({"error": f"Falha na comunicação com a API de TTS do Google: {str(tts_api_error)}"}), 502 # 502 para erro upstream


        # 4. Retorno da Resposta
        # Serve o áudio recebido diretamente como um arquivo WAV.
        logger.info("Preparando resposta com áudio WAV...")
        return send_file(
            io.BytesIO(audio_data),
            mimetype='audio/wav', # O formato que esperamos receber
            as_attachment=True, # Faz com que o navegador baixe o arquivo
            download_name='narracao.wav' # Nome padrão do arquivo baixado
        )

    except Exception as e:
        # Captura qualquer outro erro inesperado no fluxo da requisição
        logger.error(f"Ocorreu um erro crítico na API de Narração: {e}", exc_info=True)
        return jsonify({"error": "Ocorreu um erro interno no servidor."}), 500

# =========================================================================
# - PONTO DE ENTRADA PARA O GUNICORN -
# =========================================================================
# O Render usa o Gunicorn para rodar a aplicação WSGI.
# Ele procura por uma variável chamada 'application' por padrão.
# Nosso objeto Flask se chama 'application'.

if __name__ == '__main__':
    # Este bloco é executado apenas quando rodamos o script localmente (python narrador_app.py)
    # O Render não executa este bloco, ele usa o Gunicorn.
    # Usamos PORT 8080 como padrão ou a variável de ambiente PORT, comum no Render.
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Rodando Flask app localmente na porta {port}")
    # Para desenvolvimento local, debug=True é útil. No Render, isso é desativado.
    application.run(debug=True, host='0.0.0.0', port=port)