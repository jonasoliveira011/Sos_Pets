import boto3
import json
import uuid
from datetime import datetime
import os
import pandas as pd
import PyPDF2
import base64
import io
from PIL import Image

PROFILE_NAME = os.environ.get('AWS_PROFILE', 'edn')

def get_boto3_client(service_name, region_name='us-east-1', profile_name='edn'):
    try:
        session = boto3.Session(profile_name=profile_name,region_name=region_name)
        client = session.client(service_name)
        
        print(f"DEBUG: Usando IAM Role para acessar '{service_name}' na região '{region_name}'")
        return client
        
    except Exception as e:
        print(f"ERRO: Não foi possível acessar a AWS: {str(e)}")
        print("ATENÇÃO: Verifique se o IAM Role está corretamente associado à instância EC2.")
        return None

def read_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Erro ao ler PDF: {str(e)}"

def read_txt(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Erro ao ler TXT: {str(e)}"

def read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.to_string()
    except Exception as e:
        return f"Erro ao ler CSV: {str(e)}"

def convert_image_to_base64(uploaded_file):
    try:
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        print(f"Erro ao converter imagem para base64: {str(e)}")
        return None

def process_image_file(uploaded_file):
    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=85)
        img_buffer.seek(0)
        
        return base64.b64encode(img_buffer.getvalue()).decode('utf-8'), 'image/jpeg'
    except Exception as e:
        print(f"Erro ao processar imagem: {str(e)}")
        return None, None
    
def format_context(context, source="Contexto Adicional"):
    return f"\n\n{source}:\n{context}\n\n"

def generate_chat_prompt(user_message, conversation_history=None, context=""):
    system_prompt = """
    Você é atendente da S.O.S Pets. Nosso objetivo é ajudar a encontrar animais abandonados para estimular a doação dos animais abandonados.

    O usuário deve enviar a foto do animal e você deve criar um anúncio com a descrição para a doação do animal. 

    Você precisa dos seguintes dados: 

    FOTO DO ANIMAL 
    NOME DO ANIMAL 
    TELEFONE DO TUTOR 
    LOCALIZAÇÃO 
    RAÇA 
    DESCRIÇÃO DO ANIMAL (Você deve inferir através da imagem enviada). 
    PORTE DO ANIMAL 

    Pergunte uma coisa de cada vez. Seja interativo com emojis. Seja aberto, coloquial, use um tom amigável. Conduza o tutor a cuidar do animal até a doação.
    Quando mandarem foto de algum animal silvestre, direcione o usuário a ligar para algum orgão ambiental especializado.

    Recolher descrição
    """

    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
      conversation_context = "Histórico da conversa:\n"
      recent_messages = conversation_history[-8:]
      for message in recent_messages:
        role = "Usuário" if message.get('role') == 'user' else "Assistente"
        conversation_context += f"{role}: {message.get('content')}\n"
      conversation_context += "\n"

    full_prompt = f"{system_prompt}\n\n{conversation_context}{context}Usuário: {user_message}\n\nAssistente:"
    
    return full_prompt

def invoke_bedrock_model(prompt, inference_profile_arn, model_params=None):
    if model_params is None:
        model_params = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 200,
        "max_tokens": 800
        }

    bedrock_runtime = get_boto3_client('bedrock-runtime')

    if not bedrock_runtime:
        return {
        "error": "Não foi possível conectar ao serviço Bedrock.",
        "answer": "Erro de conexão com o modelo.",
        "sessionId": str(uuid.uuid4())
        }

    try:
        body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": model_params["max_tokens"],
        "temperature": model_params["temperature"],
        "top_p": model_params["top_p"],
        "top_k": model_params["top_k"],
        "messages": [
        {
        "role": "user",
        "content": [
        {
        "type": "text",
        "text": prompt
        }
    ]
    }
    ]
    })

        response = bedrock_runtime.invoke_model(
        modelId=inference_profile_arn,
        body=body,
        contentType="application/json",
        accept="application/json"
    )
        
        response_body = json.loads(response['body'].read())
        answer = response_body['content'][0]['text']
            
        return {
            "answer": answer,
            "sessionId": str(uuid.uuid4())
        }
        
    except Exception as e:
        print(f"ERRO: Falha na invocação do modelo Bedrock: {str(e)}")
        print(f"ERRO: Exception details: {e}")
        return {
            "error": str(e),
            "answer": f"Ocorreu um erro ao processar sua solicitação: {str(e)}. Por favor, tente novamente.",
            "sessionId": str(uuid.uuid4())
        }

def read_pdf_from_uploaded_file(uploaded_file):
    try:
        import io
        from PyPDF2 import PdfReader
        
        pdf_bytes = io.BytesIO(uploaded_file.getvalue())
        reader = PdfReader(pdf_bytes)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Erro ao ler PDF: {str(e)}"
    
def read_txt_from_uploaded_file(uploaded_file):
    try:
        return uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        return f"Erro ao ler TXT: {str(e)}"

def read_csv_from_uploaded_file(uploaded_file):
    try:
        import pandas as pd
        import io
        
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
        return df.to_string()
    except Exception as e:
        return f"Erro ao ler CSV: {str(e)}"