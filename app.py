import streamlit as st
import boto3
import json
import pandas as pd
import requests
import os
from dotenv import load_dotenv

#carrega variaveis do .env
load_dotenv()

#Criando a sessão
session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)
client = session.client('bedrock-runtime', region_name='us-west-2')
 
logo_path="logo.jpg"
 
 
# =========================
# Função para carregar arquivo
# =========================
def carregar_eventos(caminho_arquivo):
    try:
        # Tenta múltiplos caminhos
        possiveis_caminhos = [
            caminho_arquivo,
            os.path.join(os.path.expanduser('~'), 'Desktop', 'turist-a.i', 'eventos-buzios.txt'),
            os.path.join(os.getcwd(), 'eventos-buzios.txt')
        ]
       
        for caminho in possiveis_caminhos:
            if os.path.exists(caminho):
                with open(caminho, 'r', encoding='utf-8') as arquivo:
                    return arquivo.read()
       
        raise FileNotFoundError("Arquivo não encontrado em nenhum dos caminhos tentados")
   
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return ""
 
 
# =========================
# Função para chamada ao AWS Bedrock
# =========================
def call_bedrock_model(messages):
   
    # Carrega os eventos de Búzios
    caminho_eventos = os.path.join(os.path.expanduser('~'), 'Desktop', 'turist-a.i', 'eventos-buzios.txt')
    eventos = carregar_eventos(caminho_eventos)
   
    if not eventos:
        eventos = "Nnehum evento em Búzios no Momento"
 
    # Adiciona os eventos ao contexto do primeiro prompt
    if messages and messages[0]["role"] == "user":
        messages[0]["content"] += f"\n\nEventos em Búzios:\n{eventos}"
   
    payload = {
        "messages": [{"role": msg["role"], "content": msg["content"]} for msg in messages],
        "max_tokens": 200000,
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": 1.0,
        "top_p": 0.95,
        "stop_sequences": []
    }
 
 
    try:
        response = client.invoke_model_with_response_stream(
            modelId="anthropic.claude-v2",
            body=json.dumps(payload).encode("utf-8"),
            contentType="application/json",
            accept="application/json"
        )
        output = []
        stream = response.get("body")
        if stream:
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    chunk_data = json.loads(chunk.get("bytes").decode())
                    if chunk_data.get("type") == "content_block_delta":
                        delta = chunk_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            output.append(delta.get("text", ""))
        return "".join(output)
    except Exception as e:
        return f"Erro ao chamar o modelo: {str(e)}"
 
 
# =========================
# Interface do Chat
# =========================
 
if os.path.exists(logo_path):
    col1, col2 = st.columns([1,9])
    with col1:
        st.image(logo_path, width=200)
    with col2:
        st.title("TURIST A.I. - Descubra onde ir, sem perder tempo.")
else:
    st.error("Logomarca não encontrada.")
 
 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    context = """
    *Prompt*: A minha empresa se chama TURIST A.I. Você é um agente de viagens virtual, invente um nome para você a cada interação, e diga que você é da Turist A.I.
    Utilizando uma linguagem cordial e que seja objetiva, com emojis. Faça perguntas ao usuário, as perguntas necessárias para responder às questões abaixo:
       pergunte que dia ele irá para búzios
        Tipo de evento :
        Gêneros musicais favoritos :
        Atividades preferidas
 
        Faça perguntas diretas, uma pergunta por vez, Se apresente, encante o cliente.
     
    Com base nas informações fornecidas pelo usuário, personalize sugestões de eventos, músicas e atividades na cidade de Búzios, baseada nos dados do arquivo eventos-buzios.txt. fale também sobre o tempo,
    Considere os seguintes, quero que faça uma pergunta por vez, como em uma conversa.
 
    Seu nome é Romário.
    GUARDRAIL:
    Atente-se e responda apenas sobre o contexto enviado. Caso o usuário divague ou fale sobre outra coisa, diga que sua base de dados está preparada apenas para falar sobre eventos de Búzios.
    """
    st.session_state.chat_history.append({"role": "user", "content": context, "hidden": True})  # Adiciona o contexto como oculto
 
user_input = st.text_area("Digite sua mensagem ou personalize o prompt:", key="user_input")
 
 
def add_message_to_history(role, content, hidden=False):
 
    if not hidden:
        if not st.session_state.chat_history or st.session_state.chat_history[-1]["role"] != role:
            st.session_state.chat_history.append({"role": role, "content": content})
    else:
        st.session_state.chat_history.append({"role": role, "content": content, "hidden": True})
 
if st.button("Enviar"):
    if user_input.strip():
        add_message_to_history("user", user_input)
 
 
        with st.spinner("Buscando resposta..."):
            model_response = call_bedrock_model(st.session_state.chat_history)
 
            add_message_to_history("assistant", model_response)
 
 
st.subheader("Histórico do Chat")
for message in st.session_state.chat_history:
    if not message.get("hidden", False):  # Mostra apenas mensagens visíveis
        if message["role"] == "user":
            st.write(f"**Usuário:** {message['content']}")
        elif message["role"] == "assistant":
            st.write(f"**Modelo:** {message['content']}")