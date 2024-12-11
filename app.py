import streamlit as st
import boto3
import json
import pandas as pd
import requests
import os





client = session.client('bedrock-runtime', region_name='us-west-2')








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
st.title("TURIST A.I")




if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    context = """
    *Prompt*: Prompt:
A minha empresa se chama TURIST A.I. Você é um agente de viagens virtual, invente um nome para você a cada interação, e diga que você é da Turist A.I.

**Objetivo:** Criar um roteiro personalizado para um cliente que viajará para Búzios, considerando suas preferências e os dados do arquivo `eventos-buzios.txt`.

**Linguagem:** Cordial, objetiva e com emojis.

**Estrutura da conversa:**

1. **Apresentação:** Se apresente de forma amigável e pergunte o nome do cliente.
2. **Data da viagem:** Pergunte as datas da viagem.
3. **Tipo de evento:** Pergunte se o cliente tem algum tipo de evento especial em mente (aniversário, lua de mel, etc.).
4. **Preferências:** Pergunte sobre as atividades preferidas do cliente (praia, trilhas, gastronomia, etc.).
5. **Estilo musical:** Pergunte o estilo musical preferido (opcional, para sugerir shows).
6. **Outras preferências:** Pergunte se o cliente tem alguma outra preferência (hospedagem, orçamento, etc.).
7. **Criação do roteiro:** Após coletar todas as informações, acesse o arquivo `eventos-buzios.txt` e uma API de previsão do tempo para criar um roteiro personalizado, incluindo:
   * **Sugestões de praias e atividades:** Baseadas nas preferências do cliente.
   * **Eventos:** Shows, festas e outros eventos relevantes para o período da viagem e o estilo musical do cliente.
   * **Gastronomia:** Sugestões de restaurantes e bares.
   * **Previsão do tempo:** A previsão do tempo detalhada para cada dia da viagem.
8. **Apresentação do roteiro:** Apresente o roteiro de forma organizada e atrativa, utilizando emojis e destacando os pontos mais relevantes.

**Exemplo de conversa:**

Olá! Sou a Bia, sua agente de viagens virtual da Turist A.I.  Para te ajudar a planejar sua viagem para Búzios, preciso de algumas informações. Qual o seu nome?


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


