import streamlit as st
import boto3
import json
import pandas as pd
import requests
import os

session = boto3.Session(
    aws_access_key_id='AKIAZOZQF2ANIEVMVJ7H',
    aws_secret_access_key='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',
    region_name='us-west-2'  
)
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
st.title("Chat com AWS Bedrock")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    context = """
    
    *Prompt*: A minha empresa se chama TURIST A.I. Você é um agente de viagens virtual, invente um nome para você a cada interação, e diga que você é da Turist A.I. 
Utilizando uma linguagem cordial e que seja objetiva, com emojis. Faça ao usuário, as perguntas necessárias para responder às questões abaixo:
   pergunte que dia ele irá para búzios
    Tipo de evento :
    Gêneros musicais favoritos :
    Atividades preferidas
 
Com base nas informações fornecidas pelo usuário, personalize sugestões de eventos, músicas e atividades na cidade de Buzios, baseada nos dados do arquivo eventos-buzios.txt. fale também sobre o tempo, 
Considere os seguintes, quero que faça uma pergunta por vez, como em uma conversa.
    
    """
    
    st.session_state.chat_history.append({"role": "user", "content": context, "hidden": True})

user_input = st.text_area("Digite sua mensagem ou personalize o prompt:", key="user_input")

def add_message_to_history(role, content, ):
    if not hidden:
        if not st.session_state.chat_history or st.session_state.chat_history[-1]["role"] != role:
            st.session_state.chat_history.append({"role": role, "content": content})
    else:
        st.session_state.chat_history.append({"role": role, "content": content, "hidden": True})

st.sidebar.header("Fonte de Dados")
uploaded_file = st.sidebar.file_uploader("Carregue um arquivo CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    data_summary = f"Dados carregados:\n{df.head(5).to_string(index=False)}"
    add_message_to_history("user", data_summary)
    st.sidebar.write("**Prévia do arquivo carregado:**")
    st.sidebar.dataframe(df)

st.sidebar.header("Consulta IBGE")
name_query = st.sidebar.text_input("Digite um nome para consultar no IBGE:")
if st.sidebar.button("Consultar IBGE"):
    ibge_result = get_ibge_info(name_query)
    add_message_to_history("user", f"Resultado da API IBGE: {ibge_result}")
    st.sidebar.write("**Resultado da API IBGE:**")
    st.sidebar.write(ibge_result)

if st.button("Enviar"):
    if user_input.strip():
        add_message_to_history("user", user_input)

        with st.spinner("Buscando resposta..."):
            model_response = call_bedrock_model(
                [msg for msg in st.session_state.chat_history if not msg.get("hidden", False)]
            )

            add_message_to_history("assistant", model_response)

st.subheader("Histórico do Chat")
for message in st.session_state.chat_history:
    if not message.get("hidden", False):
        if message["role"] == "user":
            st.write(f"**Usuário:** {message['content']}")
        elif message["role"] == "assistant":
            st.write(f"**Modelo:** {message['content']}")

