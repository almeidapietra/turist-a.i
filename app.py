import streamlit as st
import boto3
import json
import pandas as pd
import requests

session = boto3.Session(profile_name="iaedn")
client = session.client('bedrock-runtime', region_name='us-west-2')

# =========================
# Função para chamada à API pública do IBGE
# =========================
def get_ibge_info(name):
    try:
        response = requests.get(f"https://servicodados.ibge.gov.br/api/v2/censos/nomes/{name}")
        st.write("**Retorno Bruto da API IBGE:**")
        st.json(response.json())

        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list) and "res" in data[0]:
                name_data = data[0]
                resumo = f"Nome: {name_data['nome']}\nLocalidade: {name_data['localidade']}\n"
                resumo += "Frequências por período:\n"
                for periodo in name_data["res"]:
                    resumo += f"Período {periodo['periodo']}: {periodo['frequencia']} ocorrências\n"
                return resumo
            else:
                return f"Resultado para {name}: Nenhuma informação encontrada ou formato inesperado."
        else:
            return f"Erro ao acessar a API do IBGE: {response.status_code}"
    except Exception as e:
        return f"Erro ao acessar a API do IBGE: {str(e)}"

# =========================
# Função para chamada ao AWS Bedrock
# =========================
def call_bedrock_model(messages):
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
    Bem-vindo! Você está interagindo com o assistente da empresa SuperNanny.
    Somos líderes em atendimento por babás.
    Posso ajudá-lo com informações sobre nossos produtos, serviços ou outras dúvidas.

    *Prompt*: Para bebês até 3 meses - 100,00 a hora, para bebês até 6 meses - 150 reais a hora e a partir de 1 ano - 200,00 a hora. Faça perguntas para o usuário final.
    Como se fosse um cadastro final, induza a venda, mas tire dúvidas. Você só atende cidades de Minas Gerais e vai cobrar 1,00 a cada km de Belo Horizonte de deslocamen.
    Pergunte o nome da pessoa, idade, quantos filhos, etc...

    Faça pergunta por pergunta, não envie muitas perguntas. Seja cordial, envie emojis fofos (infantis). 
    
    """
    st.session_state.chat_history.append({"role": "user", "content": context})

user_input = st.text_area("Digite sua mensagem ou personalize o prompt:", key="user_input")

def add_message_to_history(role, content, hidden=False):
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

