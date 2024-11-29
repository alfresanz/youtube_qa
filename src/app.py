import streamlit as st

# from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
# from langchain_openai import ChatOpenAI

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from chatbot import build_chatbot
from chatbot import load_chroma_index

from pytubefix import YouTube

from pathlib import Path
import os


DATA_DIR = Path("data")

DEFAULT_THREAD = {"configurable": {"thread_id": "default"}} # For the chatbot. Each thread is a different conversation

CHROMA_DB_PATH = DATA_DIR / "chroma"
CHROMA_COLLECTION_NAME = "libre-a-los-30-videos"

CHAT_MODES = ["best", "condense_question", "context", "condense_plus_context"]
MODELS = ["gpt-4o-mini", "gpt-4o", "o1-mini", "o1-preview"]
RESPONSE_MODES = ["compact", "refine", "tree_summarize", "accumulate", "compact_accumulate"]

# TRANSCRIPT_PATHS = [
#     DATA_DIR / "mZvnATAawpM.txt",
#     DATA_DIR / "5NUGCNhfZ4I.txt",
# ]

# VIDEO_TITLES = [
#     "Cómo Comprar Pisos sin poner Dinero - Un Inmueble por cada Día del Mes - Alex Emebe Ep:16",
#     "Libre a los 30 - Javier Medina I Podcast #50"
# ]

# DEFAULT_MODEL = "gpt-4o-mini"
# DEFAULT_PROMPT = "You are a useful agent that has access to the community knowledge"


# OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm = OpenAI(temperature=0.1, model="gpt-4")


# [ST] App basics
st.set_page_config(page_title="LibreBot", page_icon="🤖")
st.title("Accede a nuestro conocimiento acumulado")


# Initialize chatbot funtion
# def initialize_chatbot(system_prompt, model):
    
#     llm = ChatOpenAI(model=model)

#     st.session_state['chatbot'] = build_chatbot(llm)

#     transcript = ""
#     for path, title in zip(TRANSCRIPT_PATHS, VIDEO_TITLES):
#         if path.exists():
#             with open(path, "r", encoding = "ISO-8859-1") as f:
#                 transcript += f"Title: {title}\n"
#                 transcript += f.read()
#                 transcript += "\n\n"

#     full_prompt = f"""{system_prompt}\n
#     {transcript}"""

#     st.session_state['chatbot'].update_state(DEFAULT_THREAD, {"messages": [("system", full_prompt)]})

# Chatbot response stream helper function
# def chatbot_response(user_query: str):
#     for msg, metadata in st.session_state['chatbot'].stream(
#         {"messages": [("user", user_query)]},
#         config = DEFAULT_THREAD,
#         stream_mode="messages" # For some reason you have to use the NOT DOCUMENTED stream_mode="messages" argument
#         ): 
#         yield msg.content


# [ST] Initialize index in session state
if "index" not in st.session_state:
    st.session_state["index"] = load_chroma_index(data_dir=str(CHROMA_DB_PATH), collection_name=CHROMA_COLLECTION_NAME)


# [ST] Sidebar
with st.sidebar:
    st.title("Settings")

    # st.subheader("Prompt settings")
    # st.text_area("System prompt", value=DEFAULT_PROMPT, key="system_prompt", height=200)
    # st.write("Note that the video transcripts are supplied after this prompt.")

    st.subheader("Chatbot settings")

    model = st.selectbox("Model", MODELS, index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    llm = OpenAI(temperature=temperature, model=model)

    response_mode = st.selectbox("Response mode", RESPONSE_MODES, index=0)
    top_k = st.slider("Top k", min_value=1, max_value=30, value=4, step=1)

    chat_mode = st.selectbox("Chat mode", CHAT_MODES, index=0)

    if st.button("Apply + Restart"):
        st.session_state["chatbot"] = st.session_state["index"].as_chat_engine(chat_mode=chat_mode, llm=llm, response_mode=response_mode, similarity_top_k=top_k)

    st.subheader("Verbosity settings")

    debug_mode = st.checkbox("Debug mode", value=False)


# [ST] Initialize chatbot in session state
if "chatbot" not in st.session_state:
    st.session_state["chatbot"] = st.session_state["index"].as_chat_engine(chat_mode=chat_mode, llm=llm, response_mode=response_mode, similarity_top_k=top_k)


# [ST] Write previous chat messages
for message in st.session_state["chatbot"].chat_history:
    
    if message.role == "user":
        with st.chat_message("user"):
            st.write(message.content)

    if message.role == "assistant":

        if message.content is not None:
            with st.chat_message("ai"):
                st.write(message.content)            


# [ST] Get user input, answer, and write to chat
user_query = st.chat_input("¿En qué puedo ayudarte?")

if user_query is not None and user_query != "":

    with st.chat_message("user"):
        st.markdown(user_query)

    response = st.session_state["chatbot"].stream_chat(user_query)

    if debug_mode and len(response.source_nodes) > 0:

        node_dicts = [{"score": n.score, "text": n.text, "metadata": n.metadata} for n in response.source_nodes]

        with st.chat_message("tool"):
            st.write("\n\n".join([str(n) for n in node_dicts]))

    with st.chat_message("ai"):
        st.write_stream(response.response_gen)

