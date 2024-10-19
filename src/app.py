import streamlit as st

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from chatbot import build_chatbot

from pytubefix import YouTube

from pathlib import Path


DATA_DIR = Path("data")

DEFAULT_THREAD = {"configurable": {"thread_id": "default"}} # For the chatbot. Each thread is a different conversation

TRANSCRIPT_PATHS = [
    DATA_DIR / "mZvnATAawpM.txt",
    DATA_DIR / "5NUGCNhfZ4I.txt",
]

VIDEO_TITLES = [
    "Cómo Comprar Pisos sin poner Dinero - Un Inmueble por cada Día del Mes - Alex Emebe Ep:16",
    "Libre a los 30 - Javier Medina I Podcast #50"
]

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_PROMPT = "Use the content of the following videos to answer any questions. Limit yourself to what you can answer with the following content, nothing else."


# OpenAI API key
load_dotenv(dotenv_path='conf/.env')


# [ST] App basics
st.set_page_config(page_title="YoutubeQA chatbot", page_icon="🤖")
st.title("Chat with videos!")


# Initialize chatbot funtion
def initialize_chatbot(system_prompt, model):
    llm = ChatOpenAI(model=model)

    st.session_state['chatbot'] = build_chatbot(llm)

    transcript = ""
    for path, title in zip(TRANSCRIPT_PATHS, VIDEO_TITLES):
        if path.exists():
            with open(path, "r") as f:
                transcript += f"Title: {title}\n"
                transcript += f.read()
                transcript += "\n\n"

    full_prompt = f"""{system_prompt}\n
    {transcript}"""

    st.session_state['chatbot'].update_state(DEFAULT_THREAD, {"messages": [("system", full_prompt)]})

# Chatbot response stream helper function
def chatbot_response(user_query: str):
    for msg, metadata in st.session_state['chatbot'].stream(
        {"messages": [("user", user_query)]},
        config = DEFAULT_THREAD,
        stream_mode="messages" # For some reason you have to use the NOT DOCUMENTED stream_mode="messages" argument
        ): 
        yield msg.content


# [ST] Initialize session state
if 'chatbot' not in st.session_state:
    initialize_chatbot(system_prompt=DEFAULT_PROMPT, model=DEFAULT_MODEL)

# if 'transcript' not in st.session_state:
#     st.session_state['transcript'] = "There is no transcript loaded yet."


# [ST] Sidebar
with st.sidebar:
    st.title("Settings")

    st.subheader("Prompt settings")
    st.text_area("System prompt", value=DEFAULT_PROMPT, key="system_prompt", height=200)
    st.write("Note that the video transcripts are supplied after this prompt.")

    st.subheader("Model settings")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "o1-mini", "o1-preview"], index=0)

    if st.button("Restart chatbot"):
        with st.spinner("Restarting chatbot..."):
            initialize_chatbot(st.session_state['system_prompt'])


# [ST] Write previous chat messages
for message in st.session_state['chatbot'].get_state(config=DEFAULT_THREAD).values["messages"]:
    
    if message.type == "human":
        with st.chat_message("user"):
            st.write(message.content)

    elif message.type == "ai":
        with st.chat_message("ai"):
            st.write(message.content)

# [ST] Get user input, answer, and write to chat
user_query = st.chat_input("Ask anything!")

if user_query is not None and user_query != "":

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("ai"):
        bot_response = st.write_stream(chatbot_response(user_query))