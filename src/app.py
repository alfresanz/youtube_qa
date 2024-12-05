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
MODELS = ["gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview"]
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

def render_chat(debug_mode=False):
    for message in st.session_state["chat_history"]:
    
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        
        elif message["role"] == "assistant":
            with st.chat_message("ai"):
                st.write(message["content"])
        
        elif message["role"] == "tool" and debug_mode:
            with st.chat_message("tool"):
                with st.expander("Retrieved nodes"):
                    st.write(message["content"])

        # elif message.role == "assistant":

        #     if message.content is not None and message.content != "":
            
        #         if "tool_calls" in message.additional_kwargs:
        #             tool_calls_dicts = message.additional_kwargs["tool_calls"]

        #             with st.chat_message("tool"):
        #                 with st.expander("Retrieved nodes"):
        #                     for t, tool_call in enumerate(tool_calls_dicts):
        #                         st.write(f"Tool call {t+1}: {tool_call['input']}")
        #                         st.write("\n\n".join([str(n) for n in tool_call["retrieved_nodes"]]))

        #         with st.chat_message("ai"):
        #             st.write(message.content)

        # elif message.role == "tool" and debug_mode:

        #     node_dicts = message.additional_kwargs.get("retrieved_nodes")

        #     if node_dicts is not None:
        #         with st.chat_message("tool"):
        #             with st.expander("Retrieved nodes"):
        #                 st.write("\n\n".join([str(n) for n in node_dicts]))


# [ST] Initialize config in session state
if "debug_mode" not in st.session_state:
    st.session_state["debug_mode"] = False

# [ST] Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

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
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    llm = OpenAI(temperature=temperature, model=model)

    response_mode = st.selectbox("Response mode", RESPONSE_MODES, index=0)
    top_k = st.slider("Top k", min_value=1, max_value=30, value=4, step=1)

    chat_mode = st.selectbox("Chat mode", CHAT_MODES, index=0)

    if st.button("Apply + Restart"):
        st.session_state["chatbot"] = st.session_state["index"].as_chat_engine(chat_mode=chat_mode, llm=llm, response_mode=response_mode, similarity_top_k=top_k)

    st.subheader("Verbosity settings")

    st.session_state["debug_mode"] = st.checkbox("Show retrieved nodes", value=False)


# [ST] Initialize chatbot in session state
if "chatbot" not in st.session_state:
    st.session_state["chatbot"] = st.session_state["index"].as_chat_engine(chat_mode=chat_mode, llm=llm, response_mode=response_mode, similarity_top_k=top_k)


# [ST] Write previous chat messages
render_chat(debug_mode=st.session_state["debug_mode"])          


# [ST] Get user input, answer, and write to chat
user_query = st.chat_input("¿En qué puedo ayudarte?")

if user_query is not None and user_query != "":

    messages = [{"role": "user", "content": user_query}]

    with st.chat_message("user"):
        st.markdown(user_query)

    response = st.session_state["chatbot"].stream_chat(user_query)

    if len(response.source_nodes) > 0:

        tool_call_msg = None
        for message in st.session_state["chatbot"].chat_history[::-1]:
            if message.role == "assistant" and "tool_calls" in message.additional_kwargs:
                tool_call_msg = message
                break

        tool_calls_dicts = []
        for t, tool_call in enumerate(tool_call_msg.additional_kwargs["tool_calls"]):
            node_dicts = [{"score": n.score, "text": n.text, "metadata": n.metadata} for n in response.source_nodes[t*top_k:(t+1)*top_k]]
            tool_input = tool_call.function.arguments
            tool_calls_dicts.append({"input": tool_input, "retrieved_nodes": node_dicts})

        tool_calls_str = ""
        for t, tool_call in enumerate(tool_calls_dicts):
            tool_calls_str += f"Tool call {t+1}: {tool_call['input']}\n\n"
            tool_calls_str += "\n\n".join([str(n) for n in tool_call["retrieved_nodes"]])
            tool_calls_str += "\n\n\n\n\n"
        tool_calls_str = tool_calls_str.strip()

        messages.append({"role": "tool", "content": tool_calls_str})

        if st.session_state["debug_mode"]:
            with st.chat_message("tool"):
                with st.expander("Retrieved nodes"):
                    st.write(tool_calls_str)

    with st.chat_message("ai"):
        st.write_stream(response.response_gen)
    
    messages.append({"role": "assistant", "content": response.response})

    st.session_state["chat_history"].extend(messages)