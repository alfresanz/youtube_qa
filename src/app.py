import streamlit as st

from pathlib import Path
import os

from chatbot import load_index, get_chat_engine


DATA_DIR = Path("data")

DEFAULT_THREAD = {"configurable": {"thread_id": "default"}} # For the chatbot. Each thread is a different conversation

MODELS = ["gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview"]


# OpenAI API key
from llama_index.llms.openai import OpenAI
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


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


# [ST] App basics
st.set_page_config(page_title="LibreBot", page_icon="🤖")
st.title("Accede a nuestro conocimiento acumulado")

# [ST] Initialize config in session state
if "debug_mode" not in st.session_state:
    st.session_state["debug_mode"] = False

# [ST] Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# [ST] Initialize index in session state
if "index" not in st.session_state:
    st.session_state["index"] = load_index(data_dir=str(DATA_DIR))


# [ST] Sidebar
with st.sidebar:
    st.title("Settings")

    st.subheader("Chatbot settings")

    model = st.selectbox("Model", MODELS, index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    llm = OpenAI(temperature=temperature, model=model)

    top_k = st.slider("Retrieve top K", min_value=1, max_value=30, value=4, step=1)
    back_context = st.slider("Backward context", min_value=0, max_value=100, value=10, step=1)
    forw_context = st.slider("Forward context", min_value=0, max_value=100, value=20, step=1)

    if st.button("Apply + Restart"):
        st.session_state["chatbot"] = get_chat_engine(st.session_state["index"], llm, top_k, back_context, forw_context)

    st.subheader("Verbosity settings")

    st.session_state["debug_mode"] = st.checkbox("Show retrieved nodes", value=False)


# [ST] Initialize chatbot in session state
if "chatbot" not in st.session_state:
    st.session_state["chatbot"] = get_chat_engine(st.session_state["index"], llm, top_k, back_context, forw_context)


# [ST] Write previous chat messages
render_chat(debug_mode=st.session_state["debug_mode"])          

# [ST] Get user input, answer, and write to chat
user_query = st.chat_input("¿En qué puedo ayudarte?")

if user_query is not None and user_query != "":

    # User
    new_messages = []
    new_messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    response = st.session_state["chatbot"].stream_chat(user_query)

    # Tool
    for tool_call in response.sources:
        
        node_dicts = [{"score": source_node.score, 
                       "text": source_node.text, 
                       "metadata": source_node.metadata} 

                       for source_node in tool_call.raw_output.source_nodes]
        
        tool_call_str = ""
        tool_call_str += f"Tool call: {tool_call.raw_input}\n\n"
        tool_call_str += "\n\n".join([str(n) for n in node_dicts])
        tool_call_str += "\n\n\n\n\n"

        new_messages.append({"role": "tool", "content": tool_call_str})

    if st.session_state["debug_mode"]:
        for message in new_messages:
            if message["role"] == "tool":
                with st.chat_message("tool"):
                    with st.expander("Retrieved nodes"):
                        st.write(message["content"])
        
    # QA LLM
    with st.chat_message("ai"):
        st.write_stream(response.response_gen)
    
    new_messages.append({"role": "assistant", "content": response.response})

    st.session_state["chat_history"].extend(new_messages)