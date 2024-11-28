from pathlib import Path

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex

from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from typing import Annotated
from pydantic import BaseModel

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

def build_chatbot(llm):
    
    class State(BaseModel):
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)

    def chatbot_node(state: State):
        return {"messages": [llm.invoke(state.messages)]}

    graph_builder.add_node("chatbot", chatbot_node)

    graph_builder.set_entry_point("chatbot")
    graph_builder.set_finish_point("chatbot")

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)

def load_chroma_index(data_dir: str, collection_name: str):

    db = chromadb.PersistentClient(path=data_dir)
    chroma_collection = db.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

def get_chat_engine(index, top_k: int, llm, response_mode: str, chat_mode: str):

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
        llm=llm,
    )

    response_synthesizer = get_response_synthesizer(
        response_mode=response_mode,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[],
    )

    return index.as_chat_engine(query_engine=query_engine, chat_mode=chat_mode)