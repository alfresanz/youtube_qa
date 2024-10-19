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