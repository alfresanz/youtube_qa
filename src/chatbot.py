from typing import List, Optional
from llama_index.core import QueryBundle, StorageContext, load_indices_from_storage, get_response_synthesizer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeRelationship, NodeWithScore, TextNode
from llama_index.core.storage.docstore import BaseDocumentStore
from llama_index.core.bridge.pydantic import Field
from llama_index.core.agent import AgentRunner
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


# from typing import Annotated
# from pydantic import BaseModel

# from langgraph.graph.message import add_messages
# from langgraph.graph import StateGraph
# from langgraph.checkpoint.memory import MemorySaver


# def build_chatbot(llm):
    
#     class State(BaseModel):
#         messages: Annotated[list, add_messages]

#     graph_builder = StateGraph(State)

#     def chatbot_node(state: State):
#         return {"messages": [llm.invoke(state.messages)]}

#     graph_builder.add_node("chatbot", chatbot_node)

#     graph_builder.set_entry_point("chatbot")
#     graph_builder.set_finish_point("chatbot")

#     memory = MemorySaver()
#     return graph_builder.compile(checkpointer=memory)

# def chatbot_response(user_query: str):
#     for msg, metadata in st.session_state['chatbot'].stream(
#         {"messages": [("user", user_query)]},
#         config = DEFAULT_THREAD,
#         stream_mode="messages" # For some reason you have to use the NOT DOCUMENTED stream_mode="messages" argument
#         ): 
#         yield msg.content


def load_index(data_dir: str):
    storage_context = StorageContext.from_defaults(persist_dir="./data")
    indices = load_indices_from_storage(storage_context)
    return indices[0]


class Timestamp:
    def __init__(self, time_str: str):
        self.hours, self.minutes, self.seconds = map(int, time_str.split(':'))

    def __lt__(self, other):
        return (self.hours, self.minutes, self.seconds) < (other.hours, other.minutes, other.seconds)

    def __le__(self, other):
        return (self.hours, self.minutes, self.seconds) <= (other.hours, other.minutes, other.seconds)

    def __eq__(self, other):
        return (self.hours, self.minutes, self.seconds) == (other.hours, other.minutes, other.seconds)

    def __ne__(self, other):
        return (self.hours, self.minutes, self.seconds) != (other.hours, other.minutes, other.seconds)

    def __gt__(self, other):
        return (self.hours, self.minutes, self.seconds) > (other.hours, other.minutes, other.seconds)

    def __ge__(self, other):
        return (self.hours, self.minutes, self.seconds) >= (other.hours, other.minutes, other.seconds)
    
    def to_seconds(self):
        return self.hours * 3600 + self.minutes * 60 + self.seconds


class ContextInflator(BaseNodePostprocessor):
    
    docstore: BaseDocumentStore
    num_backward_nodes: int = Field(default=1)
    num_forward_nodes: int = Field(default=1)
    timestamp_keys: list[str] = Field(default=['start_timestamp', 'end_timestamp'])
    proximity_merge: int = Field(default=0) # seconds

    def get_previous_node(self, node):
        prev_node_ref = node.relationships.get(NodeRelationship.PREVIOUS)
        
        if prev_node_ref is None:
            return None
        
        return self.docstore.get_node(prev_node_ref.node_id)

    def get_next_node(self, node):
        next_node_ref = node.relationships.get(NodeRelationship.NEXT)
        
        if next_node_ref is None:
            return None
        
        return self.docstore.get_node(next_node_ref.node_id)

    def _postprocess_nodes(
            self, 
            nodes: List[NodeWithScore], 
            query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        
        # Find all unique parent nodes
        parent_nodes = [] # Non-repeating list of parent nodes' ids
        for node in nodes:
            parent_node = node.node.relationships.get(NodeRelationship.PARENT)
            if parent_node is not None and parent_node.node_id not in parent_nodes:
                parent_nodes.append(parent_node.node_id)

        # Create a nodes containing the context for the children nodes
        context_nodes = []
        for parent_node in parent_nodes:
            child_nodes = [node for node in nodes if node.node.relationships.get(NodeRelationship.PARENT).node_id == parent_node]
            child_nodes.sort(key=lambda x: Timestamp(x.node.metadata[self.timestamp_keys[0]]))
            
            # Jump a series of nodes from each child node to find its context
            context_snippets = [] # List of tuples of 2 nodes defining a context snippet, ordered in time
            for node in child_nodes:

                # Get node num_backward_nodes steps before
                start_node = node.node
                for i in range(self.num_backward_nodes):
                    prev_node = self.get_previous_node(start_node)
                    if prev_node is None:
                        break
                    start_node = prev_node

                # Get node num_forward_nodes steps after
                end_node = node.node
                for i in range(self.num_forward_nodes):
                    next_node = self.get_next_node(end_node)
                    if next_node is None:
                        break
                    end_node = next_node

                context_snippets.append((start_node, end_node))
            
            # Find where is it worth merging the context of two child nodes
            merged_snippets = [] # List of merged snippets due to overlap or proximity
            previous_snippet = None
            for snippet in context_snippets:
                if previous_snippet is None:
                    previous_snippet = snippet
                    continue
                
                # Overlapping merge
                if Timestamp(snippet[0].metadata[self.timestamp_keys[0]]) <= Timestamp(previous_snippet[1].metadata[self.timestamp_keys[1]]):
                    previous_snippet = (previous_snippet[0], snippet[1])

                # Proximity merge
                elif Timestamp(snippet[0].metadata[self.timestamp_keys[0]]).to_seconds() - Timestamp(previous_snippet[1].metadata[self.timestamp_keys[1]]).to_seconds() <= self.proximity_merge:
                    previous_snippet = (previous_snippet[0], snippet[1])

                else:
                    merged_snippets.append(previous_snippet)
                    previous_snippet = snippet

            merged_snippets.append(previous_snippet)

            # Extract actual text for each snippet as well as the timestamps
            snippet_texts = []
            snippet_timestamps = []
            source_text = self.docstore.get_node(parent_node).text
            for snippet in merged_snippets:
                start_text = snippet[0].text.strip()
                end_text = snippet[1].text.strip()

                start_index = source_text.find(start_text)
                end_index = source_text.find(end_text) + len(end_text)

                if start_index == -1 or end_index == -1:
                    print('Warning: Text not found in parent node.')

                cropped_text = source_text[start_index:end_index]

                snippet_texts.append(cropped_text)
                snippet_timestamps.append((snippet[0].metadata[self.timestamp_keys[0]], snippet[1].metadata[self.timestamp_keys[1]]))

            # Create composed text
            _ = [(f'[{timestamp[0]}]\n' + text + f'\n[{timestamp[1]}]\n') for text, timestamp in zip(snippet_texts, snippet_timestamps)]
            composed_text = '(...)\n'.join(_)

            # Create context node
            text_node = TextNode(
                text=composed_text,
                text_template='{content}\n\nRelevant quotes\' timestamps:\n',
            )

            context_node = NodeWithScore(node=text_node)
            context_nodes.append(context_node)

        return context_nodes + nodes

def get_chat_engine(
        index, llm, 
        top_k: int = 2, 
        back_context: int = 10, 
        forw_context: int = 20, 
        response_mode: str = 'compact',  
        proximity_merge: int = 15):

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
        node_postprocessors=[
            ContextInflator(
                docstore=index.docstore,
                num_backward_nodes=back_context,
                num_forward_nodes=forw_context,
                proximity_merge=proximity_merge
            )
        ],
    )

    query_engine_tool = QueryEngineTool.from_defaults(query_engine=query_engine)

    chat_engine = AgentRunner.from_llm(
        tools=[query_engine_tool],
        llm=llm
    )

    return chat_engine