from dotenv import load_dotenv
from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import (
    END, # When the graph reach the node with this key, stops the execution
    MessageGraph # A type of graph that it state is a simple sequence of messages
    # class MessageGraph(StateGraph):
    #     """A StateGraph where every node 
    #     receives a sequence of messages 
    #     and returns one or more messages 
    #     as output."""

    #     def __init__(self) -> None:
    #         super().__init__(Annotated[list[AnyMessage], add_messages])
)
from chains import generation_chain, reflection_chain

load_dotenv()

# Keys used to identify the nodes in the graph
# reflect node will run the reflection_chain
# generate node will run the generation_chain
REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    return generation_chain.invoke({"messages": state})

def reflection_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
    response = reflection_chain.invoke({"messages": state})
    return [HumanMessage(content=response.content)] # Trick the LLM to think a human is sending the message

builder = MessageGraph()

builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)

def should_continue(state: List[BaseMessage]):
    return END if len(state) > 6 else REFLECT

builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

graph.get_graph().print_ascii()


if __name__ == "__main__":
    print("--- Hello LangGraph ---")

    inputs = HumanMessage(content="""Make this tweet better:
                          @LangChainAI
        - newly Tool Calling feature is seriously underrated.
                          
        After a long wait, it's here! 🎉
    """)

    response = graph.invoke(inputs)
