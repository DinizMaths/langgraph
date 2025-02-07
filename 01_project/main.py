from typing import List, Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import reflection_chain, generation_chain

from dotenv import load_dotenv


load_dotenv()


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    return generation_chain.invoke({"messages": state})

def reflection_node(state: Sequence[BaseMessage]):
    res = reflection_chain.invoke({"messages": state})

    return [HumanMessage(content=res.content)]

builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

if __name__ == "__main__":
    inputs = HumanMessage(content="""
    Maje this tweet better:
    @LangChainAI
    -  newly Tool Calling feature is seriouly underrated.
                          
    After a long wait, it's here- making implementations o agents across different modelos with function calling""")

    response = graph.invoke(inputs)

    print(response)
