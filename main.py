from langchain_ollama import ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
from langgraph.graph import END, MessageGraph
from langchain_core.messages import BaseMessage, HumanMessage
from typing import List, Sequence
from chains import generate_chain, reflect_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: Sequence[BaseMessage]):
    res = generate_chain.invoke({"messages": state})
    return res

def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]

builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE) ## Start with this node

def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue) # Condition
builder.add_edge(REFLECT, GENERATE) # When REFLECT is executed GENERATE is the next

graph = builder.compile()

if __name__ == "__main__":
    load_dotenv()
    inputs = HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)
    response = graph.invoke(inputs)
    # print(response)