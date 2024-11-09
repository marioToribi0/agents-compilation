from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chains import revisor, first_responder, pydantic_first_responder, pydantic_revisor
from tool_executor import execute_tools

from dotenv import load_dotenv

load_dotenv()

MAX_ITERATIONS = 4
DRAFT = "draft"
EXECUTE_TOOLS = "execute_tools"
REVISE = "revise"

builder = MessageGraph()

builder.add_node(DRAFT, first_responder)
builder.add_node(EXECUTE_TOOLS, execute_tools)
builder.add_node(REVISE, revisor)

builder.add_edge(DRAFT, EXECUTE_TOOLS)
builder.add_edge(EXECUTE_TOOLS, REVISE)

def event_loop(state: List[BaseMessage]):
    if len(state)>2+MAX_ITERATIONS:
        return END
    return EXECUTE_TOOLS

builder.add_conditional_edges(REVISE, event_loop, [END, EXECUTE_TOOLS])
builder.set_entry_point(DRAFT)

graph = builder.compile()

# print(graph.get_graph().draw_ascii())
# graph.get_graph().draw_mermaid_png(output_file_path="writer.png")

if __name__ == "__main__":
    print("Hello Reflexion")
    
    res = graph.invoke(
        "Write about a comparison between nemotron, GPT and Claude as LLM. Is Nemotron a good option to start to buy GPU instead pay tokens?"
    )
    
    print("\nArticle: \n")
    print(pydantic_revisor.invoke(res[-1]).answer)
    
    print("\nReferences: \n")
    print(pydantic_revisor.invoke(res[-1]).references)