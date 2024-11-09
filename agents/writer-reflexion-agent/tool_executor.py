from collections import defaultdict
import json
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, FunctionMessage
from typing import List
from schemas import AnswerQuestion, ReviseAnswer, Reflection
from chains import pydantic_first_responder
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from uuid import uuid4
from langchain_community.tools import TavilySearchResults

tavily_search = TavilySearchResults()
duckduckgo_search = DuckDuckGoSearchResults(output_format="list")
tool_executor = ToolExecutor([tavily_search])

def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    tool_invocation: AIMessage = state[-1]
    answer: AnswerQuestion = pydantic_first_responder.invoke(tool_invocation)
    
    queries = answer.search_queries
    
    main_id = f"call_{str(uuid4()).replace('-','')[:10]}"
    main_id = "call_JM4a1T6KTqTOstpMwsZ8bAh8"
    tool_invocation = []
    ids = [main_id for _ in queries]
    
    for query in queries:
        tool_invocation.append(
            ToolInvocation(
                tool="tavily_search_results_json",
                tool_input=query
            )
        )
    outputs = tool_executor.batch(tool_invocation)
    
    # Map each output to its corresponding ID and tool input
    outputs_map = defaultdict(dict)
    for id_, output, invocation in zip(ids, outputs, tool_invocation):
        outputs_map[id_][invocation.tool_input] = output
        
    # Convert the mapped outputs to ToolMessage objects
    tools_messages = []
    for id_, mapped_output in outputs_map.items():
        tools_messages.append(AIMessage(content=json.dumps(mapped_output)))
        
    return tools_messages
    
if __name__ == "__main__":
    print("Tool Executor Enter")
    
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain,"
        " list startups that do that and raised capital."
    )
    
    answer = AnswerQuestion(
        answer="",
        reflection=Reflection(missing="", superflous=""),
        search_queries=[
            "AI-powered SOC startups funding",
            "AI SOC problem domain specifics",
            "Technologies used by AI-powered SOC startups",
        ],
        id="call_afdakfnadSFHKFSFDA"
    )
    
    raw_res = execute_tools(
        state=[
            human_message,
            AIMessage(
                content="""{
  "answer": "AI-powered Security Operations Centers (SOCs) leverage artificial intelligence and machine learning to enhance threat detection, incident response, and overall security management. They address several challenges in cybersecurity, such as the overwhelming volume of alerts, the shortage of skilled security professionals, and the need for rapid response to threats. Autonomous SOCs aim to reduce human intervention by automating repetitive tasks, allowing analysts to focus on high-priority incidents. However, these systems struggle with false positives, the complexity of integrating AI with existing security tools, and the challenge of ensuring accurate threat intelligence. Startups in this space have gained attention and funding to tackle these issues. Notable examples include:\n\n1. **Snyk** - Raised over $650 million, focusing on securing cloud-native applications.\n2. **Darktrace** - A leader in AI-driven cybersecurity, raised $230 million, known for its self-learning technology.\n3. **Cymulate** - Raised $70 million to provide continuous security validation.\n4. **XDR** providers like **SentinelOne**, which raised $267 million, emphasize automated detection and response across various vectors.\n5. **Chronicle** (now part of Google Cloud) focuses on threat detection and response using AI to analyze large datasets.\n\nThese startups exemplify the shift towards AI-enhanced SOCs, addressing critical cybersecurity needs while highlighting the ongoing challenges within the sector.",
  "reflection": {
    "missing": "The answer lacks a thorough analysis of specific technological challenges faced by AI-powered SOCs, such as data privacy issues and the limitations of current AI models in understanding context. Additionally, it could benefit from including more examples of recent funding rounds and specific features of the startups mentioned.",
    "superflous": "The mention of large funding amounts could be seen as excessive without context on what those funds are specifically being used for within the startups. A more concise mention of these figures would maintain focus on the startups' relevance to AI-powered SOC challenges."
  },
  "search_queries": [
    "AI-powered SOC challenges 2024",
    "autonomous SOC startups funding",
    "recent investment in AI cybersecurity companies"
  ]
}"""
            )
        ]
    )
    
    print(raw_res)
    