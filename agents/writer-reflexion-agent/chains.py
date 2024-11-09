import sys
sys.path.append("./")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llm_models import ChatModels
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser, PydanticToolsParser
from langchain.output_parsers import PydanticOutputParser
from schemas import AnswerQuestion, ReviseAnswer
import datetime
from langchain_core.messages import BaseMessage, HumanMessage

# llm = ChatModels.OLLAMA("llama3.1")
# llm = ChatModels.GROQ("llama-3.2-90b-text-preview")
llm = ChatModels.OPENAI("gpt-4o-mini")

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format. {format_instructions}"),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat()
)

pydantic_first_responder = PydanticOutputParser(pydantic_object=AnswerQuestion)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer",
    format_instructions = pydantic_first_responder.get_format_instructions()
)

first_responder = (
    first_responder_prompt_template
    | llm
)

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
        - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 word
"""

pydantic_revisor = PydanticOutputParser(pydantic_object=ReviseAnswer)

revisor_prompt = actor_prompt_template.partial(
    first_instruction=revise_instructions,
    format_instructions=pydantic_revisor.get_format_instructions()
)

revisor = (
        revisor_prompt
        | llm
)

if __name__ == "__main__":
    human_messsage = HumanMessage(
        content="Write about about AI-Powered SOC / autonomous soc problems domain,"
        " list startups that do that and raised capital."
    )
    
    res = first_responder.invoke(input={"messages": [human_messsage]})
    # res = pydantic_first_responder.invoke(res)
    # print(f"{res.answer=}")
    # print(f"{res.search_queries=}")
    # res = 