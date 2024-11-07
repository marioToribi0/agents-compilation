from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc."
            "You will receive a conversation, however always provide a recommendation throught the last tweet"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing exellecnt twitter posts."
            "Generate the best twitter post possible for the user's request."
            "You can customize your tweet with emojis or others."
            "If the user provides critique, respond with a revised version of your previous attemps."
            "Only respond with the result tweet"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

llm = ChatOllama(model="llama3.1", base_url="http://192.168.18.10:11434/")
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm