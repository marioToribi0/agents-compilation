from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from config import BASE_URL_OLLAMA, GROQ_API_KEY
from dotenv import load_dotenv

class ChatModels:
    OLLAMA: BaseChatModel = lambda model: ChatOllama(model=model, base_url=BASE_URL_OLLAMA)
    OPENAI: BaseChatModel = lambda model: ChatOpenAI(model=model)
    GROQ: BaseChatModel = lambda model: ChatGroq(model=model, api_key=GROQ_API_KEY)