from decouple import config
from dotenv import load_dotenv

load_dotenv()

BASE_URL_OLLAMA = config("BASE_URL_OLLAMA")
TAVILY_API_KEY = config("TAVILY_API_KEY")
LANGCHAIN_TRACING_V2 = config("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = config("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = config("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = config("LANGCHAIN_PROJECT")
GROQ_API_KEY = config("GROQ_API_KEY")
TAVILY_API_KEY = config("TAVILY_API_KEY")