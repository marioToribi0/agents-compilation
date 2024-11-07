from decouple import config
from dotenv import load_dotenv

load_dotenv()

BASE_URL_OLLAMA = config("BASE_URL_OLLAMA")
LANGCHAIN_TRACING_V2 = config("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = config("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = config("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = config("LANGCHAIN_PROJECT")