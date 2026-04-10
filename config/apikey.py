"""llm api keys"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

def load_api_key(model_name:str):
    """load api key from env"""
    apikey=os.getenv(model_name)
    logging.warning("Loading API key from %s", apikey)
    return apikey


ZHIPU_API_KEY = load_api_key("ZAI_API_KEY")
TONGYI_API_KEY = load_api_key("TONGYI_API_KEY")
TAVILY_API_KEY = load_api_key("TAVILY_API_KEY")
