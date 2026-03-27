import os
import logging

def load_api_key(model_name:str):
    apikey=os.environ[model_name]
    logging.warning(f"Loading API key from {apikey}")
    return apikey


ZHIPU_API_KEY = load_api_key("ZAI_API_KEY")
TONGYI_API_KEY = load_api_key("TONGYI_API_KEY")
TAVILY_API_KEY = load_api_key("TAVILY_API_KEY")