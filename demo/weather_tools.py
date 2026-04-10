"""weather info query tools"""
from langchain_core.tools import tool


@tool
def get_weather(location: str) -> str:
    """ Get weather information for a location """
    return f"Weather in {location}: Sunny 72C"
