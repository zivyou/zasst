from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import tool
from tavily import TavilyClient

from agent.agent import Agent
from config.apikey import ZHIPU_API_KEY
from tool.error_handler import handle_tool_errors

tavily = TavilyClient()

@tool
def internet_search(
        query: str,
        max_results: int=10,
        include_raw_content: bool=False,
):
    """ run a web search and return the top results """
    return tavily.search(query, max_results=max_results, include_raw_content=include_raw_content)


default_system_prompt = """
    你是一个专业的求职助手，你的工作是收集工作机会信息，并形成简报。

使用 internet_search 工具搜索相关信息，然后回答问题。
internet_search 有三个参数：query代表要查询的问题，max_results控制要返回的结果数量，include_raw_content控制是否要返回原始内容。
"""


class JobSearchAgent(Agent):
    def __init__(self, system_prompt=default_system_prompt):
        super().__init__(system_prompt)
        model = ChatZhipuAI(model="glm-4.5-air", api_key=ZHIPU_API_KEY)
        self.agent = create_agent(
            model=model,
            tools=[internet_search],
            system_prompt=system_prompt,
            middleware=[handle_tool_errors]
        )

    def run(self, user_prompt: str):
        pass

    def call(self, user_prompt: str):
        pass