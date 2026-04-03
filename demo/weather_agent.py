from langchain.agents import create_agent
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain_core.messages import HumanMessage

from demo.agent import Agent
from demo.weather_tools import get_weather
from tool.error_handler import handle_tool_errors
from config.apikey import ZHIPU_API_KEY


class WeatherAgent(Agent):
    def __init__(self, system_prompt="你是一个天气查询小助手，为我提供各个城市的天气查询服务"):
        super().__init__(system_prompt)
        model = ChatZhipuAI(model="glm-4.5-air", api_key=ZHIPU_API_KEY)
        self.agent = create_agent(
            model, tools=[get_weather], middleware=[handle_tool_errors],
            # langchain框架会自动将system prompt 设置成内部的标准ReAct模式，langchain社区将其称为LangGraph
        )

    def run(self, user_prompt: str):
        prompt = self.agent.get_prompts({"messages": [HumanMessage(content=user_prompt)]})
        response = self.agent.invoke({"messages": [HumanMessage(content=user_prompt)]})
        print(response)

    def call(self, user_prompt: str):
        pass




