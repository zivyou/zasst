import asyncio
from typing import Dict, List

from langchain.agents import create_agent
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from agent.chat_agent import ChatAgent
from agent.library_agent import LibraryAgent
from config.apikey import ZHIPU_API_KEY

model = ChatZhipuAI(model="glm-4.5-air", temperature=0, api_key=ZHIPU_API_KEY)

@tool
def list_agents() -> List[Dict[str, str]]:
    """list all agents, return each agent name and agent description"""
    return [{
        "agent_name": "LibraryAgent",
        "agent_description": "personal knowledge base agent. research fields includes: AI, Computer Science, Software Engineering, math",
    }, {
        "agent_name": "ChatAgent",
        "agent_description": "a general LLM chat agent",
    }]

library_agent = LibraryAgent()
chat_agent = ChatAgent()

@tool
def delegate_to_agent(agent_name:str, ctx: str) -> str:
    """
    delegate task to agent.
    Args:
        agent_name: name of the agent
        ctx: context, usually the args to call agent entry function
    return:
        response content
    """
    if agent_name == "LibraryAgent":
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(library_agent.ask(ctx))
        loop.close()
        return result
    elif agent_name == "ChatAgent":
        return chat_agent.chat(ctx)
    else:
        return "zasst does not know what to do"


class SupervisorAgent:
    def __init__(self):
        self.system_prompt = """
        你是一个multi agents系统的协调者。
        你需要按照你的理解将任务委托给合适的agent去运行。
        """
        self.agent = create_agent(
            model=model,
            system_prompt=self.system_prompt,
            tools=[list_agents,delegate_to_agent]
        )

    def execute(self, ctx: str) -> str:
        response = self.agent.invoke(
            input= {"messages": [HumanMessage(ctx)]}
        )
        return response["messages"][-1].content