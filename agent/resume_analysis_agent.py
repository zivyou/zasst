import logging


from langchain.agents import create_agent
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from agent.agent import Agent
from config.apikey import ZHIPU_API_KEY
from tool.resume_tools import query_resume_info, get_current_date
from tool.error_handler import handle_tool_errors




sys_prompt_template = (
    """
    你是一个 LangChain 专家助手。

使用 query_resume_info 工具搜索相关信息，然后回答问题。

注意：
1. 优先使用检索到的信息
2. 如果信息不足，诚实告知
3. 回答要简洁准确
    """
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("resume_analysis_agent.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class ResumeAnalysisAgent(Agent):
    def __init__(self, system_prompt=sys_prompt_template):
        super().__init__(system_prompt)
        model = ChatZhipuAI(model="glm-4.5-air", api_key=ZHIPU_API_KEY)
        self.agent = create_agent(model=model, tools=[query_resume_info, get_current_date], system_prompt=system_prompt, middleware=[handle_tool_errors])


    def run(self, user_prompt: str):
        pass

    def call(self, user_prompt: str):
        config = {
            "callbacks": [AgentMonitor()]
        }
        response = self.agent.invoke(
            input={"messages": [HumanMessage(content=user_prompt)]},
            config=config
        )
        return response['messages'][-1].pretty_print()



class AgentMonitor(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        logger.info(f"  🔧 调用: input: {input_str}")

    def on_tool_end(self, output, **kwargs):
        logger.info(f"  ✅ 返回: output: {output}")

