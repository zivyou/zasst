from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from rag.library import model, otel_handler
from tool.library_research import research


class LibraryAgent:
    """library agent"""

    def __init__(self):
        system_prompt = """
        你是一个专业的科研助手，严格的使用ReAct模式来工作。
        你必须严格的按照 思考->行动->观察 这个循环来进行推理和行动，思考->行动->观察这个流程可以重复多次，最多重复10次。

        输出格式：
        用户会输入一个研究问题，使用工具列表中提供的工具在知识库中检索上下文，在获取足够多的信息后进行总结，总结的结果用中文按照
        论文的结构输出成Markdown格式。如果上下文信息不足，请直接回答信息不足。
        """

        self.agent = create_agent(
            model=model,
            system_prompt=system_prompt,
            tools=[research],
        )

    async def ask(self, question: str) -> str:
        """talk to the library agent"""
        response = await self.agent.ainvoke(
            input={"messages": [HumanMessage(question)]},
            config=RunnableConfig(callbacks=[otel_handler])
        )
        return response["messages"][-1].content
