import json
import os
import sqlite3
import uuid
from pathlib import Path

from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.apikey import ZHIPU_API_KEY
from infra.open_telemetry_callback_handler import OpenTelemetryCallbackHandler

model = ChatZhipuAI(
    model="glm-4.7", api_key=ZHIPU_API_KEY
)

otel_handler = OpenTelemetryCallbackHandler()

class Library:
    def __init__(self, dir_path: str | None, session_id=str(uuid.uuid4())) -> None:
        if dir_path:
            self._index_pdfs(dir_path)
        embedding = ZhipuAIEmbeddings(model="embedding-3", api_key=ZHIPU_API_KEY)
        self._vector_store = Chroma(persist_directory="./data/zasst_library.db", embedding_function=embedding)
        self._rag_query_generator = self._init_rag_query_generator(session_id)


    def _init_rag_query_generator(self, session_id):
        agent = create_agent(
            model=model,
            system_prompt="""
                        你是一个专业的搜索查询优化专家。

                        请根据用户问题生成1个优化后的搜索查询，用于从知识库中检索最相关的技术文档：
                        1. 去除礼貌用语和口语化表达；
                        2. 使用精确的技术术语；
                        3. 从不同角度覆盖用户需求（原理、实现、应用场景）；
                        4. 用英文和中文各生成一个版本；

                        输出格式： 严格输出一个json字符串，JSON示例如下，确保你的JSON格式正确无误，可以直接被程序解析。
                        请返回一个纯文本的JSON格式，不包含任何额外的标记或者格式化符号，如代码块标记(```)，不要在JSON前后添加任何说明或附加文本：
                        {
                          "queries": {"zh": "...", "en": "..."},
                          "keywords": ["...", "..."],
                          "intent": "..."
                        }
                    """,
            # ChatZhipuAI不支持response_format!!
            # response_format=Response
        )
        return agent

    def _index_pdfs(self, dir_path: str) -> None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "；", ";", ".", ","]
        )
        root = Path(dir_path)
        pdfs = list(root.glob("**/*.pdf"))
        pdfs += list(root.glob("*.pdf"))
        with sqlite3.connect("./data/library_menus.db") as connection:
            cursor = connection.cursor()
            cursor.execute(
                "create table if not exists menu(id integer primary key autoincrement, file_name varchar(256), file_update_time timestamp)")
            connection.commit()
            for pdf in pdfs:
                modification_time = os.path.getmtime(pdf)
                menu = cursor.execute("select * from menu where file_name=?", (pdf.name,)).fetchone()
                connection.commit()
                if menu is None or menu[2] < modification_time:
                    docs = PyMuPDFLoader(pdf).load()
                    chunks = splitter.split_documents(docs)
                    # 一次存储的chunks不能超过64条；
                    for i in range(0, len(chunks), 60):
                        self._vector_store.add_documents(
                            documents=chunks[i:i + 60],
                        )
                    if menu is None:
                        cursor.execute(
                            "insert into menu(file_name, file_update_time) values (?, ?)", (pdf.name, modification_time)
                        )
                        connection.commit()

    def query(self, query: str) -> str:
        if not self._vector_store:
            return "vector store not loaded"
        zh, en = self._generate_rag_query(query)
        retrieved_zh_docs = self._vector_store.similarity_search(
            query=zh,
            k=3
        )
        retrieved_en_docs = self._vector_store.similarity_search(
            query=en,
            k=3
        )
        retrieved_docs = retrieved_zh_docs + retrieved_en_docs
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized


    def _generate_rag_query(self, query: str) -> tuple[str, str]:
        response = self._rag_query_generator.invoke(
            input={"messages": [HumanMessage(f"用户问题：{query}")]},
            config=RunnableConfig(callbacks=[otel_handler])
        )
        content = response["messages"][-1].content
        content = json.loads(content)
        zh = content["queries"]["zh"]
        en = content["queries"]["en"]
        return zh, en

library = Library(dir_path=os.getenv("LIBRARY_DIR"))

@tool
def research(question:str) -> str:
    """
    在知识库中搜索问题相关的上下文
    :param question: str
    :return: str
    """
    return library.query(question)


class LibraryAgent:
    def __init__(self):
        system_prompt = f"""
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
        response = await self.agent.ainvoke(
            input={"messages": [HumanMessage(question)]},
            config=RunnableConfig(callbacks=[otel_handler])
        )
        return response["messages"][-1].content


