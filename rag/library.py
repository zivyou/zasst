import json
import os
import sqlite3
from pathlib import Path

from langchain.agents import create_agent
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from config.apikey import ZHIPU_API_KEY

model = ChatZhipuAI(
    model="glm-4.7", api_key=ZHIPU_API_KEY
)


class Library:
    def __init__(self, dir_path="/home/ziv/Documents/library") -> None:
        self._dir = dir_path
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "；", ";", ".", ","]
        )
        embedding = ZhipuAIEmbeddings(model="embedding-3", api_key=ZHIPU_API_KEY)
        self.vector_store = Chroma(persist_directory="./data/zasst_library.db", embedding_function=embedding)
        root = Path(dir_path)
        pdfs = list(root.glob("**/*.pdf"))
        pdfs += list(root.glob("*.pdf"))
        with sqlite3.connect("./data/library_menus.db") as connection:
            cursor = connection.cursor()
            cursor.execute("create table if not exists menu(id integer primary key autoincrement, file_name varchar(256), file_update_time timestamp)")
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
                        self.vector_store.add_documents(
                            documents=chunks[i:i+60],
                        )
                    if menu is None:
                        cursor.execute(
                            "insert into menu(file_name, file_update_time) values (?, ?)", (pdf.name, modification_time)
                        )
                        connection.commit()

    def _query(self, query: str) -> str:
        if not self.vector_store:
            return "vector store not loaded"
        zh, en = self._generate_rag_query(query)
        retrieved_zh_docs = self.vector_store.similarity_search(
            query=zh,
            k=3
        )
        retrieved_en_docs = self.vector_store.similarity_search(
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
        class Response(BaseModel):
            queries: list[dict[str, str]]
            keywords: list[str]
            intent: str
        agent = create_agent(
            model=model,
            system_prompt="""
                你是一个专业的搜索查询优化专家。
                
                请根据用户问题生成3个优化后的搜索查询，用于从知识库中检索最相关的技术文档：
                1. 去除礼貌用语和口语化表达；
                2. 使用精确的技术术语；
                3. 从不同角度覆盖用户需求（原理、实现、应用场景）；
                4. 每个查询用英文和中文各生成一个版本；
                
                输出格式： 严格输出一个json字符串，JSON示例如下，确保你的JSON格式正确无误，可以直接被程序解析。不要在JSON前后添加任何说明或附加文本：  
                {
                  "queries": [
                    {{"zh": "...", "en": "..."}},
                    ...
                  ],
                  "keywords": ["...", "..."],
                  "intent": "..."
                }
            """,
            # ChatZhipuAI不支持response_format!!
            # response_format=Response
        )
        response = agent.invoke(
            input={"messages": [HumanMessage(f"用户问题：{query}")]},
        )
        content = response["messages"][-1].content
        content = json.loads(content)
        zh = content["queries"][0]["zh"]
        en = content["queries"][0]["en"]
        return zh, en

    async def ask(self, question: str) -> str:
        context = self._query(question)

        system_prompt = f"""
        你是一个专业的科研助手，请严格根据以下来源的上下文：
        {context}
        按步骤思考和回答问题：{question}
        如果上下文信息不足，请明确指出缺失的信息。最后用中文给出结构化答案。
        """

        agent = create_agent(
            model=model,
            system_prompt=system_prompt,
        )
        response = await agent.ainvoke(
            input={"messages": [HumanMessage(question)]}
        )
        return response["messages"][-1].content


