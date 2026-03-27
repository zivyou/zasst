from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings

from config.apikey import ZHIPU_API_KEY


"""
万恶之源： pip3 install langchain-huggingface sentence_transformers
这行命令会去huggingface下载一个embedding模型到本地来跑，这个模型部署到本地会耗费大量的存储空间和计算资源；
后期考虑用zhipu的远程模型来替换；
"""

class Resume:
    def __init__(self, file_path: str):
        self.file_path = file_path
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "；", ";", ]
        )
        docs = PyMuPDFLoader(self.file_path).load()
        chunks = splitter.split_documents(docs)

        embedding = ZhipuAIEmbeddings(model="embedding-3", api_key=ZHIPU_API_KEY)
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory="./chroma_langchain.db"
        )


    def query(self, query: str):
        if not self.vector_store:
            return "vector store not loaded"
        retrieved_docs = self.vector_store.similarity_search(
            query=query,
            k=3
        )
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized


    def ask(self, question: str):
        if not self.vector_store:
            return "vector store not loaded"


