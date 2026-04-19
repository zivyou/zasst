import os

from langchain_core.tools import tool

from rag.library import Library


@tool
def research(question:str) -> str:
    """
    在知识库中搜索问题相关的上下文
    :param question: str
    :return: str
    """
    return library.query(question)


library = Library(dir_path=os.getenv("LIBRARY_DIR"))
