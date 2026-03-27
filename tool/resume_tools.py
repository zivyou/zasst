import datetime

from langchain_core.tools import tool

from rag.resume_embedding import Resume

resume=Resume("~/Documents/my-resume.pdf")


@tool(response_format="content")
def query_resume_info(query:str):
    """ query my personal information from my resume """
    return resume.query(query)


@tool
def get_current_date():
    """
    查询今天的日期
    :return:
    """
    return datetime.datetime.now().strftime("%Y%m%d")