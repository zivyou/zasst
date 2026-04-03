import json
import sqlite3
import uuid
from typing import TypedDict, List, Dict, Any, Optional

from langchain.agents import create_agent
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from tavily import TavilyClient

from config.apikey import TAVILY_API_KEY
from config.apikey import ZHIPU_API_KEY
from tool.error_handler import handle_tool_errors
from tool.resume_tools import resume, get_current_date

tavily = TavilyClient(api_key=TAVILY_API_KEY)
# 调用 LLM 进行匹配度评估
model = ChatZhipuAI(
    model="glm-4.7", api_key=ZHIPU_API_KEY
)

system_prompt = """你是一个专业的简历匹配专家。
你的任务是根据岗位要求和候选人简历，评估匹配程度。

请从以下维度评估（满分100分）：
1. 技术栈匹配度：候选人掌握的技能是否与岗位要求匹配
2. 工作经验匹配度：候选人的工作年限和项目经验是否符合要求
3. 岗位要求符合度：其他硬性要求（如学历、证书等）的符合程度

评估标准：
- 90-100分：完全匹配，所有要求都满足
- 75-89分：基本匹配，大部分要求满足，部分略有差距
- 60-74分：部分匹配，核心技能满足，其他方面有差距
- 0-59分：不匹配，核心技能或关键要求不满足

请只返回一个0-100之间的整数分数，不要其他文字。"""


agent = create_agent(
    model=model,
    system_prompt=system_prompt,
    tools=[get_current_date],
    middleware=[handle_tool_errors]
)


# ==================== State Definition ====================
class JobSearchState(TypedDict):
    # 搜索条件
    search_conditions: Dict[str, Any]  # 地点、职位类型、技术栈等
    # 候选岗位列表
    job_candidates: List[Dict[str, Any]]
    # 当前正在处理的岗位索引
    current_index: int
    # 当前岗位的匹配结果
    current_match: Optional[Dict[str, Any]]
    # 匹配成功的岗位结果集
    matched_jobs: List[Dict[str, Any]]
    # 目标结果数量
    target_count: int
    # 匹配阈值 (0-100)
    match_threshold: float
    # 是否完成
    completed: bool

# ==================== Nodes ====================
def search_jobs_node(state: JobSearchState) -> JobSearchState:
    """
    节点1: 根据条件搜索岗位
    一次性获取多个岗位候选
    """
    conditions = state["search_conditions"]
    location = conditions.get("location", "北京")
    job_type = conditions.get("job_type", "后端")
    tech_stack = conditions.get("tech_stack", "Java")
    job_website = conditions.get("job_website", "Boss直聘")

    # 构建搜索查询
    query = f"在{job_website}上搜索 {location} {job_type} 工程师 {tech_stack} 招聘职位"

    print(f"🔍 正在搜索: {query}")

    candidates = []


    # 使用 Tavily 搜索
    search_result = tavily.search(
        query=query,
        max_results=15,  # 获取足够多的候选
        include_raw_content=True,
        search_depth="advanced"
    )

    # 解析搜索结果，提取岗位信息
    for result in search_result.get("results", []):
        candidates.append({
            "title": extract_title(result.get("title", "")),
            "company": extract_company(result.get("title", "")),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
            "raw_content": result.get("content", ""),
            "publish_date": extract_date(result.get("content", ""))
        })

    print(f"✅ 搜索到 {len(candidates)} 个候选岗位")

    return {
        "job_candidates": candidates,
        "current_index": 0,
        "current_match": None
    }


def match_job_node(state: JobSearchState) -> JobSearchState:
    """
    节点2: 将当前岗位与简历匹配
    使用 LLM 结合简历 RAG 评估匹配程度
    """
    candidates = state["job_candidates"]
    current_idx = state["current_index"]

    if current_idx >= len(candidates):
        return {"current_match": None}

    job = candidates[current_idx]
    print(f"📋 正在匹配: {job['title']} @ {job['company']}")


    # 查询简历信息
    try:
        resume_info = resume.query(f"我的技术栈、工作经验和技能")
    except Exception as e:
        print(f"⚠️  简历查询失败: {e}")
        resume_info = "简历信息获取失败"


    user_prompt = f"""
    【候选人简历信息】：
    {resume_info}
    
    【岗位信息】：
    职位: {job['title']}
    公司: {job['company']}
    岗位描述: {job['raw_content']}
    
    请根据以上信息，评估候选人与该岗位的匹配度（0-100）：
    """

    try:
        response = agent.invoke(
            input={"messages": [HumanMessage(user_prompt)]},
        )

        # 提取分数
        score = extract_score(response['messages'][-1].content)

        print(f"📊 匹配度: {score}%")

        return {
            "current_match": {
                "title": job['title'],
                "company": job['company'],
                "url": job['url'],
                "score": score,
                "publish_date": job['publish_date']
            }
        }
    except Exception as e:
        print(f"❌ 匹配评估失败: {e}")
        return {"current_match": None}


def collect_result_node(state: JobSearchState) -> JobSearchState:
    """
    节点3: 根据匹配度决定是否加入结果集
    """
    current_match = state["current_match"]
    threshold = state["match_threshold"]
    matched_jobs = state["matched_jobs"]

    if current_match and current_match["score"] >= threshold:
        print(f"✅ 匹配成功，加入结果集！")
        return {
            "matched_jobs": matched_jobs + [current_match],
            "current_match": None
        }
    else:
        print(f"⏭️  匹配度不足，跳过")
        return {"current_match": None}


def check_completion_node(state: JobSearchState) -> Dict[str, Any]:
    """
    节点4: 检查是否完成
    - 如果匹配结果 >= target_count，则结束
    - 如果已经处理完所有候选，则结束
    - 否则继续处理下一个
    """
    matched_jobs = state["matched_jobs"]
    candidates = state["job_candidates"]
    current_idx = state["current_index"]
    target_count = state["target_count"]

    # 检查是否达到目标数量
    if len(matched_jobs) >= target_count:
        print(f"🎯 已达到目标数量 {target_count}！")
        with open(f"./data/job_candidates.json", "w", encoding='utf-8') as f:
            json.dump(matched_jobs, f, ensure_ascii=False, indent=4)
        return {"completed": True}

    # 检查是否处理完所有候选
    next_idx = current_idx + 1
    if next_idx >= len(candidates):
        print(f"🏁 已处理完所有候选，共找到 {len(matched_jobs)} 个匹配岗位")
        with open(f"./data/job_candidates.json", "w", encoding='utf-8') as f:
            json.dump(candidates, f, ensure_ascii=False, indent=4)
        return {"completed": True}

    # 继续处理
    return {"completed": False}


def next_job_node(state: JobSearchState) -> JobSearchState:
    """
    节点5: 移动到下一个岗位
    """
    next_idx = state["current_index"] + 1
    return {"current_index": next_idx}


# ==================== Helper Functions ====================
def extract_title(title: str) -> str:
    """从标题中提取职位名称"""
    # 简单的提取逻辑，可以根据实际情况调整
    return title.strip()


def extract_company(title: str) -> str:
    """从标题中提取公司名称"""
    # 这里需要根据实际搜索结果格式调整
    # 简单的示例逻辑
    parts = title.split("_")
    if len(parts) > 1:
        return parts[1].strip()
    return "未知公司"


def extract_date(content: str) -> str:
    """从内容中提取发布时间"""
    # 简单的提取逻辑
    return "最近"


def extract_score(response: str) -> int:
    """从 LLM 响应中提取分数"""
    import re
    # 查找数字
    match = re.search(r'\d+', response)
    if match:
        score = int(match.group())
        return min(100, max(0, score))  # 限制在 0-100 范围内
    return 50  # 默认值


def create_graph() -> StateGraph:
    """构建求职搜索 LangGraph"""
    graph = StateGraph(JobSearchState)

    # 添加节点
    graph.add_node("search_jobs", search_jobs_node)
    graph.add_node("match_job", match_job_node)
    graph.add_node("collect_result", collect_result_node)
    graph.add_node("check_completion", check_completion_node)
    graph.add_node("next_job", next_job_node)

    # 添加边
    graph.add_edge(START, "search_jobs")
    graph.add_edge("search_jobs", "match_job")
    graph.add_edge("match_job", "collect_result")
    graph.add_edge("collect_result", "check_completion")

    # 条件边
    graph.add_conditional_edges(
        "check_completion",
        lambda state: "continue" if not state["completed"] else "end",
        {
            "continue": "next_job",
            "end": END
        }
    )
    graph.add_edge("next_job", "match_job")
    return graph

# ==================== Graph Building ====================
def build_job_search_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    g = create_graph()
    conn = sqlite3.connect("./data/checkpoint.db")
    sqlite_checkpointer = SqliteSaver(conn)
    return g.compile(checkpointer=sqlite_checkpointer)


if __name__ == '__main__':
    # 初始化状态 - 可自定义搜索条件
    initial_state = JobSearchState(
        search_conditions={
            "location": "北京",        # 工作地点
            "job_type": "后端",        # 职位类型：后端/架构
            "tech_stack": "Java"       # 技术栈
        },
        job_candidates=[],
        current_index=0,
        current_match=None,
        matched_jobs=[],
        target_count=10,              # 目标结果数量
        match_threshold=65.0,         # 匹配阈值
        completed=False
    )

    # 构建并运行图
    print("🚀 开始求职搜索...")
    print("="*60)

    graph = build_job_search_graph()
    png = graph.get_graph().draw_mermaid_png()
    with open("job_search_graph.png", "wb") as f:
        f.write(png)


    result = graph.invoke(initial_state, config={
        "checkpoint_id": str(uuid.uuid4()),
    })

    # 输出结果
    print("\n" + "="*60)
    print("📋 求职匹配结果")
    print("="*60)
    for i, job in enumerate(result["matched_jobs"], 1):
        print(f"\n{i}. {job['title']} @ {job['company']}")
        print(f"   匹配度: {job['score']}%")
        print(f"   链接: {job['url']}")
        print(f"   发布时间: {job['publish_date']}")

    print("\n" + "="*60)
    print(f"✅ 共找到 {len(result['matched_jobs'])} 个匹配的岗位")
    print("="*60)