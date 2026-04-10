"""
@version: v1.0
"""
import json
import unittest
import uuid

from langgraph.checkpoint.memory import InMemorySaver

from agent.job_search_graph import create_graph, JobSearchState, build_job_search_graph


class MyTestCase(unittest.TestCase):
    """job search graph test cases"""
    def test_graph_init_state(self):
        """test graph init state"""
        graph = create_graph()
        saver = InMemorySaver()
        g = graph.compile(checkpointer=saver)
        job_search_state = JobSearchState(target_count=10, search_conditions={})
        re = g.nodes["search_jobs"].invoke(job_search_state)
        print(json.dumps(re, ensure_ascii=False, indent=2))
        self.assertNotEqual(len(job_search_state['job_candidates']), 0)


    def test_job_search_graph(self):
        """test job search graph"""
        # 初始化状态 - 可自定义搜索条件
        initial_state = JobSearchState(
            search_conditions={
                "location": "北京",  # 工作地点
                "job_type": "后端",  # 职位类型：后端/架构
                "tech_stack": "Java"  # 技术栈
            },
            job_candidates=[],
            current_index=0,
            current_match=None,
            matched_jobs=[],
            target_count=10,  # 目标结果数量
            match_threshold=65.0,  # 匹配阈值
            completed=False
        )

        # 构建并运行图
        print("🚀 开始求职搜索...")
        print("=" * 60)

        graph = build_job_search_graph()
        png = graph.get_graph().draw_mermaid_png()
        with open("job_search_graph.png", "wb") as f:
            f.write(png)

        result = graph.invoke(initial_state, config={
            "checkpoint_id": str(uuid.uuid4()),
        })

        self.assertTrue(len(result['matched_jobs']) > 0)

        # 输出结果
        print("\n" + "=" * 60)
        print("📋 求职匹配结果")
        print("=" * 60)
        for i, job in enumerate(result["matched_jobs"], 1):
            print(f"\n{i}. {job['title']} @ {job['company']}")
            print(f"   匹配度: {job['score']}%")
            print(f"   链接: {job['url']}")
            print(f"   发布时间: {job['publish_date']}")

        print("\n" + "=" * 60)
        print(f"✅ 共找到 {len(result['matched_jobs'])} 个匹配的岗位")
        print("=" * 60)


if __name__ == '__main__':
    unittest.main()
