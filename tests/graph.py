import json
import logging
import unittest

from langgraph.checkpoint.memory import InMemorySaver

from agent.job_search_graph import create_graph, JobSearchState


class MyTestCase(unittest.TestCase):

    def test_graph_init_state(self):
        graph = create_graph()
        saver = InMemorySaver()
        g = graph.compile(checkpointer=saver)
        job_search_state = JobSearchState(target_count=10, search_conditions={})
        re = g.nodes["search_jobs"].invoke(job_search_state)
        print(json.dumps(re, ensure_ascii=False, indent=2))
        self.assertNotEqual(len(job_search_state['job_candidates']), 0)



if __name__ == '__main__':
    unittest.main()
