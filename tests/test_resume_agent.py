"""
@File    :   test_resume_agent.py
@Desc    :
@Author  :   zivyou
@Contact :
"""
import unittest

from rag.resume_embedding import Resume


class MyTestCase(unittest.TestCase):
    """resume agent test cases"""
    def test_something(self):
        """resume agent test cases"""
        resume = Resume("~/Documents/my-resume.pdf")
        result = resume.query("我在最后一份工作干了几年？")
        print(result)
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
