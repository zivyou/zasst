"""
author: zivyou
"""
import os
import time
import unittest
from rag.library import Library



class MyTestCase(unittest.TestCase):
    """ library test cases """
    def test_something(self):
        """ test library """
        library = Library("/home/ziv/Documents/library")
        start = time.time()
        content = library.query("volatile 可见性 happens-before")
        end = time.time()
        print(content)
        print(f"time cost = {end - start}")
        self.assertTrue(content is not None)

if __name__ == '__main__':
    unittest.main()
