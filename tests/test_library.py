import os
import time
import unittest

import dotenv
from sympy import content

dotenv.load_dotenv()
from rag.library import Library


class MyTestCase(unittest.TestCase):
    def test_something(self):
        library = Library(os.getenv("LIBRARY_DIR"))
        start = time.time()
        content = library.query("volatile 可见性 happens-before")
        end = time.time()
        print(content)
        print(f"time cost = {end - start}")
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
