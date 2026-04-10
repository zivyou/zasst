"""
@File    :   test_sqlite3.py
@Time    :   2020/9/26 16:57
@Author  :   zivyou
@Email   :
"""
import time
import unittest
import uuid
from pathlib import Path
from time import sleep

from dotenv import load_dotenv
from opentelemetry.trace import StatusCode

from infra.tracer import tracer

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

class MyTestCase(unittest.TestCase):
    """tracer test cases"""
    def test_something(self):
        """tracer test cases"""
        uid = uuid.uuid4()
        span = tracer.start_span(uid)
        self.assertIsNotNone(span)
        span.set_attribute("foo", "bar")
        start_time = time.perf_counter()
        sleep(1)
        end_time = time.perf_counter()
        span.set_attribute("time.count", (end_time - start_time) * 1000)
        span.set_status(StatusCode.OK)
        span.end(end_time=time.time_ns())
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
