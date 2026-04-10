import asyncio
import unittest
from asyncio import as_completed

from concurrent.futures import ThreadPoolExecutor


class MyTestCase(unittest.TestCase):
    def print(self, i: int) -> None:
        print(i)

    def test_something(self):
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(10):
                futures.append(executor.submit(print, self, i))

            as_completed(futures)
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
