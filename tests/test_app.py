import unittest

from app import TuiApp


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_list_string(self):
        s = "hello world i     am testing"
        s2 = s.split()[1:]
        for x in s2:
            print(x)
        print(type(s2))
        self.assertTrue(type(s2) == list)
        self.assertTrue(type(s2[0]) == str)


if __name__ == '__main__':
    unittest.main()
