"""
@version: v1.0
"""
import unittest


class MyTestCase(unittest.TestCase):
    """ test main app """
    def test_something(self):
        """ test main app """
        self.assertEqual(True, True)  # add assertion here

    def test_list_string(self):
        """ test list string """
        s = "hello world i     am testing"
        s2 = s.split()[1:]
        for x in s2:
            print(x)
        print(type(s2))
        self.assertTrue(isinstance(s2, list))
        self.assertTrue(isinstance(s2[0], str))


if __name__ == '__main__':
    unittest.main()
