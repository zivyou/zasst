"""
@File    :   test_sqlite3.py
@Time    :   2020/9/26 16:57
@Author  :   zivyou
@Email   :
"""
import sqlite3
import unittest


class MyTestCase(unittest.TestCase):
    """sqlite3 test cases"""

    def test_sqlit3(self):
        """sqlite3 test cases"""
        connect = sqlite3.connect("./data/library_menus.db")
        cursor = connect.cursor()
        menu = cursor.execute("select * from menu").fetchone()
        print(menu[2])
        cursor.close()
        connect.close()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
