import sqlite3
import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_sqlit3(self):
        connect = sqlite3.connect("./data/library_menus.db")
        cursor = connect.cursor()
        menu = cursor.execute("select * from menu").fetchone()
        print(menu[2])
        cursor.close()
        connect.close()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
