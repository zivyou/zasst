"""
@version: v1.0
zasst
"""

from langchain_core.globals import set_debug
import dotenv

from app import TuiApp

dotenv.load_dotenv()
set_debug(True)

if __name__ == '__main__':
    app = TuiApp()
    app.run()
