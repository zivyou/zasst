

from langchain_core.globals import set_debug

from app import TuiApp

set_debug(True)


if __name__ == '__main__':
    app = TuiApp()
    app.run()