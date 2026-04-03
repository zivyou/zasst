import json
import os
from typing import Callable

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call

import logging

def setup_logger(file_path:str, level=logging.DEBUG) -> logging.Logger:
    log_dir = os.path.dirname(file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    re = logging.getLogger()
    re.setLevel(level)

    re.handlers.clear()

    file_handler = logging.FileHandler(file_path, encoding="utf-8", mode='w')
    file_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    )
    re.addHandler(file_handler)
    return re

logger = setup_logger("./logs/prompt_debugger.log", logging.DEBUG)


@wrap_model_call
def debug_prompt(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    logger.debug(f"request: {json.dumps(request.messages, indent=4)}")
    return handler(request)

