from abc import abstractmethod, ABC


class Agent(ABC):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    @abstractmethod
    def run(self, user_prompt: str):
        pass

    @abstractmethod
    def call(self, user_prompt: str):
        pass