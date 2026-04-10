"""a demo agent base class"""
from abc import abstractmethod, ABC


class Agent(ABC):
    """agent base class"""
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    @abstractmethod
    def run(self, user_prompt: str):
        """run this agent"""

    @abstractmethod
    def call(self, user_prompt: str):
        """default call of this agent"""
