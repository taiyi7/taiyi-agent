'''Agent基类'''
from abc import ABC, abstractmethod
from typing import Optional
from config import Config
from llm import TaiyiAgentLLM
from message import Message

class Agent(ABC):
    '''
    本项目中所有agent的基类, 本身是抽象类不能被实例化
    但是在本类中有被 @abstractmethod 修饰的run()方法
    Agent的子类可以通过实现run()来可以被实例化
    '''
    def __init__(
            self,
            name: str,
            llm: TaiyiAgentLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config
        self._history: list[Message] = []

    @ abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        '''运行Agent'''
        pass
