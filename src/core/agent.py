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

    def add_history(self, message:Message):
        '''将消息添加到历史'''
        self._history.append(message)

    def clear_history(self):
        '''清空历史记录'''
        self._history.clear()

    def get_history(self):
        '''获取历史记录'''
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, model={self.llm.llm_model_id})"