'''Agent基类'''
from abc import ABC, abstractmethod
from typing import Optional
from taiyi_agent.core.config import Config
from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.core.message import Message

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

    def get_history(self):
        '''获取历史记录'''
        return self._history.copy()
    
    def add_history(self, message:Message):
        '''将消息添加到历史'''
        self._history.append(message)

    def clear_history(self):
        '''清空历史记录'''
        self._history.clear()

    def get_system_prompt(self) -> str:
        '''读取系统提示词'''
        return self.system_prompt

    def set_system_prompt(self, new_system_prompt: str):
        '''重置系统提示词'''
        self.system_prompt = new_system_prompt

    def append_system_prompt(self, extra: str):
        '''补充系统提示词'''
        self.system_prompt += '\n' + extra

    def __str__(self) -> str:
        return f"Agent(name={self.name}, model={self.llm.llm_model_id})"