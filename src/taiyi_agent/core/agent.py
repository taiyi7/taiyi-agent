'''Agent基类'''
from abc import ABC, abstractmethod
from typing import Optional
from taiyi_agent.core.config import Config
from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.history.history import History

# 本身是抽象类不能被实例化，但是在本类中有被 @abstractmethod 修饰的run()方法
class BaseAgent(ABC):
    '''
    本项目中所有agent的基类, 
    Agent的子类可以通过实现run()来可以被实例化

    参数:
        name: agent命名
        llm: llm客户端实例
        history: 历史消息实例
        system_propmpt: 初始系统提示词
        config: 硬编码配置
    '''
    def __init__(
            self,
            name: str,
            llm: TaiyiAgentLLM,
            history: Optional[History] = None,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
    ):
        self.name = name
        self.llm = llm
        self.history = history if history is not None else History()
        self.system_prompt = system_prompt
        self.config = config

    @ abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        '''运行Agent'''
        ...

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