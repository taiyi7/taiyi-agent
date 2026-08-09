# complex_agent.py
from typing import Optional, Iterator, List, Any, Dict
from taiyi_agent.core.agent import BaseAgent
from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.core.config import Config
from taiyi_agent.history.history import History
from taiyi_agent.context.context_manager import ContextManager
from taiyi_agent.tool.tool_registry import ToolRegistry

class StandardAgent(BaseAgent):
    '''
    包含工具管理、上下文管理、记忆管理等能力的标准Agent
    '''
    def __init__(
            self,
            name: str,
            llm: TaiyiAgentLLM,
            context: ContextManager,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            history: Optional[History] = None,
            tool_registry: Optional[ToolRegistry] = None,
            enable_tool_calling: bool = True

    ):
        super().__init__(name=name, llm=llm, system_prompt=system_prompt, config=config, history=history)
        self.context = context
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None

    def run(self, input: str, max_tool_iterations: int=3,  **kwargs) -> str:
        '''
        重写的run方法， 实现工具调用，上下文管理等功能
        '''
        pass
