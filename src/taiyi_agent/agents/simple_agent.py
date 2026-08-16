# simple_agent.py
from typing import Optional, Iterator, List, Any, Dict
from taiyi_agent.core.agent import BaseAgent
from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.core.message import Message
from taiyi_agent.core.config import Config
from taiyi_agent.history.history import History


class SimpleAgent(BaseAgent):
    '''
    实现一个简单的有记忆的对话Agent框架
    '''
    def __init__(
            self,
            name: str,
            llm: TaiyiAgentLLM,
            history: Optional[History] = None,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None
    ):
        super().__init__(name, llm, history, system_prompt, config)

    # 重写run方法， 实现简单的对话功能, 默认采用非流式输出
    def run(self, input: str, **kwargs) -> str:
        '''
        参数：
            input: 用户输入
            kwargs: 其他输入
        输出：
            agent返回值
        '''
        # 构造消息
        messages = self._build_messages(input)

        # 调用大模型进行思考
        response = self.llm.invoke(messages).content

        # 保存到历史记录，包含客户问题，以及大模型回答
        self.history.add_history(Message("user", input))
        self.history.add_history(Message("assistant", response))

        return response

    # 流式输出的run功能
    def stream_run(self, input: str, **kwargs) -> Iterator[str]:
        '''
        参数：
            input: 用户输入
            kwargs: 其他输入
        输出：
            agent 流式输出生成器
        '''
        # 构造消息
        messages = self._build_messages(input)
        response = self.llm.stream(messages)
        full_output = ''

        for chunk in response:
            full_output += chunk
            yield chunk                             # 创建生成器，正常外部不会执行遍历，只有外部使用驱动生成器的时候才会遍历

        # 保存到历史记录，包含客户问题，以及大模型回答
        self.history.add_history(Message("user", input))
        self.history.add_history(Message("assistant", full_output))
    
    def _build_messages(self, input: str) -> List[Dict[str, Any]]:
        # 创建初始消息
        messages = []

        # 首先添加系统消息
        if self.get_system_prompt():
            messages.append({"role": "system", "content": self.get_system_prompt()})

        # 添加历史消息
        for msg in  self.history.get_history():
            messages.append({"role": msg.role, "content": msg.content})

        # 添加用户输入内容
        messages.append({"role": "user", "content": input})

        return messages
