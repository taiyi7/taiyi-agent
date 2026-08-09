# history.py
from taiyi_agent.core.message import Message

class History:
    def __init__(self):
        self._history: list[Message] = []

    def get_history(self):
        '''获取历史记录'''
        return self._history.copy()
    
    def add_history(self, message:Message):
        '''将消息添加到历史'''
        self._history.append(message)

    def clear_history(self):
        '''清空历史记录'''
        self._history.clear()
