from taiyi_agent.context.context_manager import ContextManager

class ContextBuilder:
    '''
    核心职责：请求初始化阶段，组装生成 ContextManager，只执行一次
    可接收输入：query、system 模板、user_id、session_id、memory、初始历史等
    执行流程：
    创建空 History
    从 MemoryHub 根据 query 召回长期记忆片段
    渲染 system prompt（支持模板变量填充）
    加载已有会话历史（可选，从存储恢复历史）
    将系统提示、召回记忆、历史消息依次填入 History
    实例化 ContextManager 并返回
    '''
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens

    async def build() -> ContextManager:
        pass


# ContextBuilder 是生产 ContextManager 的工厂，放在 Agent 上游；
# Agent 只接收工厂产出的成品 ContextManager，不需要持有工厂本身。