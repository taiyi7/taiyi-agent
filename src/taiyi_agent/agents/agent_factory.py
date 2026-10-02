
from pydantic import BaseModel
from typing import Callable

from taiyi_agent.agents.agent_step import AgentStepExecutor
from taiyi_agent.agents.react_agent import ReactAgent
from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.sessions.session import SessionState
from taiyi_agent.tool.tool_calling import ToolCallingStrategy
from taiyi_agent.tool.tool_registry import ToolRegistry
from taiyi_agent.context.context_assembler import ContextAssembler


class AgentBuildModel(BaseModel):
    """创建 Agent 时需要的共享依赖。"""

    session: SessionState
    llm: TaiyiAgentLLM
    context_assembler:ContextAssembler
    tool_registry: ToolRegistry | None = None
    tool_calling_strategy: ToolCallingStrategy | None = None

    # Pydantic 只会接受传入的对象，*不会验证这个对象内部属性，不会序列化它*
    model_config = {"arbitrary_types_allowed": True}

# AgentBuilder 是一个构建函数，接收构建上下文，产出一个 Agent 实例（返回 object）
AgentBuilder = Callable[
    [AgentBuildModel],
    object,
]


class AgentFactory:
    """
    根据 session.agent_type 创建 Agent。

    新增 Agent 类型时，只需要注册 builder，
    不需要修改 SessionManager。
    """

    def __init__(self) -> None:
        self._builders: dict[str, AgentBuilder] = {}

        self.register(
            "react",
            self._build_react_agent,
        )

    def register(
        self,
        agent_type: str,
        builder: AgentBuilder,
    ) -> None:
        if agent_type in self._builders:
            raise ValueError(
                f"Agent 类型已注册：{agent_type}"
            )

        self._builders[agent_type] = builder

    def create(
        self,
        session: SessionState,
        *,
        llm: TaiyiAgentLLM,
        context_assembler: ContextAssembler,
        tool_registry: ToolRegistry | None = None,
        tool_calling_strategy: ToolCallingStrategy | None = None,
    ) -> object:
        builder = self._builders.get(session.agent_type)

        if builder is None:
            raise ValueError(
                f"不支持的 Agent 类型：{session.agent_type}"
            )

        context = AgentBuildModel(
            session=session,
            llm=llm,
            context_assembler=context_assembler,
            tool_registry=tool_registry,
            tool_calling_strategy=tool_calling_strategy,
        )

        return builder(context)

    @staticmethod
    def _build_react_agent(
        context: AgentBuildModel,
    ) -> ReactAgent:
        step_executor = AgentStepExecutor(
            llm=context.llm,
            tool_registry=context.tool_registry,
            tool_calling_strategy=context.tool_calling_strategy,
        )

        return ReactAgent(
            step_executor=step_executor,
            context_assembler=context.context_assembler,
            agent_system_prompt=context.session.system_prompt,
            max_steps=context.session.max_steps,
        )