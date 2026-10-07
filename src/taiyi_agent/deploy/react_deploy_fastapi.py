"""使用 FastAPI 部署 Taiyi Agent 的多 session 对话接口。

每个 session_id 对应一个独立的 SessionState。客户端在后续请求中继续
使用同一个 session_id，即可复用该 session 的 history，形成连续对话。

当前使用 InMemorySessionStore，因此数据只存在于当前进程内。生产环境可在
不改变本文件路由的前提下，将它替换成实现 SessionStore 协议的持久化存储。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from taiyi_agent.agents.agent_factory import AgentFactory
from taiyi_agent.context.context_assembler import ContextAssembler
from taiyi_agent.context.context_builder import ContextBuilder
from taiyi_agent.context.context_data import ContextConfig
from taiyi_agent.context.token_counter import TokenCounter
from taiyi_agent.context.token_window import TokenWindow
from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.sessions.session import SessionState
from taiyi_agent.sessions.session_manager import SessionManager
from taiyi_agent.sessions.session_store import InMemorySessionStore
from taiyi_agent.tool.tool_registry import ToolRegistry
from taiyi_agent.tool.datetime_tool.get_current_time_tool import GetCurrentTimeTool
from taiyi_agent.tool.search_tool.tavily_search import TavilySearchTool
from taiyi_agent.tool.search_tool.web_search_tool import SerpSearchTool
from taiyi_agent.tool.search_tool.weather_tool import WeatherTool


class CreateSessionRequest(BaseModel):
    """创建 session 时可选的会话级配置。"""

    system_prompt: str | None = None
    agent_type: str = "react"
    max_steps: int = Field(default=6, ge=1, le=100)


class CreateSessionResponse(BaseModel):
    session_id: UUID
    agent_type: str
    max_steps: int
    title: str = "新对话"


class ChatRequest(BaseModel):
    """一次用户 turn。一个请求只处理一条用户消息。"""

    message: str = Field(min_length=1)
    max_steps: int | None = Field(default=None, ge=1, le=100)


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str


class SessionSummary(BaseModel):
    """左侧会话列表使用的轻量会话信息。"""

    session_id: UUID
    title: str
    revision: int
    message_count: int
    created_at: str
    updated_at: str


class RenameSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class SessionResponse(BaseModel):
    session_id: UUID
    agent_type: str
    system_prompt: str | None
    max_steps: int
    revision: int
    history: list[dict[str, Any]]
    title: str = "新对话"


def get_weather(date: str, city: str) -> str:
    """示例工具；可替换为项目实际的业务工具。"""

    return f"{date}{city}的天气为晴天"


def build_session_manager() -> SessionManager:
    """构造一次应用级共享依赖。

    LLM、ContextBuilder、TokenWindow 和 SessionStore 在所有 session 之间共享，
    但 SessionState 由 session_id 分开保存，因此对话历史彼此隔离。
    """

    llm = TaiyiAgentLLM()
    store = InMemorySessionStore()
    factory = AgentFactory()

    tool_registry = ToolRegistry()
    tool_list = [
        GetCurrentTimeTool(),
        TavilySearchTool(),
        SerpSearchTool(),
        WeatherTool()
    ]
    for tool in tool_list:
        tool_registry.register_tool(tool)

    context_config = ContextConfig()
    token_counter = TokenCounter()
    context_builder = ContextBuilder(
        config=context_config,
        llm=llm,
        counter=token_counter,
    )
    token_window = TokenWindow(
        config=context_config,
        counter=token_counter,
    )
    context_assembler = ContextAssembler(
        builder=context_builder,
        token_window=token_window,
    )

    return SessionManager(
        store=store,
        agent_factory=factory,
        llm=llm,
        context_assembler=context_assembler,
        tool_registry=tool_registry,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """在应用启动时初始化共享 manager，避免导入模块时立即连接 LLM。"""

    app.state.session_manager = build_session_manager()
    app.state.session_titles = {}
    yield
    app.state.session_manager = None
    app.state.session_titles = {}


app = FastAPI(
    title="Taiyi Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

# 前端可以作为独立静态页面运行，因此允许本地开发服务器跨域访问 API。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_manager(request: Request) -> SessionManager:
    manager = getattr(request.app.state, "session_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent 服务尚未完成初始化",
        )
    return manager


async def load_existing_session(
    manager: SessionManager,
    session_id: UUID,
) -> SessionState:
    session = await manager.load(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session 不存在：{session_id}",
        )
    return session


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    request: Request,
    payload: CreateSessionRequest | None = None,
) -> CreateSessionResponse:
    """创建一个新的、与其他会话隔离的 session。"""

    manager = require_manager(request)
    data = payload or CreateSessionRequest()

    session = SessionState(
        session_id=uuid4(),
        agent_type=data.agent_type,
        system_prompt=data.system_prompt,
        max_steps=data.max_steps,
    )
    await manager.store.save(session)
    request.app.state.session_titles[str(session.session_id)] = "新对话"

    return CreateSessionResponse(
        session_id=session.session_id,
        agent_type=session.agent_type,
        max_steps=session.max_steps,
        title="新对话",
    )


@app.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(request: Request) -> list[SessionSummary]:
    """返回当前进程内的会话索引，供前端渲染左侧列表。"""

    manager = require_manager(request)
    summaries: list[SessionSummary] = []
    titles = request.app.state.session_titles
    # SessionStore 当前是内存实现；索引只保存由本 API 创建的 session id。
    for session_id_text in list(titles):
        session = await manager.load(UUID(session_id_text))
        if session is None:
            titles.pop(session_id_text, None)
            continue
        summaries.append(
            SessionSummary(
                session_id=session.session_id,
                title=titles.get(session_id_text, "新对话"),
                revision=session.revision,
                message_count=len(session.history),
                created_at=session.created_at.isoformat(),
                updated_at=session.updated_at.isoformat(),
            )
        )
    summaries.sort(key=lambda item: item.updated_at, reverse=True)
    return summaries


@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: UUID, request: Request) -> SessionResponse:
    """读取会话元数据和已完成的长期历史。"""

    manager = require_manager(request)
    session = await load_existing_session(manager, session_id)

    return SessionResponse(
        session_id=session.session_id,
        agent_type=session.agent_type,
        system_prompt=session.system_prompt,
        max_steps=session.max_steps,
        revision=session.revision,
        history=[message.model_dump(mode="json") for message in session.history],
        title=request.app.state.session_titles.get(str(session_id), "新对话"),
    )


@app.patch("/sessions/{session_id}", response_model=SessionSummary)
async def rename_session(
    session_id: UUID,
    payload: RenameSessionRequest,
    request: Request,
) -> SessionSummary:
    """更新会话标题；会话历史仍由同一个 SessionStore 保存。"""

    manager = require_manager(request)
    session = await load_existing_session(manager, session_id)
    title = " ".join(payload.title.split())
    if not title:
        raise HTTPException(status_code=422, detail="会话名称不能为空")
    request.app.state.session_titles[str(session_id)] = title
    return SessionSummary(
        session_id=session.session_id,
        title=title,
        revision=session.revision,
        message_count=len(session.history),
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
    )


@app.post("/sessions/{session_id}/messages", response_model=ChatResponse)
async def chat(session_id: UUID, payload: ChatRequest, request: Request) -> ChatResponse:
    """向指定 session 追加一轮消息并返回回答。

    同一个 session_id 的多次调用会读取并更新同一个 SessionState；不同
    session_id 之间不会共享 history。SessionManager 内部还会串行化同一
    session 的并发 turn，避免两条消息同时修改同一份历史。
    """

    manager = require_manager(request)
    session = await load_existing_session(manager, session_id)

    try:
        answer = await manager.send(
            session_id=session.session_id,
            user_input=payload.message,
            agent_type=session.agent_type,
            system_prompt=session.system_prompt,
            max_steps=payload.max_steps or session.max_steps,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Agent 调用失败：{exc}",
        ) from exc

    return ChatResponse(session_id=session_id, answer=answer)


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: UUID, request: Request) -> None:
    """删除会话及其历史。"""

    manager = require_manager(request)
    await load_existing_session(manager, session_id)
    await manager.delete(session_id)
    request.app.state.session_titles.pop(str(session_id), None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "taiyi_agent.deploy.react_deploy_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
