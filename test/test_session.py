


store = InMemorySessionStore()
factory = AgentFactory()

manager = SessionManager(
    store=store,
    agent_factory=factory,
    llm=llm,
    tool_registry=tool_registry,
)

answer = await manager.send(
    session_id="user-1001",
    user_input="帮我查询今天的天气",
)

answer = await manager.send(
    session_id="user-1001",
    user_input="那明天呢？",
)