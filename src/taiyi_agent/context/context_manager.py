
class ContextManager:
    '''
    1. 用户输入 query
    2. ContextManager 记录用户消息
    3. ContextBuilder 读取 Manager 当前状态
    4. Builder 执行 GSSC：
    Gather → Select → Structure → Compress
    5. 生成本轮 messages
    6. 调用 LLM
    7. Manager 记录 assistant 消息
    8. Manager 更新 token、工具状态和任务状态
    9. 进入下一轮

    伪代码
        manager.add_message(Message(role="user", content=query))

        messages = builder.build(
            query=query,
            state=manager,
        )

        response = llm.invoke(messages)

        manager.add_message(
            Message(role="assistant", content=response)
        )
    '''
    pass
