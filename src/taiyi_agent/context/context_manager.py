
class ContextManager:
    '''
    核心职责：维护会话持续状态，会话存续期间反复调用
    持有成员：history: History、session_id、user_id、token 上限、当前 token 统计
    消息操作封装：对外提供 add_message（优先使用，不允许外部直接操作 history）
    上下文策略：基于 token 限制滑动窗口裁剪、淘汰旧消息
    快照 snapshot ()：克隆一份独立上下文，用于 Agent 嵌套、分支隔离
    格式转换：输出模型可识别的 list[dict] LLM 消息数组
    快照序列化 / 反序列化（会话归档恢复使用）
    清空会话、重置上下文
    '''
    pass

# 用户提问
#     ↓
# ContextBuilder.build()
#     ├─ 调用memory检索长期记忆
#     ├─ 拼装system、记忆片段、历史消息 → History
#     ↓
# 产出 ContextManager 实例 → 注入Agent
#     ↓
# Agent运行期间持续调用 ctx.add_message()、ctx.trim()
#     ↓
# 会话结束：ctx可以序列化归档；Builder生命周期结束，不再复用