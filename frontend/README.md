# Taiyi Agent 前端

这是一个零构建依赖的静态前端页面，直接打开 `index.html` 即可使用。默认连接 `http://127.0.0.1:8000`；部署到其他环境时，在加载 `app.js` 前设置：

```html
<script>window.APP_CONFIG = { API_BASE: 'https://你的-api-地址' };</script>
```

启动 FastAPI：

```bash
uvicorn taiyi_agent.deploy.react_deploy_fastapi:app --reload --port 8000
```

页面支持连续对话、历史上翻、创建会话、会话重命名和删除。删除确认后，前端调用 `DELETE /sessions/{session_id}`，后端会同步删除 `SessionStore` 中的历史消息。
