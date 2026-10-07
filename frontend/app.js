(() => {
  const API_BASE = (window.APP_CONFIG?.API_BASE || 'http://127.0.0.1:8000').replace(/\/$/, '');
  const state = { sessions: [], activeId: null, loading: false, online: false };
  const $ = (selector) => document.querySelector(selector);
  const els = {
    list: $('#session-list'), count: $('#session-count'), title: $('#conversation-title'), messages: $('#messages'), welcome: $('#welcome-card'),
    input: $('#message-input'), composer: $('#composer'), send: $('#send-button'), typing: $('#typing-row'), toast: $('#toast'), statusDot: $('#status-dot'), statusLabel: $('#status-label'),
    renameModal: $('#rename-modal'), renameForm: $('#rename-form'), renameInput: $('#rename-input'),
  };

  const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#039;' }[char]));
  const formatTime = (iso) => { if (!iso) return ''; const date = new Date(iso); return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }); };
  const sessionById = (id) => state.sessions.find((item) => item.session_id === id);
  const showToast = (message) => { els.toast.textContent = message; els.toast.classList.add('visible'); clearTimeout(showToast.timer); showToast.timer = setTimeout(() => els.toast.classList.remove('visible'), 2800); };
  const setOnline = (online) => { state.online = online; els.statusDot.className = `status-dot ${online ? 'online' : 'offline'}`; els.statusLabel.textContent = online ? '服务已连接' : '无法连接服务'; };

  async function request(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, { headers: { 'Content-Type': 'application/json', ...(options.headers || {}) }, ...options });
    if (!response.ok) { let detail = `请求失败（${response.status}）`; try { const body = await response.json(); detail = body.detail || detail; } catch (_) { /* empty response */ } throw new Error(detail); }
    return response.status === 204 ? null : response.json();
  }

  function renderSessions() {
    els.count.textContent = state.sessions.length;
    els.list.innerHTML = state.sessions.length ? state.sessions.map((session) => `
      <div class="session-item ${session.session_id === state.activeId ? 'active' : ''}" role="listitem" data-session-id="${session.session_id}">
        <button class="session-select" type="button" aria-label="打开 ${escapeHtml(session.title)}">
          <span class="session-icon">✦</span><span class="session-copy"><span class="session-title">${escapeHtml(session.title)}</span><span class="session-preview">${session.message_count ? `${session.message_count} 条消息 · ${formatTime(session.updated_at)}` : '开始一段新的对话'}</span></span>
        </button>
        <button class="session-more" type="button" aria-label="管理 ${escapeHtml(session.title)}">···</button>
      </div>`).join('') : '<div class="empty-sessions">还没有对话<br><span>点击上方按钮开始</span></div>';
  }

  function renderHistory(history = []) {
    els.messages.innerHTML = history.map((message) => {
      const isUser = message.role === 'user';
      if (!['user', 'assistant'].includes(message.role)) return '';
      return `<article class="message-row ${isUser ? 'user' : 'agent'}"><div class="avatar ${isUser ? 'user-avatar' : 'agent-avatar'}">${isUser ? '我' : '✦'}</div><div class="message-content"><div class="message-meta"><span>${isUser ? '你' : '太乙 Agent'}</span></div><div class="message-bubble">${escapeHtml(message.content)}</div></div></article>`;
    }).join('');
    els.welcome.classList.toggle('hidden', history.length > 0);
    requestAnimationFrame(() => { const stage = document.querySelector('.chat-stage'); stage.scrollTop = stage.scrollHeight; });
  }

  function appendMessage(role, content) {
    els.welcome.classList.add('hidden');
    els.messages.insertAdjacentHTML('beforeend', `<article class="message-row ${role === 'user' ? 'user' : 'agent'}"><div class="avatar ${role === 'user' ? 'user-avatar' : 'agent-avatar'}">${role === 'user' ? '我' : '✦'}</div><div class="message-content"><div class="message-meta"><span>${role === 'user' ? '你' : '太乙 Agent'}</span></div><div class="message-bubble">${escapeHtml(content)}</div></div></article>`);
    requestAnimationFrame(() => { const stage = document.querySelector('.chat-stage'); stage.scrollTop = stage.scrollHeight; });
  }

  async function refreshSessions() { state.sessions = await request('/sessions'); renderSessions(); }

  async function selectSession(id) {
    if (!id || state.loading) return;
    try { const session = await request(`/sessions/${id}`); state.activeId = id; els.title.textContent = session.title || '新对话'; renderSessions(); renderHistory(session.history); $('#sidebar').classList.remove('open'); }
    catch (error) { showToast(error.message); }
  }

  async function createSession() {
    try { const session = await request('/sessions', { method: 'POST', body: JSON.stringify({}) }); await refreshSessions(); await selectSession(session.session_id); els.input.focus(); }
    catch (error) { showToast(`创建对话失败：${error.message}`); }
  }

  async function sendMessage(message) {
    if (!state.activeId || !message.trim() || state.loading) return;
    const text = message.trim(); els.input.value = ''; resizeInput(); appendMessage('user', text); state.loading = true; els.send.disabled = true; els.typing.classList.remove('hidden');
    try { const result = await request(`/sessions/${state.activeId}/messages`, { method: 'POST', body: JSON.stringify({ message: text }) }); els.typing.classList.add('hidden'); appendMessage('assistant', result.answer); await refreshSessions(); }
    catch (error) { els.typing.classList.add('hidden'); appendMessage('assistant', `抱歉，这次请求没有完成：${error.message}`); showToast(error.message); }
    finally { state.loading = false; els.send.disabled = false; els.input.focus(); }
  }

  function openRename() { if (!state.activeId) return; const session = sessionById(state.activeId); els.renameInput.value = session?.title || '新对话'; els.renameModal.classList.remove('hidden'); requestAnimationFrame(() => { els.renameInput.focus(); els.renameInput.select(); }); }
  function closeRename() { els.renameModal.classList.add('hidden'); }
  async function renameSession(event) { event.preventDefault(); const title = els.renameInput.value.trim(); if (!title || !state.activeId) return; try { const session = await request(`/sessions/${state.activeId}`, { method: 'PATCH', body: JSON.stringify({ title }) }); const local = sessionById(state.activeId); if (local) local.title = session.title; els.title.textContent = session.title; renderSessions(); closeRename(); showToast('对话名称已更新'); } catch (error) { showToast(`重命名失败：${error.message}`); } }
  async function deleteSession() { if (!state.activeId) return; const session = sessionById(state.activeId); if (!window.confirm(`确定删除“${session?.title || '新对话'}”吗？\n删除后，这段对话的历史消息也会从 SessionStore 中清除。`)) return; const deleting = state.activeId; try { await request(`/sessions/${deleting}`, { method: 'DELETE' }); state.activeId = null; await refreshSessions(); els.messages.innerHTML = ''; els.welcome.classList.remove('hidden'); els.title.textContent = '新对话'; if (state.sessions[0]) await selectSession(state.sessions[0].session_id); else await createSession(); showToast('对话及其历史消息已删除'); } catch (error) { showToast(`删除失败：${error.message}`); } }

  function resizeInput() { els.input.style.height = 'auto'; els.input.style.height = `${Math.min(els.input.scrollHeight, 130)}px`; }
  els.list.addEventListener('click', (event) => { const item = event.target.closest('.session-item'); if (!item) return; const id = item.dataset.sessionId; if (event.target.closest('.session-more')) { state.activeId = id; openRename(); return; } selectSession(id); });
  $('#new-chat').addEventListener('click', createSession); $('#rename-current').addEventListener('click', openRename); $('#delete-current').addEventListener('click', deleteSession);
  $('#close-rename').addEventListener('click', closeRename); $('#cancel-rename').addEventListener('click', closeRename); els.renameForm.addEventListener('submit', renameSession); els.renameModal.addEventListener('click', (event) => { if (event.target === els.renameModal) closeRename(); });
  els.composer.addEventListener('submit', (event) => { event.preventDefault(); sendMessage(els.input.value); }); els.input.addEventListener('input', resizeInput); els.input.addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); els.composer.requestSubmit(); } });
  document.querySelectorAll('.suggestion').forEach((button) => button.addEventListener('click', () => { els.input.value = button.dataset.prompt; resizeInput(); els.composer.requestSubmit(); }));
  $('#open-sidebar').addEventListener('click', () => $('#sidebar').classList.add('open')); $('#close-sidebar').addEventListener('click', () => $('#sidebar').classList.remove('open'));
  document.addEventListener('keydown', (event) => { if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); createSession(); } if (event.key === 'Escape') closeRename(); });

  async function boot() { try { await request('/health'); setOnline(true); await refreshSessions(); if (state.sessions[0]) await selectSession(state.sessions[0].session_id); else await createSession(); } catch (error) { setOnline(false); showToast(`无法连接 Agent 服务：${error.message}`); renderSessions(); els.welcome.classList.remove('hidden'); } }
  boot();
})();
