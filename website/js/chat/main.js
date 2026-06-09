const API_BASE = (() => {
  const param = new URLSearchParams(location.search).get("api");
  if (param) return param.replace(/\/$/, "");
  if (location.port === "3000") return "/api";
  return "http://127.0.0.1:9001";
})();

let nextId = 1;
let activeId = 0;
const chats = new Map();

const tabRoot = document.getElementById("chat-tabs");
const tabNew = document.getElementById("chat-tab-new");
const messageRoot = document.getElementById("chat-messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");

function createChat(title = "") {
  const id = nextId++;
  chats.set(id, {
    title: title || `chat ${id}`,
    messages: [{ role: "assistant", content: "Ready." }],
    pending: false,
    draft: "",
  });
  return id;
}

function activeChat() {
  return chats.get(activeId);
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function tabLabel(chat) {
  const firstUser = chat.messages.find((message) => message.role === "user");
  if (firstUser) {
    const text = firstUser.content.trim();
    return text.length > 18 ? `${text.slice(0, 18)}...` : text;
  }
  return chat.title;
}

function messageRow(role, content, extraClass = "") {
  const arrow = role === "user" ? "&lt;" : "&gt;";
  return `<article class="chat-message ${role}">
    <div class="chat-row">
      <span class="chat-arrow" aria-hidden="true">${arrow}</span>
      <div class="chat-content${extraClass}">${escapeHtml(content)}</div>
    </div>
  </article>`;
}

function renderTabs() {
  if (!tabRoot) return;
  tabRoot.innerHTML = Array.from(chats.entries()).map(([id, chat]) => {
    const active = id === activeId ? " active" : "";
    const label = escapeHtml(tabLabel(chat));
    return `<div class="chat-tab${active}" data-chat-id="${id}">
      <button type="button" class="chat-tab-label" role="tab" aria-selected="${id === activeId}">${label}</button>
      <button type="button" class="chat-tab-close" aria-label="Close chat">x</button>
    </div>`;
  }).join("");
}

function renderMessages() {
  if (!messageRoot) return;
  const chat = activeChat();
  if (!chat) {
    messageRoot.innerHTML = "";
    return;
  }
  const rows = chat.messages.map((message) => messageRow(message.role, message.content));
  if (chat.pending) {
    rows.push(messageRow("assistant", "thinking...", " pending"));
  }
  messageRoot.innerHTML = rows.join("");
  messageRoot.scrollTop = messageRoot.scrollHeight;
}

function syncInput() {
  if (!input) return;
  const chat = activeChat();
  input.value = chat?.draft || "";
  input.disabled = !!chat?.pending;
  autoSizeInput();
}

function render() {
  renderTabs();
  renderMessages();
  syncInput();
}

function saveDraft() {
  const chat = activeChat();
  if (!chat || !input) return;
  chat.draft = input.value;
}

function switchChat(id) {
  if (!chats.has(id) || id === activeId) return;
  saveDraft();
  activeId = id;
  render();
  input?.focus();
}

function addChat() {
  saveDraft();
  activeId = createChat();
  render();
  input?.focus();
}

function deleteChat(id) {
  if (!chats.has(id)) return;
  if (chats.size === 1) {
    chats.delete(id);
    activeId = createChat();
    render();
    input?.focus();
    return;
  }
  const wasActive = id === activeId;
  const ids = Array.from(chats.keys());
  const idx = ids.indexOf(id);
  chats.delete(id);
  if (wasActive) {
    activeId = ids[idx + 1] ?? ids[idx - 1];
  }
  render();
  if (wasActive) input?.focus();
}

function setPending(nextPending) {
  const chat = activeChat();
  if (!chat) return;
  chat.pending = nextPending;
  if (input) input.disabled = nextPending;
  render();
}

function autoSizeInput() {
  if (!input) return;
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
}

async function fetchReply(chat) {
  const messages = chat.messages.map((message) => ({
    role: message.role,
    content: message.content,
  }));
  const resp = await fetch(`${API_BASE}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || !data.ok) {
    throw new Error(data.error || `request failed (${resp.status})`);
  }
  return data.content || "";
}

async function sendMessage(event) {
  event.preventDefault();
  const chat = activeChat();
  if (!chat || !input || chat.pending) return;
  const content = input.value.trim();
  if (!content) return;
  chat.messages.push({ role: "user", content });
  chat.draft = "";
  input.value = "";
  autoSizeInput();
  setPending(true);
  try {
    const reply = await fetchReply(chat);
    chat.messages.push({ role: "assistant", content: reply || "..." });
  } catch (err) {
    chat.messages.push({
      role: "assistant",
      content: err instanceof Error ? err.message : "request failed",
    });
  } finally {
    setPending(false);
    input.focus();
  }
}

async function bootstrap() {
  try {
    const resp = await fetch(`${API_BASE}/chat/config`);
    const data = await resp.json();
    if (data.ok && activeChat()) {
      activeChat().messages[0].content = "Ready.";
    }
  } catch {
    // offline or proxy missing; send will surface the error
  }
  render();
}

tabRoot?.addEventListener("click", (event) => {
  const closeBtn = event.target.closest(".chat-tab-close");
  if (closeBtn) {
    event.preventDefault();
    const tab = closeBtn.closest("[data-chat-id]");
    if (tab) deleteChat(Number(tab.dataset.chatId));
    return;
  }
  const label = event.target.closest(".chat-tab-label");
  if (!label) return;
  const tab = label.closest("[data-chat-id]");
  if (tab) switchChat(Number(tab.dataset.chatId));
});

tabNew?.addEventListener("click", addChat);
form?.addEventListener("submit", sendMessage);
input?.addEventListener("input", () => {
  saveDraft();
  autoSizeInput();
});
input?.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form?.requestSubmit();
  }
});

activeId = createChat();
bootstrap();
