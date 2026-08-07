"use strict";

const AGENT_COLORS = { triage: "#3b82f6", researcher: "#22c55e", writer: "#a855f7" };
const FALLBACK_COLOR = "#94a3b8";

const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");

let sessionId = null;
let busy = false;
let currentAgent = null;
let assistantBubble = null;
let pendingChips = [];

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function agentColor(name) {
  return AGENT_COLORS[name] || FALLBACK_COLOR;
}

function scrollBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

async function boot() {
  try {
    const res = await fetch("/api/agents");
    const data = await res.json();
    const legend = document.getElementById("legend");
    for (const a of data.agents || []) {
      const item = el("span", "legend-item", a.name);
      item.style.borderColor = agentColor(a.name);
      item.style.color = agentColor(a.name);
      item.title = a.description || "";
      legend.appendChild(item);
    }
  } catch (err) {
    console.error("failed to load agents", err);
  }
}

function addUserMessage(text) {
  const bubble = el("div", "bubble user");
  bubble.appendChild(el("div", "content", text));
  chatEl.appendChild(bubble);
  scrollBottom();
}

function newAssistantBubble() {
  assistantBubble = el("div", "bubble assistant");
  assistantBubble.appendChild(el("div", "tool-feed"));
  assistantBubble.appendChild(el("div", "content"));
  chatEl.appendChild(assistantBubble);
  scrollBottom();
}

function appendTokens(text) {
  if (!assistantBubble) newAssistantBubble();
  const content = assistantBubble.querySelector(".content");
  content.textContent += text;
  scrollBottom();
}

function addAgentDivider(name) {
  const strip = el("div", "agent-strip");
  const pill = el("span", "agent-pill", "handoff to " + name);
  pill.style.borderColor = agentColor(name);
  pill.style.color = agentColor(name);
  strip.appendChild(pill);
  chatEl.appendChild(strip);
  scrollBottom();
}

function addToolCall(data) {
  if (!assistantBubble) newAssistantBubble();
  const feed = assistantBubble.querySelector(".tool-feed");
  const chip = el("div", "tool-chip");
  const argsText = JSON.stringify(data.arguments || {});
  chip.appendChild(el("span", "tool-name", data.agent + " -> " + data.tool));
  chip.appendChild(el("span", "tool-args", argsText.length > 120 ? argsText.slice(0, 120) + "..." : argsText));
  feed.appendChild(chip);
  pendingChips.push({ chip, tool: data.tool });
  scrollBottom();
}

function addToolResult(data) {
  const idx = pendingChips.findIndex((p) => p.tool === data.tool && !p.done);
  if (idx === -1) return;
  const entry = pendingChips[idx];
  entry.done = true;
  const resultText = String(data.result || "").slice(0, 200);
  entry.chip.appendChild(el("span", "tool-result", "=> " + resultText));
  scrollBottom();
}

function addError(message) {
  chatEl.appendChild(el("div", "bubble error", "error: " + message));
  scrollBottom();
}

function setBusy(state) {
  busy = state;
  sendBtn.disabled = state;
}

function send() {
  const text = inputEl.value.trim();
  if (!text || busy) return;
  setBusy(true);
  currentAgent = null;
  assistantBubble = null;
  pendingChips = [];
  addUserMessage(text);

  const params = new URLSearchParams({ message: text });
  if (sessionId) params.set("session_id", sessionId);

  const es = new EventSource("/api/chat?" + params.toString());

  es.addEventListener("session", (e) => {
    sessionId = JSON.parse(e.data).session_id;
  });

  es.addEventListener("agent", (e) => {
    const name = JSON.parse(e.data).name;
    if (currentAgent && name !== currentAgent) addAgentDivider(name);
    currentAgent = name;
  });

  es.addEventListener("token", (e) => {
    appendTokens(JSON.parse(e.data).content);
  });

  es.addEventListener("tool_call", (e) => {
    addToolCall(JSON.parse(e.data));
  });

  es.addEventListener("tool_result", (e) => {
    addToolResult(JSON.parse(e.data));
  });

  es.addEventListener("error", (e) => {
    if (e.data) addError((JSON.parse(e.data).message) || "unknown error");
    es.close();
    setBusy(false);
  });

  es.addEventListener("done", () => {
    es.close();
    setBusy(false);
    if (assistantBubble && !assistantBubble.querySelector(".content").textContent) {
      assistantBubble.querySelector(".content").textContent = "(no response)";
    }
  });
}

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
  setTimeout(autoGrow, 0);
});

function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
}

sendBtn.addEventListener("click", send);
boot();
