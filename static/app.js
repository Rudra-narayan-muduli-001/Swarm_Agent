"use strict";

/* Swarm Agent — diagrammatic SSE client
   Contract per app/main.py:72 SSE events: session, agent, token, tool_call, tool_result, error, done
   Diagram = rail + lanes (one per agent turn). Chat = grouped bubbles.
*/

const AGENT_COLORS = { triage: "#E8B86A", researcher: "#7FC49A", writer: "#E07A5F" };
const AGENT_LABELS = { triage: "Planner", researcher: "Researcher", writer: "Writer" };
const FALLBACK = "#9F9FA6";

const chatEl = document.getElementById("chat");
const lanesEl = document.getElementById("lanes");
const traceScroll = document.getElementById("traceScroll");
const railSvg = document.getElementById("railSvg");
const railWrap = document.getElementById("railWrap");
const emptyState = document.getElementById("emptyState");
const liveDot = document.getElementById("liveDot");
const liveLabel = document.getElementById("liveLabel");
const sessionBadge = document.getElementById("sessionBadge");
const runCounterEl = document.getElementById("runCounter");
const legendEl = document.getElementById("legend");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");
const clearTraceBtn = document.getElementById("clearTraceBtn");

let sessionId = null;
let busy = false;
let currentAgent = null;
let runId = 0;
let runCount = 0;
let activeLane = null;
let chatBubbleForLane = new Map(); // laneId -> bubble .content
let pendingChips = []; // {tool, laneId, chipEl, resultEl}
let tokenQueue = "";
let rafPending = false;

// helpers
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}
function agentColor(name) { return AGENT_COLORS[name] || FALLBACK; }
function agentLabel(name) { return AGENT_LABELS[name] || name; }
function nowTime() {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function scrollBottom(node) { node.scrollTop = node.scrollHeight; }
function setLive(state, label) {
  if (state === "live") {
    liveDot.classList.add("live");
    liveLabel.classList.add("live");
    liveLabel.textContent = label || "live";
  } else if (state === "idle") {
    liveDot.classList.remove("live");
    liveLabel.classList.remove("live");
    liveLabel.textContent = label || "idle";
  } else {
    liveDot.classList.remove("live");
    liveLabel.classList.remove("live");
    liveLabel.textContent = label || state;
  }
}

// boot: legend + model hint
async function boot() {
  try {
    const res = await fetch("/api/agents");
    const data = await res.json();
    for (const a of data.agents || []) {
      const item = el("span", "legend-item");
      const dot = el("span", "legend-dot");
      dot.style.background = agentColor(a.name);
      dot.style.borderColor = agentColor(a.name);
      dot.style.boxShadow = `0 0 0 3px ${hexSoft(agentColor(a.name), 0.18)}`;
      item.appendChild(dot);
      item.appendChild(document.createTextNode(a.name));
      item.title = a.description || "";
      legendEl.appendChild(item);
    }
    const mh = document.getElementById("modelHint");
    if (mh && data.agents) mh.textContent = `${data.agents.length} agents · triage default`;
  } catch (e) {
    console.error("boot agents failed", e);
  }
  requestAnimationFrame(() => updateRail());
}

function hexSoft(hex, a) {
  // hex #RRGGBB -> rgba
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

// rail drawing — simple vertical spine with nodes
function updateRail() {
  const lanes = lanesEl.querySelectorAll(".lane");
  if (!lanes.length) {
    railSvg.innerHTML = "";
    railSvg.style.height = "320px";
    return;
  }
  // compute Y for each lane centre relative to railWrap
  const wrapRect = railWrap.getBoundingClientRect();
  const points = [];
  lanes.forEach((lane) => {
    const r = lane.getBoundingClientRect();
    const y = (r.top - wrapRect.top) + r.height / 2 + traceScroll.scrollTop - 14; // 14 = trace padding top
    // clamp
    points.push(Math.max(20, y));
  });
  const h = Math.max(320, (points[points.length-1] || 0) + 40);
  railSvg.style.height = h + "px";
  railSvg.setAttribute("viewBox", `0 0 28 ${h}`);
  railSvg.setAttribute("height", h);
  // build path
  let svg = "";
  // vertical spine
  if (points.length > 1) {
    const pathD = points.map((y, i) => `${i===0?'M':'L'} 14 ${y}`).join(" ");
    const isDone = !busy;
    svg += `<path d="${pathD}" class="rail-segment ${busy ? 'active':'done'}" stroke-linecap="round"/>`;
  }
  // nodes
  lanes.forEach((lane, i) => {
    const y = points[i];
    const agent = lane.dataset.agent;
    const color = agentColor(agent);
    const isActive = lane.classList.contains("active");
    const isDoneLane = lane.classList.contains("done");
    const r = isActive ? 6 : 5;
    const ringR = isActive ? 9 : 0;
    const fill = isDoneLane ? color : (isActive ? color : "#343438");
    const stroke = isActive ? hexSoft(color, 0.35) : "#2A2A2E";
    if (isActive) {
      svg += `<circle cx="14" cy="${y}" r="${ringR}" fill="${hexSoft(color,0.14)}"/>`;
    }
    svg += `<circle cx="14" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="1.4"/>`;
    if (isDoneLane) {
      svg += `<text x="14" y="${y+3.5}" text-anchor="middle" font-size="7" fill="#0C0C0E" font-family="Inter" font-weight="700">✓</text>`;
    }
  });
  railSvg.innerHTML = svg;
}

// lanes
function ensureLane(agent) {
  // hide empty
  if (emptyState) emptyState.style.display = "none";
  // if last lane is same agent and still active (streaming), reuse
  if (activeLane && activeLane.dataset.agent === agent && activeLane.classList.contains("active")) {
    return activeLane;
  }
  // deactivate previous
  if (activeLane) {
    activeLane.classList.remove("active");
  }
  const lane = el("div", "lane active");
  lane.dataset.agent = agent;
  lane.dataset.run = String(runId);

  const head = el("div", "lane-head");
  const left = el("div", "lane-agent");
  const dot = el("span", "lane-dot");
  left.appendChild(dot);
  left.appendChild(document.createTextNode(agentLabel(agent) + " · " + agent));
  const meta = el("div", "lane-meta");
  const time = el("span", "lane-time", nowTime());
  const badge = el("span", "lane-badge live", "live");
  meta.appendChild(time);
  meta.appendChild(badge);
  head.appendChild(left);
  head.appendChild(meta);

  const body = el("div", "lane-body");
  body.textContent = "";

  const toolList = el("div", "tool-list");
  // keep refs
  lane._body = body;
  lane._badge = badge;
  lane._toolList = toolList;

  lane.appendChild(head);
  lane.appendChild(body);
  lane.appendChild(toolList);

  lanesEl.appendChild(lane);
  activeLane = lane;

  // also create chat bubble for this lane (grouped streaming)
  const bubble = el("div", "bubble assistant");
  bubble.dataset.lane = String(Date.now()) + Math.random().toString(16).slice(2);
  // tiny agent tag inside bubble
  const tag = el("div", "agent-strip");
  const pill = el("span", "agent-pill", agent);
  pill.style.borderColor = agentColor(agent);
  pill.style.color = agentColor(agent);
  // strip is outside bubble for cleaner look — insert strip then bubble
  const strip = el("div", "agent-strip");
  strip.appendChild(pill);
  chatEl.appendChild(strip);
  bubble.appendChild(el("div", "content"));
  chatEl.appendChild(bubble);
  // map lane -> bubble content
  const content = bubble.querySelector(".content");
  chatBubbleForLane.set(lane, content);

  scrollBottom(traceScroll);
  scrollBottom(chatEl);
  requestAnimationFrame(() => updateRail());
  return lane;
}

function markLaneDone(lane, state) {
  if (!lane) return;
  lane.classList.remove("active");
  lane.classList.add("done");
  if (lane._badge) {
    lane._badge.textContent = state === "error" ? "error" : "done";
    lane._badge.className = "lane-badge " + (state==="error" ? "error" : "done");
  }
}

function addRunSeparator(text) {
  const sep = el("div", "agent-strip");
  sep.style.margin = "6px 0 2px 0";
  const line = el("span", "", text);
  line.style.fontSize = "11px";
  line.style.color = "var(--ink-3)";
  line.style.letterSpacing = "0.04em";
  sep.appendChild(line);
  lanesEl.appendChild(sep);
  // also chat gap
  const csep = el("div", "agent-strip");
  csep.appendChild(el("span", "agent-pill", text));
  const pill = csep.querySelector(".agent-pill");
  pill.style.borderColor = "var(--line-strong)";
  pill.style.color = "var(--ink-3)";
  pill.style.background = "var(--bg-2)";
  chatEl.appendChild(csep);
}

// tokens — batched via rAF to avoid layout thrash
function appendTokens(fragment) {
  if (!fragment) return;
  tokenQueue += fragment;
  if (!rafPending) {
    rafPending = true;
    requestAnimationFrame(() => {
      const text = tokenQueue;
      tokenQueue = "";
      rafPending = false;
      if (!activeLane) return;
      activeLane._body.textContent += text;
      const chatContent = chatBubbleForLane.get(activeLane);
      if (chatContent) chatContent.textContent += text;
      scrollBottom(traceScroll);
      scrollBottom(chatEl);
    });
  }
}

function addToolCall(data) {
  const agent = data.agent || currentAgent || "triage";
  const lane = ensureLane(agent);
  // ensure lane is visible
  const chip = el("div", "tool-chip running");
  const head = el("div", "tool-chip-head");
  const name = el("span", "tool-chip-name", data.tool);
  // color name by tool
  if (data.tool === "web_search") name.style.color = "var(--sage)";
  else if (data.tool === "read_url") name.style.color = "var(--clay)";
  else if (String(data.tool).startsWith("handoff")) name.style.color = agentColor(agent);
  const agentTag = el("span", "tool-chip-agent", agent + " · " + nowTime());
  head.appendChild(name);
  head.appendChild(agentTag);
  chip.appendChild(head);
  const argsText = JSON.stringify(data.arguments || {});
  const preview = argsText.length > 180 ? argsText.slice(0,180) + "…" : argsText;
  const argsEl = el("div", "tool-chip-args", preview);
  chip.appendChild(argsEl);
  // placeholder result
  const resultEl = el("div", "tool-chip-result collapsed");
  resultEl.style.display = "none";
  chip.appendChild(resultEl);
  lane._toolList.appendChild(chip);
  pendingChips.push({ tool: data.tool, lane, chip, resultEl });
  scrollBottom(traceScroll);
  requestAnimationFrame(() => updateRail());
}

function addToolResult(data) {
  // find most recent pending chip with same tool that hasn't resolved
  let idx = -1;
  for (let i = pendingChips.length - 1; i >= 0; i--) {
    if (pendingChips[i].tool === data.tool && pendingChips[i].chip.classList.contains("running")) {
      idx = i; break;
    }
  }
  if (idx === -1) {
    // fallback: find any pending
    idx = pendingChips.findIndex(p => p.tool === data.tool && p.chip.classList.contains("running"));
    if (idx === -1) return;
  }
  const entry = pendingChips[idx];
  const chip = entry.chip;
  chip.classList.remove("running");
  const isError = String(data.result || "").toLowerCase().includes("error");
  chip.classList.add(isError ? "error" : "done");
  const resultText = String(data.result || "").trim();
  const preview = resultText.slice(0, 900);
  entry.resultEl.textContent = preview + (resultText.length > 900 ? "\n…[truncated, click expand]" : "");
  entry.resultEl.style.display = "block";
  // if handoff, also show in lane as handoff banner
  if (String(data.tool).startsWith("handoff") || preview.toLowerCase().includes("handoff to")) {
    const banner = el("div", "lane-handoff");
    const arrow = el("span", "arrow", "→");
    const txt = el("span", "", preview.slice(0, 220));
    banner.appendChild(arrow);
    banner.appendChild(txt);
    entry.lane.appendChild(banner);
  }
  // expand toggle
  if (resultText.length > 220) {
    const btn = el("button", "chip-toggle", "Expand");
    btn.addEventListener("click", () => {
      const collapsed = entry.resultEl.classList.contains("collapsed");
      if (collapsed) {
        entry.resultEl.classList.remove("collapsed");
        entry.resultEl.textContent = resultText.slice(0, 8000) + (resultText.length > 8000 ? "\n…[truncated]" : "");
        btn.textContent = "Collapse";
      } else {
        entry.resultEl.classList.add("collapsed");
        entry.resultEl.textContent = preview + (resultText.length > 900 ? "\n…[truncated]" : "");
        btn.textContent = "Expand";
      }
    });
    chip.appendChild(btn);
  }
  scrollBottom(traceScroll);
  requestAnimationFrame(() => updateRail());
}

function addUserMessage(text) {
  const bubble = el("div", "bubble user");
  bubble.appendChild(el("div", "content", text));
  chatEl.appendChild(bubble);
  scrollBottom(chatEl);
}

function addErrorBubble(msg) {
  const bubble = el("div", "bubble error", "error: " + msg);
  chatEl.appendChild(bubble);
  if (activeLane) markLaneDone(activeLane, "error");
  scrollBottom(chatEl);
  scrollBottom(traceScroll);
}

function setBusy(state) {
  busy = state;
  sendBtn.disabled = state;
  inputEl.disabled = state;
  if (state) {
    statusEl.textContent = "listening…";
    setLive("live", "listening");
  } else {
    statusEl.textContent = "";
    setLive("idle", "idle");
    updateRail();
  }
}

// send
function send() {
  const text = inputEl.value.trim();
  if (!text || busy) return;
  setBusy(true);
  runId += 1;
  runCount += 1;
  runCounterEl.textContent = runCount + (runCount===1 ? " run" : " runs");
  currentAgent = null;
  activeLane = null;
  pendingChips = [];
  tokenQueue = "";
  addUserMessage(text);
  addRunSeparator("Run #" + runCount + " · " + nowTime());
  // reset input
  inputEl.value = "";
  autoGrow();
  // hide greeting after first send
  const greet = document.getElementById("greeting");
  if (greet) greet.style.display = "none";

  const params = new URLSearchParams({ message: text });
  if (sessionId) params.set("session_id", sessionId);

  const es = new EventSource("/api/chat?" + params.toString());

  es.addEventListener("session", (e) => {
    try {
      const d = JSON.parse(e.data);
      sessionId = d.session_id;
      sessionBadge.textContent = sessionId;
      sessionBadge.title = sessionId;
    } catch {}
  });

  es.addEventListener("agent", (e) => {
    try {
      const d = JSON.parse(e.data);
      const name = d.name;
      if (currentAgent && name !== currentAgent) {
        // mark previous done
        if (activeLane) {
          activeLane.classList.remove("active");
          // keep badge live until done
        }
      }
      currentAgent = name;
      ensureLane(name);
      setLive("live", name);
    } catch {}
  });

  es.addEventListener("token", (e) => {
    try {
      const d = JSON.parse(e.data);
      appendTokens(d.content || "");
    } catch {}
  });

  es.addEventListener("tool_call", (e) => {
    try { addToolCall(JSON.parse(e.data)); } catch {}
  });

  es.addEventListener("tool_result", (e) => {
    try { addToolResult(JSON.parse(e.data)); } catch {}
  });

  es.addEventListener("error", (e) => {
    let msg = "unknown error";
    try { if (e.data) msg = JSON.parse(e.data).message || msg; } catch {}
    addErrorBubble(msg);
    es.close();
    setBusy(false);
    if (activeLane) markLaneDone(activeLane, "error");
  });

  es.addEventListener("done", () => {
    es.close();
    // flush any pending tokens
    if (tokenQueue) {
      const t = tokenQueue; tokenQueue = "";
      if (activeLane) {
        activeLane._body.textContent += t;
        const cc = chatBubbleForLane.get(activeLane);
        if (cc) cc.textContent += t;
      }
    }
    // mark active lane done
    if (activeLane) markLaneDone(activeLane, "done");
    // if bubble empty show placeholder
    chatBubbleForLane.forEach((content) => {
      if (!content.textContent.trim()) content.textContent = "(no text — see trace for tool steps)";
    });
    setBusy(false);
    updateRail();
  });

  es.onerror = () => {
    // EventSource fires onerror on close; ignore if already done will be handled via done/error
  };
}

// input handlers
function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
}

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
  setTimeout(autoGrow, 0);
});
inputEl.addEventListener("input", autoGrow);
sendBtn.addEventListener("click", send);

clearTraceBtn.addEventListener("click", () => {
  // keep last run separator style but clear lanes except empty state for visual reset
  // we don't clear chat — only trace view per button label
  lanesEl.innerHTML = "";
  lanesEl.appendChild(emptyState);
  emptyState.style.display = "";
  activeLane = null;
  pendingChips = [];
  chatBubbleForLane.clear();
  updateRail();
  setLive("idle", "idle");
});

// resize observer for rail
const ro = new ResizeObserver(() => updateRail());
ro.observe(lanesEl);
ro.observe(traceScroll);
window.addEventListener("resize", () => updateRail());

// boot
boot();
autoGrow();
updateRail();
