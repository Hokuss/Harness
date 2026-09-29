const logEl = document.getElementById("log");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const resetBtn = document.getElementById("reset-btn");
const statusEl = document.getElementById("status");
const toolListEl = document.getElementById("tool-list");
const toolCountEl = document.getElementById("tool-count");

function scrollToBottom() {
  logEl.scrollTop = logEl.scrollHeight;
}

function addBubble(text, cls) {
  const div = document.createElement("div");
  div.className = `bubble ${cls}`;
  div.textContent = text;
  logEl.appendChild(div);
  scrollToBottom();
  return div;
}

function addCard(summaryText, bodyObj) {
  const details = document.createElement("details");
  details.className = "card";
  const summary = document.createElement("summary");
  summary.textContent = summaryText;
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(bodyObj, null, 2);
  details.appendChild(summary);
  details.appendChild(pre);
  logEl.appendChild(details);
  scrollToBottom();
  return details;
}

async function loadStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  const stats = data.tree_stats?.stats || {};
  const langs = Object.entries(stats.files_by_language || {})
    .map(([lang, n]) => `${lang}:${n}`)
    .join(", ") || "none";
  statusEl.innerHTML =
    `<b>target:</b> ${data.target_dir}<br>` +
    `<b>files:</b> ${stats.total_files ?? 0}<br>` +
    `<b>languages:</b> ${langs}<br>` +
    `<b>nodes:</b> ${stats.total_nodes ?? 0}`;

  toolCountEl.textContent = data.tools.length;
  toolListEl.innerHTML = "";
  for (const t of data.tools) {
    const li = document.createElement("li");
    li.innerHTML = `<b>${t.name}</b> — ${t.description}`;
    toolListEl.appendChild(li);
  }
}

function handleEvent(evt) {
  switch (evt.type) {
    case "step_start":
      // Silent — the assistant/tool_call/tool_result cards that follow
      // already convey progress; a per-step divider would just be noise.
      break;
    case "assistant":
      if (evt.content && evt.content.trim()) {
        addBubble(evt.content.trim(), "assistant");
      }
      break;
    case "tool_call":
      addCard(`🛠 ${evt.name}(…) — click to view arguments`, evt.arguments);
      break;
    case "tool_result": {
      const preview = JSON.stringify(evt.result).slice(0, 60);
      addCard(`↳ result of ${evt.name}: ${preview}…`, evt.result);
      break;
    }
    case "final":
      if (evt.content && evt.content.trim()) {
        addBubble(evt.content.trim(), "assistant");
      }
      break;
    case "error":
      addBubble(`Error: ${evt.content}`, "error");
      break;
    case "max_iterations":
      addBubble("Stopped: hit the max iteration limit.", "system-note");
      break;
    case "done":
      sendBtn.disabled = false;
      inputEl.disabled = false;
      inputEl.focus();
      break;
    default:
      break;
  }
}

async function sendMessage(message) {
  addBubble(message, "user");
  sendBtn.disabled = true;
  inputEl.disabled = true;

  let res;
  try {
    res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
  } catch (err) {
    addBubble(`Network error: ${err}`, "error");
    sendBtn.disabled = false;
    inputEl.disabled = false;
    return;
  }

  if (!res.ok || !res.body) {
    let msg = res.statusText;
    try {
      const errJson = await res.json();
      msg = errJson.error || msg;
    } catch {
      // ignore
    }
    addBubble(`Request failed: ${msg}`, "error");
    sendBtn.disabled = false;
    inputEl.disabled = false;
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const line = rawEvent.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      try {
        handleEvent(JSON.parse(line.slice(6)));
      } catch (err) {
        console.error("bad SSE payload", err, line);
      }
    }
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;
  inputEl.value = "";
  sendMessage(message);
});

resetBtn.addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  logEl.innerHTML = "";
  addBubble("Conversation reset.", "system-note");
  await loadStatus();
});

loadStatus();
