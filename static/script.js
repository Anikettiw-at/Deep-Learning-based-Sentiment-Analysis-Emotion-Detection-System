const EMOJI = {
  sadness: "😢",
  joy: "😄",
  love: "❤️",
  anger: "😠",
  fear: "😨",
  surprise: "😲",
};

const EXAMPLES = [
  "I feel so happy and excited today",
  "I feel lonely and empty tonight",
  "I feel so loved by my friends",
  "I feel really annoyed that nobody showed up",
  "I feel scared about the exam tomorrow",
  "I was amazed when I saw the results",
];

const HISTORY_KEY = "moodline-history";
const HISTORY_MAX = 50;

const $ = (id) => document.getElementById(id);

const input = $("textInput");
const button = $("analyzeBtn");
const orb = $("orb");
const orbEmoji = $("orbEmoji");

let modelReady = false;
let busy = false;
let history = loadHistory();

// ---- server status ----

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    modelReady = data.model_loaded;
    if (modelReady) {
      setStatus("live", "model ready");
      return updateButton();
    }
    setStatus("warming", "model is loading…");
    setTimeout(checkHealth, 3000);
  } catch {
    setStatus("down", "can't reach the server, retrying…");
    setTimeout(checkHealth, 5000);
  }
  updateButton();
}

function setStatus(kind, text) {
  $("statusDot").className = "dot " + kind;
  $("statusText").textContent = text;
}

// ---- input ----

function updateButton() {
  button.disabled = busy || !modelReady || !input.value.trim();
}

input.addEventListener("input", () => {
  $("charCount").textContent = input.value.length;
  updateButton();
});

input.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    analyze();
  }
});

button.addEventListener("click", analyze);

for (const text of EXAMPLES) {
  const chip = document.createElement("button");
  chip.className = "chip";
  chip.textContent = text;
  chip.addEventListener("click", () => {
    input.value = text;
    input.dispatchEvent(new Event("input"));
    analyze();
  });
  $("examples").appendChild(chip);
}

// ---- prediction ----

async function analyze() {
  const text = input.value.trim();
  if (!text || !modelReady || busy) return;

  busy = true;
  updateButton();
  $("btnLabel").textContent = "Analyzing…";
  $("errorMsg").hidden = true;
  orb.classList.remove("settled");
  orb.classList.add("thinking");
  orbEmoji.style.opacity = 0;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = typeof data.detail === "string" ? data.detail : `Request failed (${res.status})`;
      throw new Error(msg);
    }
    showResult(data);
    addToHistory(data);
  } catch (err) {
    orbEmoji.textContent = "✎";
    $("errorMsg").textContent = err.message || "Something went wrong.";
    $("errorMsg").hidden = false;
  } finally {
    busy = false;
    orb.classList.remove("thinking");
    orbEmoji.style.opacity = 1;
    $("btnLabel").textContent = "Analyze";
    updateButton();
  }
}

function showResult(data) {
  const emotion = data.predicted_emotion;
  document.body.dataset.emotion = emotion;

  orbEmoji.textContent = EMOJI[emotion];
  orb.classList.add("settled");

  $("emotionEmoji").textContent = EMOJI[emotion];
  $("emotionWord").textContent = emotion;
  $("confidenceText").textContent = `${pct(data.confidence)} confidence`;
  $("echoedText").textContent = `“${data.text}”`;

  renderBars(data.all_probabilities);

  const section = $("resultSection");
  section.hidden = false;
  section.classList.remove("entering");
  void section.offsetWidth; // restart the animation
  section.classList.add("entering");
  section.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderBars(probs) {
  const bars = $("bars");
  bars.replaceChildren();

  Object.entries(probs)
    .sort((a, b) => b[1] - a[1])
    .forEach(([label, p], i) => {
      const row = document.createElement("div");
      row.className = `bar-row bar-${label}`;

      const name = document.createElement("span");
      name.className = "bar-label";
      name.textContent = `${EMOJI[label]} ${label}`;

      const track = document.createElement("span");
      track.className = "bar-track";
      const fill = document.createElement("span");
      fill.className = "bar-fill";
      track.appendChild(fill);

      const value = document.createElement("span");
      value.className = "bar-pct";
      value.textContent = pct(p);

      row.append(name, track, value);
      bars.appendChild(row);

      setTimeout(() => (fill.style.width = p * 100 + "%"), 60 + i * 70);
    });
}

const pct = (x) => (x * 100).toFixed(1) + "%";

// ---- history + downloads ----

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch {
    return [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch {
    // private mode / storage full - history just won't survive a reload
  }
}

function addToHistory(data) {
  history.unshift({ ...data, time: new Date().toISOString() });
  history = history.slice(0, HISTORY_MAX);
  saveHistory();
  renderHistory();
}

function renderHistory() {
  const list = $("historyList");
  list.replaceChildren();
  $("historySection").hidden = history.length === 0;

  for (const item of history) {
    const li = document.createElement("li");
    li.className = `history-item bar-${item.predicted_emotion}`;

    const tag = document.createElement("span");
    tag.className = "history-tag";
    tag.textContent = `${EMOJI[item.predicted_emotion]} ${item.predicted_emotion}`;

    const text = document.createElement("span");
    text.className = "history-text";
    text.textContent = item.text;
    text.title = "Click to load this sentence again";
    text.addEventListener("click", () => {
      input.value = item.text;
      input.dispatchEvent(new Event("input"));
      input.focus();
    });

    const conf = document.createElement("span");
    conf.className = "mono faint";
    conf.textContent = pct(item.confidence);

    li.append(tag, text, conf);
    list.appendChild(li);
  }
}

function download(filename, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function csvCell(value) {
  const s = String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

$("downloadCsv").addEventListener("click", () => {
  const labels = Object.keys(EMOJI);
  const header = ["time", "text", "predicted_emotion", "confidence", ...labels];
  const rows = history.map((h) => [
    h.time,
    h.text,
    h.predicted_emotion,
    h.confidence.toFixed(4),
    ...labels.map((l) => (h.all_probabilities[l] ?? 0).toFixed(4)),
  ]);
  const csv = [header, ...rows].map((r) => r.map(csvCell).join(",")).join("\n");
  download("moodline-history.csv", "﻿" + csv, "text/csv;charset=utf-8");
});

$("downloadJson").addEventListener("click", () => {
  download("moodline-history.json", JSON.stringify(history, null, 2), "application/json");
});

$("clearHistory").addEventListener("click", () => {
  if (!confirm("Clear all saved history?")) return;
  history = [];
  saveHistory();
  renderHistory();
});

renderHistory();
checkHealth();
