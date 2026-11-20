"use strict";
const $ = (id) => document.getElementById(id);
const DEFAULT_THRESHOLD = { audio: 0.5, vision: 0.25 };
const WINDOW_SEC = 30;

let mode = "audio";
let running = false;
let status = { audio_available: true, vision_available: true };
let devices = { microphones: [], cameras: [] };
let history = []; // {t, prob}
const thresholds = { ...DEFAULT_THRESHOLD };

function showError(text) {
  $("error").textContent = text || "";
  $("error").hidden = !text;
}

function setState(text) { $("state").textContent = text; }

function setRunning(value) {
  running = value;
  $("start").disabled = value;
  $("stop").disabled = !value;
  $("device").disabled = value;
  $("file").disabled = value;
  document.querySelectorAll('input[name="kind"]').forEach((r) => (r.disabled = value));
  if (value) setState("Работает");
}

async function api(path, options) {
  const resp = await fetch(path, options);
  let body = {};
  try { body = await resp.json(); } catch (e) { /* пустой ответ */ }
  if (!resp.ok) throw new Error(body.error || "Ошибка сервера (" + resp.status + ")");
  return body;
}

function post(path, data) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
}

function kind() { return document.querySelector('input[name="kind"]:checked').value; }

function fillDevices() {
  const list = mode === "audio" ? devices.microphones : devices.cameras;
  const sel = $("device");
  sel.innerHTML = "";
  if (!list.length) {
    const o = document.createElement("option");
    o.textContent = mode === "audio" ? "Микрофоны не найдены" : "Камеры не найдены";
    o.value = "";
    sel.appendChild(o);
    return;
  }
  list.forEach((d) => {
    const o = document.createElement("option");
    o.value = d.id;
    o.textContent = d.name;
    sel.appendChild(o);
  });
}

function updateSourceControls() {
  const isFile = kind() === "file";
  $("device").hidden = isFile;
  $("file").hidden = !isFile;
  $("file").accept = mode === "audio" ? ".wav,.mp3,.flac,.ogg,audio/*" : "video/*";
}

function applyMode() {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.mode === mode));
  $("panel-audio").hidden = mode !== "audio";
  $("panel-vision").hidden = mode !== "vision";
  $("device-label").textContent = mode === "audio" ? "Микрофон" : "Камера";
  $("threshold").value = thresholds[mode];
  $("threshold-value").textContent = Number(thresholds[mode]).toFixed(2);
  const available = mode === "audio" ? status.audio_available : status.vision_available;
  const note = $("unavailable");
  note.hidden = available;
  if (!available) {
    note.textContent = "Режим недоступен: не найден файл модели в папке models/. Проверьте config.yaml.";
  }
  $("start").disabled = running || !available;
  fillDevices();
  updateSourceControls();
}

async function switchMode(newMode) {
  if (newMode === mode) return;
  try { await post("/api/stop", {}); } catch (e) { showError(e.message); }
  resetView();
  $("file").value = "";
  setRunning(false);
  setState("Остановлено");
  showError("");
  mode = newMode;
  applyMode();
}

function resetView() {
  history = [];
  $("indicator").className = "indicator off";
  $("indicator").textContent = "дрон не обнаружен";
  $("prob").textContent = "—";
  $("fps").textContent = "—";
  $("count").textContent = "—";
  $("video").removeAttribute("src");
  $("video").hidden = true;
  drawChart();
}

async function start() {
  showError("");
  try {
    const source = { kind: kind() };
    if (source.kind === "device") {
      const v = $("device").value;
      if (v !== "") source.device_id = Number(v);
    } else {
      const f = $("file").files[0];
      if (!f) throw new Error("Выберите файл");
      setState("Загрузка файла…");
      const fd = new FormData();
      fd.append("file", f);
      source.file_id = (await api("/api/upload", { method: "POST", body: fd })).file_id;
    }
    resetView();
    await post("/api/start", { mode, source, threshold: thresholds[mode] });
    setRunning(true);
    if (mode === "vision") {
      $("video").hidden = false;
      $("video").src = "/api/video.mjpg?t=" + Date.now();
    }
  } catch (e) {
    setRunning(false);
    setState("Остановлено");
    showError(e.message);
  }
}

async function stop() {
  try { await post("/api/stop", {}); } catch (e) { showError(e.message); }
  setRunning(false);
  setState("Остановлено");
  $("video").hidden = true;
  $("video").removeAttribute("src");
  setIndicator(false);
}

function setIndicator(on) {
  const el = $("indicator");
  el.className = "indicator " + (on ? "on" : "off");
  el.textContent = on ? "ДРОН ОБНАРУЖЕН" : "дрон не обнаружен";
}

function onEvent(ev) {
  if (ev.type === "audio" && mode === "audio") {
    setIndicator(ev.detected);
    $("prob").textContent = ev.prob.toFixed(2);
    history.push({ t: ev.t, prob: ev.prob });
    while (history.length && ev.t - history[0].t > WINDOW_SEC) history.shift();
    drawChart();
  } else if (ev.type === "vision" && mode === "vision") {
    $("fps").textContent = ev.fps.toFixed(1);
    $("count").textContent = ev.count;
  } else if (ev.type === "ended") {
    setRunning(false);
    setState("Файл закончился");
    setIndicator(false);
  } else if (ev.type === "error") {
    setRunning(false);
    setState("Остановлено");
    showError(ev.message);
  }
}

function drawChart() {
  const c = $("chart");
  const g = c.getContext("2d");
  const w = c.width, h = c.height;
  const css = getComputedStyle(document.body);
  g.clearRect(0, 0, w, h);
  const thr = h - thresholds.audio * h;
  g.strokeStyle = css.getPropertyValue("--muted");
  g.setLineDash([6, 4]);
  g.beginPath(); g.moveTo(0, thr); g.lineTo(w, thr); g.stroke();
  g.setLineDash([]);
  if (history.length < 2) return;
  const tEnd = history[history.length - 1].t;
  g.strokeStyle = css.getPropertyValue("--accent");
  g.lineWidth = 2;
  g.beginPath();
  history.forEach((p, i) => {
    const x = w - ((tEnd - p.t) / WINDOW_SEC) * w;
    const y = h - p.prob * (h - 4) - 2;
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  });
  g.stroke();
  g.lineWidth = 1;
}

function connectEvents() {
  const es = new EventSource("/api/events");
  es.onmessage = (m) => { try { onEvent(JSON.parse(m.data)); } catch (e) { /* пропускаем */ } };
}

async function init() {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchMode(t.dataset.mode)));
  document.querySelectorAll('input[name="kind"]').forEach((r) => r.addEventListener("change", updateSourceControls));
  $("start").addEventListener("click", start);
  $("stop").addEventListener("click", stop);
  $("threshold").addEventListener("input", () => {
    thresholds[mode] = Number($("threshold").value);
    $("threshold-value").textContent = thresholds[mode].toFixed(2);
    drawChart();
  });
  $("threshold").addEventListener("change", () => {
    post("/api/threshold", { value: thresholds[mode] }).catch((e) => showError(e.message));
  });
  try { status = await api("/api/status"); } catch (e) { showError(e.message); }
  try { devices = await api("/api/devices"); } catch (e) { /* списки останутся пустыми */ }
  if (!status.audio_available && status.vision_available) mode = "vision";
  $("tab-audio").disabled = !status.audio_available;
  $("tab-vision").disabled = !status.vision_available;
  setRunning(!!status.running);
  if (status.running && status.mode) mode = status.mode;
  applyMode();
  if (!running) $("stop").disabled = true;
  resetView();
  connectEvents();
}

init();
