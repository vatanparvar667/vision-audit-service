"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  drop: $("drop"), fileInput: $("fileInput"), player: $("player"), video: $("video"),
  marks: $("marks"), played: $("played"), durLabel: $("durLabel"),
  idle: $("stateIdle"), work: $("stateWork"), fail: $("stateFail"), done: $("stateDone"),
  workFile: $("workFile"), meter: $("meter"), meterFill: $("meterFill"), workStage: $("workStage"),
  failText: $("failText"), retryBtn: $("retryBtn"), resetBtn: $("resetBtn"),
  stamp: $("stamp"), stampText: $("stampText"), verdictMeta: $("verdictMeta"), verdictNote: $("verdictNote"),
  findings: $("findings"), typesList: $("typesList"), lightbox: $("lightbox"),
  lightboxImg: $("lightboxImg"), lightboxCap: $("lightboxCap"), tpl: $("findingTpl"),
};

const STAGES = {
  queued: "В очереди на анализ",
  decoding: "Чтение видео",
  analyzing: "Анализ видео",
  detecting: "Поиск людей и предметов",
  tracking: "Отслеживание действий",
  rules: "Проверка правил",
  verifying: "Проверка найденных моментов",
  report: "Подготовка отчёта",
};

let pollTimer = null;
let objectUrl = null;
let currentJob = null;

function show(state) {
  for (const el of [els.idle, els.work, els.fail, els.done]) el.hidden = el !== state;
}

function fmtTime(sec) {
  const s = Math.max(0, sec);
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  return `${m}:${rest.toFixed(1).padStart(4, "0")}`;
}

function setMeter(fraction, text) {
  const pct = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
  els.meterFill.style.width = pct + "%";
  els.meter.setAttribute("aria-valuenow", String(pct));
  if (text) els.workStage.textContent = text;
}

// ---- violation catalog for the idle screen --------------------------------------------------
fetch("/api/v1/violation-types").then((r) => r.json()).then((data) => {
  els.typesList.replaceChildren(...data.types.map((t) => {
    const li = document.createElement("li");
    const b = document.createElement("b");
    b.textContent = t.name;
    const span = document.createElement("span");
    span.textContent = t.summary;
    li.append(b, span);
    return li;
  }));
}).catch(() => {});

// ---- upload ---------------------------------------------------------------------------------
function pickFile(file) {
  if (!file) return;
  reset(false);
  els.drop.hidden = true;
  els.player.hidden = false;
  els.resetBtn.hidden = false;
  objectUrl = URL.createObjectURL(file);
  els.video.src = objectUrl;
  els.workFile.textContent = file.name;
  show(els.work);
  setMeter(0, "Загрузка видео на сервер — 0%");

  const form = new FormData();
  form.append("file", file);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/ui/analyze");
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) setMeter(e.loaded / e.total * 0.15, `Загрузка видео на сервер — ${Math.round(e.loaded / e.total * 100)}%`);
  };
  xhr.onload = () => {
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch (_) {}
    if (xhr.status >= 200 && xhr.status < 300 && body.job_id) {
      currentJob = body.job_id;
      setMeter(0.15, STAGES.queued);
      poll();
    } else {
      fail(body.detail || `Сервер ответил кодом ${xhr.status}. Проверьте, что файл — видео в формате .mov.`);
    }
  };
  xhr.onerror = () => fail("Не удалось отправить файл: сервер недоступен. Проверьте подключение и попробуйте ещё раз.");
  xhr.send(form);
}

function poll() {
  clearTimeout(pollTimer);
  fetch(`/ui/jobs/${currentJob}`).then((r) => r.json()).then((job) => {
    if (job.status === "done") return render(job);
    if (job.status === "failed") return fail(`Анализ остановился с ошибкой: ${job.error || "неизвестная ошибка"}. Попробуйте загрузить файл ещё раз.`);
    let text = STAGES[job.stage] || STAGES.analyzing;
    if (job.stage === "queued" && job.queue_position > 0) text += ` — перед вами ${job.queue_position}`;
    else if (job.progress) text += ` — ${Math.round(job.progress * 100)}%`;
    setMeter(0.15 + 0.85 * (job.progress || 0), text);
    pollTimer = setTimeout(poll, 1000);
  }).catch(() => { pollTimer = setTimeout(poll, 2500); });
}

function fail(text) {
  els.failText.textContent = text;
  show(els.fail);
}

// ---- result ---------------------------------------------------------------------------------
function render(job) {
  const violations = job.violations || [];
  const isViolation = job.verdict === "violation";
  els.stamp.classList.toggle("is-clear", !isViolation);
  els.stampText.textContent = isViolation ? "Обнаружены признаки возможных нарушений" : "Нарушение не обнаружено";
  els.stamp.classList.remove("is-pressed");
  void els.stamp.offsetWidth;
  els.stamp.classList.add("is-pressed");

  const meta = [["Файл", els.workFile.textContent]];
  if (isViolation) meta.push(["Найдено моментов", String(violations.length)]);
  if (job.processing_sec) meta.push(["Время анализа", `${Math.round(job.processing_sec)} с`]);
  els.verdictMeta.replaceChildren(...meta.flatMap(([k, v]) => {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v;
    return [dt, dd];
  }));
  els.verdictNote.textContent = isViolation
    ? "Проверьте каждый момент по кадрам и видео: сервис фиксирует только то, что видно в кадре."
    : (job.notes || "В кадре не найдено действий, похожих на нарушения из перечня, который проверяет сервис.");

  els.findings.replaceChildren(...violations.map(renderFinding));
  drawMarks(violations);
  show(els.done);
}

function renderFinding(v, i) {
  const node = els.tpl.content.firstElementChild.cloneNode(true);
  node.dataset.index = String(i);
  const tc = node.querySelector(".tc");
  tc.textContent = fmtTime(v.timecode_sec);
  tc.addEventListener("click", () => seek(v.timecode_sec, i));
  node.querySelector(".finding__type").textContent = v.type;
  const conf = node.querySelector(".finding__conf");
  if (v.confidence) conf.textContent = `уверенность: ${v.confidence}`; else conf.remove();
  node.querySelector(".finding__desc").textContent = v.description;

  const frames = node.querySelector(".frames");
  for (const f of v.evidence_frames || []) {
    const fig = document.createElement("figure"); fig.className = "frame";
    const btn = document.createElement("button"); btn.type = "button";
    const img = document.createElement("img");
    img.src = "data:image/jpeg;base64," + f.image_base64;
    img.alt = `Кадр на ${fmtTime(f.timestamp_sec)}`;
    img.loading = "lazy";
    btn.append(img);
    btn.addEventListener("click", () => openFrame(img.src, `${v.type} — ${fmtTime(f.timestamp_sec)}`));
    const cap = document.createElement("figcaption"); cap.textContent = fmtTime(f.timestamp_sec);
    fig.append(btn, cap);
    frames.append(fig);
  }

  const norm = v.norm;
  const ref = node.querySelector(".norm__ref");
  const quote = node.querySelector(".norm__quote");
  if (norm) {
    const label = `${norm.document}${norm.clause ? ", " + norm.clause : ""}`;
    if (norm.url) {
      const a = document.createElement("a");
      a.href = norm.url; a.target = "_blank"; a.rel = "noopener"; a.textContent = label;
      ref.append(a);
    } else ref.textContent = label;
    if (norm.quote) quote.textContent = norm.quote; else quote.remove();
  } else node.querySelector(".norm").remove();
  return node;
}

function drawMarks(violations) {
  const dur = els.video.duration;
  if (!isFinite(dur) || dur <= 0) {
    els.video.addEventListener("loadedmetadata", () => drawMarks(violations), { once: true });
    return;
  }
  els.durLabel.textContent = fmtTime(dur).replace(/\.\d$/, "");
  els.marks.replaceChildren(...violations.map((v, i) => {
    const m = document.createElement("button");
    m.type = "button"; m.className = "mark";
    m.style.left = `${(v.timecode_sec / dur) * 100}%`;
    m.dataset.label = fmtTime(v.timecode_sec).replace(/\.\d$/, "");
    m.setAttribute("aria-label", `${v.type}, ${fmtTime(v.timecode_sec)}`);
    m.addEventListener("click", () => seek(v.timecode_sec, i));
    return m;
  }));
}

function seek(t, i) {
  els.video.currentTime = t;
  els.video.pause();
  els.findings.querySelectorAll(".finding").forEach((n) => n.classList.toggle("is-active", n.dataset.index === String(i)));
  const target = els.findings.querySelector(`[data-index="${i}"]`);
  if (target && window.matchMedia("(max-width: 820px)").matches) target.scrollIntoView({ behavior: "smooth", block: "start" });
}

function openFrame(src, caption) {
  els.lightboxImg.src = src;
  els.lightboxCap.textContent = caption;
  els.lightbox.showModal();
}

els.video.addEventListener("timeupdate", () => {
  const d = els.video.duration;
  if (isFinite(d) && d > 0) els.played.style.width = `${(els.video.currentTime / d) * 100}%`;
});
els.video.addEventListener("loadedmetadata", () => {
  els.durLabel.textContent = fmtTime(els.video.duration).replace(/\.\d$/, "");
});

// ---- reset / inputs -------------------------------------------------------------------------
function reset(toIdle = true) {
  clearTimeout(pollTimer);
  currentJob = null;
  if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
  els.video.removeAttribute("src"); els.video.load();
  els.marks.replaceChildren(); els.findings.replaceChildren();
  els.played.style.width = "0";
  if (toIdle) {
    els.drop.hidden = false; els.player.hidden = true; els.resetBtn.hidden = true;
    els.fileInput.value = "";
    show(els.idle);
  }
}

els.fileInput.addEventListener("change", () => pickFile(els.fileInput.files[0]));
els.resetBtn.addEventListener("click", () => reset());
els.retryBtn.addEventListener("click", () => { reset(); els.fileInput.click(); });
["dragenter", "dragover"].forEach((ev) => els.drop.addEventListener(ev, (e) => { e.preventDefault(); els.drop.classList.add("is-over"); }));
["dragleave", "drop"].forEach((ev) => els.drop.addEventListener(ev, (e) => { e.preventDefault(); els.drop.classList.remove("is-over"); }));
els.drop.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => { e.preventDefault(); if (!els.drop.hidden) pickFile(e.dataTransfer.files[0]); });
