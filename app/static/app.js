"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  drop: $("drop"), pickBtn: $("pickBtn"), fileInput: $("fileInput"), player: $("player"), video: $("video"),
  videoNote: $("videoNote"), marks: $("marks"), played: $("played"), durLabel: $("durLabel"),
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
const POLL_RETRIES = 5;  // failed status requests in a row (1, 2, 4, 8, 16 s apart) before the error is shown

let run = new AbortController();  // one per picked file: reset() aborts it, so a late upload or poll response is dropped
let pollTimer = null;
let objectUrl = null;
let shown = null;   // the job whose result is on screen
let fileDur = 0;    // duration from the file header, for a video the browser cannot decode

function show(state) {
  for (const el of [els.idle, els.work, els.fail, els.done]) el.hidden = el !== state;
}

function fmtTime(sec) {
  const s = Math.round(Math.max(0, sec) * 10) / 10;  // round first: 59.96 s is 1:00.0, not 0:60.0
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
  const { signal } = run;
  els.drop.hidden = true;
  els.player.hidden = false;
  els.resetBtn.hidden = false;
  objectUrl = URL.createObjectURL(file);
  els.video.src = objectUrl;
  els.workFile.textContent = file.name;
  show(els.work);
  setMeter(0, "Загрузка видео на сервер — 0%");
  headerDuration(file).then((d) => { if (!signal.aborted && d > 0) { fileDur = d; drawTimeline(); } }, () => {});

  const form = new FormData();
  form.append("file", file);
  const xhr = new XMLHttpRequest();
  signal.addEventListener("abort", () => xhr.abort());
  xhr.open("POST", "/ui/analyze");
  xhr.upload.onprogress = (e) => {
    if (!signal.aborted && e.lengthComputable) setMeter(e.loaded / e.total * 0.15, `Загрузка видео на сервер — ${Math.round(e.loaded / e.total * 100)}%`);
  };
  xhr.onload = () => {
    if (signal.aborted) return;
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch (_) {}
    if (xhr.status >= 200 && xhr.status < 300 && body.job_id) {
      setMeter(0.15, STAGES.queued);
      poll(body.job_id, signal);
    } else {
      fail(body.detail || `Сервер ответил кодом ${xhr.status}. Проверьте, что файл — видео в формате .mov.`);
    }
  };
  xhr.onerror = () => {
    if (!signal.aborted) fail("Не удалось отправить файл: сервер недоступен. Проверьте подключение и попробуйте ещё раз.");
  };
  xhr.send(form);
}

function poll(jobId, signal, failures = 0) {
  fetch(`/ui/jobs/${jobId}`, { signal }).then((r) => (r.ok ? r.json() : Promise.reject(r.status))).then((job) => {
    if (signal.aborted) return;
    if (job.status === "done") return render(job);
    if (job.status === "failed") return fail(`Анализ остановился с ошибкой: ${job.error || "неизвестная ошибка"}. Попробуйте загрузить файл ещё раз.`);
    let text = STAGES[job.stage] || STAGES.analyzing;
    if (job.stage === "queued" && job.queue_position > 0) text += ` — перед вами ${job.queue_position}`;
    else if (job.progress) text += ` — ${Math.round(job.progress * 100)}%`;
    setMeter(0.15 + 0.85 * (job.progress || 0), text);
    pollTimer = setTimeout(() => poll(jobId, signal), 1000);
  }, (err) => {
    if (signal.aborted) return;
    const why = typeof err === "number" ? `Сервер ответил кодом ${err}` : "Сервер недоступен";
    if (failures >= POLL_RETRIES) return fail(`Не удалось получить результат анализа. ${why}. Попробуйте загрузить файл ещё раз.`);
    els.workStage.textContent = `${why} — повторяем запрос…`;
    pollTimer = setTimeout(() => poll(jobId, signal, failures + 1), 1000 * 2 ** failures);
  });
}

function fail(text) {
  els.failText.textContent = text;
  show(els.fail);
}

// ---- result ---------------------------------------------------------------------------------
function render(job) {
  shown = job;
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
  drawTimeline();
  show(els.done);
}

function renderFinding(v, i) {
  const node = els.tpl.content.firstElementChild.cloneNode(true);
  node.dataset.index = String(i);
  const tc = node.querySelector(".tc");
  tc.textContent = fmtTime(v.timecode_sec);
  tc.addEventListener("click", () => seek(v.timecode_sec, i, true));
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
    if (norm.quote) quote.textContent = `«${norm.quote}»`; else quote.remove();
    const note = node.querySelector(".norm__note");
    if (norm.interpretation) note.textContent = `${norm.strength === "interpretation" ? "Толкование" : "Пояснение"}: ${norm.interpretation}`; else note.remove();
  } else node.querySelector(".norm").remove();
  return node;
}

// Redrawn whenever the duration becomes known. Without the player's own duration (a file the browser
// cannot decode) the scale comes from the result, then from the file header.
function drawTimeline() {
  const d = els.video.duration;
  const dur = isFinite(d) && d > 0 ? d : shown?.video?.duration_sec || fileDur;
  els.durLabel.textContent = dur ? fmtTime(dur).replace(/\.\d$/, "") : "";
  els.marks.replaceChildren(...(dur && shown?.violations || []).map((v, i) => {
    const m = document.createElement("button");
    m.type = "button"; m.className = "mark";
    m.style.left = `${Math.min(100, (v.timecode_sec / dur) * 100)}%`;
    m.dataset.label = fmtTime(v.timecode_sec).replace(/\.\d$/, "");
    m.setAttribute("aria-label", `${v.type}, ${fmtTime(v.timecode_sec)}`);
    m.addEventListener("click", () => seek(v.timecode_sec, i));
    return m;
  }));
}

// Movie duration from the QuickTime/MP4 header (moov > mvhd); 0 when the file has none.
async function headerDuration(file) {
  const read = async (pos, len) => new DataView(await file.slice(pos, pos + len).arrayBuffer());
  const u64 = (v, at) => v.getUint32(at) * 2 ** 32 + v.getUint32(at + 4);
  let pos = 0, end = file.size;
  for (let n = 0; n < 100 && pos + 8 <= end; n++) {
    const h = await read(pos, 16);
    let size = h.getUint32(0), head = 8;
    if (size === 1) { size = u64(h, 8); head = 16; } else if (size === 0) size = end - pos;
    if (size < head) return 0;
    const type = String.fromCharCode(h.getUint8(4), h.getUint8(5), h.getUint8(6), h.getUint8(7));
    if (type === "moov") { end = pos + size; pos += head; continue; }
    if (type === "mvhd") {
      const v = await read(pos + head, 32);
      const long = v.getUint8(0) === 1;
      const scale = v.getUint32(long ? 20 : 12);
      return scale ? (long ? u64(v, 24) : v.getUint32(16)) / scale : 0;
    }
    pos += size;
  }
  return 0;
}

// On a phone the player is above the report: a timecode tap brings the video into view,
// a tap on a timeline mark brings its finding.
function seek(t, i, toVideo = false) {
  els.video.currentTime = t;
  els.video.pause();
  els.findings.querySelectorAll(".finding").forEach((n) => n.classList.toggle("is-active", n.dataset.index === String(i)));
  if (!window.matchMedia("(max-width: 820px)").matches) return;
  const target = toVideo ? els.player : els.findings.querySelector(`[data-index="${i}"]`);
  if (target) target.scrollIntoView({ behavior: "smooth", block: toVideo ? "nearest" : "start" });
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
els.video.addEventListener("durationchange", drawTimeline);
// the server reads more codecs than browsers do: explain the dead player instead of leaving it black
els.video.addEventListener("loadedmetadata", () => { if (!els.video.videoWidth) els.videoNote.hidden = false; });
els.video.addEventListener("error", () => { if (els.video.getAttribute("src")) els.videoNote.hidden = false; });

// ---- reset / inputs -------------------------------------------------------------------------
function reset(toIdle = true) {
  run.abort();
  run = new AbortController();
  clearTimeout(pollTimer);
  shown = null; fileDur = 0;
  if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
  els.video.removeAttribute("src"); els.video.load();
  els.marks.replaceChildren(); els.findings.replaceChildren();
  els.played.style.width = "0";
  els.durLabel.textContent = "";
  els.videoNote.hidden = true;
  if (toIdle) {
    els.drop.hidden = false; els.player.hidden = true; els.resetBtn.hidden = true;
    els.fileInput.value = "";
    show(els.idle);
    els.pickBtn.focus();
  }
}

els.fileInput.addEventListener("change", () => pickFile(els.fileInput.files[0]));
els.drop.addEventListener("click", () => els.fileInput.click());
els.resetBtn.addEventListener("click", () => reset());
els.retryBtn.addEventListener("click", () => { reset(); els.fileInput.click(); });
["dragenter", "dragover"].forEach((ev) => els.drop.addEventListener(ev, (e) => { e.preventDefault(); els.drop.classList.add("is-over"); }));
["dragleave", "drop"].forEach((ev) => els.drop.addEventListener(ev, (e) => { e.preventDefault(); els.drop.classList.remove("is-over"); }));
els.drop.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => { e.preventDefault(); if (!els.drop.hidden) pickFile(e.dataTransfer.files[0]); });
