// Runs the real app/static/app.js against the real index.html in Node, with small stand-ins for the DOM,
// XHR, fetch and timers (no browser, no server). Usage: node tests/frontend_harness.js <scenario>.
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const STATIC = path.join(__dirname, "..", "app", "static");
const log = { focused: null, scrolls: [], pickerOpened: 0 };

class El {
  constructor(tag) {
    Object.assign(this, { tagName: tag.toUpperCase(), id: "", children: [], parent: null, dataset: {}, style: {},
      attrs: {}, listeners: {}, hidden: false, text: "" });
    const cls = (this.classes = new Set());
    this.classList = {
      add: (...c) => c.forEach((x) => cls.add(x)),
      remove: (...c) => c.forEach((x) => cls.delete(x)),
      toggle: (c, on = !cls.has(c)) => ((on ? cls.add(c) : cls.delete(c)), on),
      contains: (c) => cls.has(c),
    };
  }
  set className(v) { this.classes.clear(); String(v).split(/\s+/).filter(Boolean).forEach((c) => this.classes.add(c)); }
  get className() { return [...this.classes].join(" "); }
  set textContent(v) { this.text = String(v); this.children = []; }
  get textContent() { return this.text + this.children.map((c) => c.textContent).join(""); }
  get firstElementChild() { return this.children[0] || null; }
  get offsetWidth() { return 0; }
  get content() { return this; }  // <template>
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }
  append(...cs) { for (const c of cs) { c.parent = this; this.children.push(c); } }
  replaceChildren(...cs) { this.children = []; this.text = ""; this.append(...cs); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this); this.parent = null; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  dispatch(type, { bubbles = true } = {}) {
    const ev = { type, target: this, preventDefault() {} };
    for (let el = this; el; el = bubbles ? el.parent : null) (el.listeners[type] || []).slice().forEach((fn) => fn(ev));
  }
  click() { if (this.attrs.type === "file") log.pickerOpened++; this.dispatch("click"); }
  focus() { log.focused = this; }
  scrollIntoView(opts) { log.scrolls.push({ el: this, opts }); }
  showModal() { this.open = true; }
  cloneNode(deep) {
    const c = new El(this.tagName);
    Object.assign(c, { id: this.id, className: this.className, dataset: { ...this.dataset }, attrs: { ...this.attrs }, text: this.text, hidden: this.hidden });
    if (deep) c.append(...this.children.map((x) => x.cloneNode(true)));
    return c;
  }
  *walk() { for (const c of this.children) { yield c; yield* c.walk(); } }
  matches(sel) {
    let m;
    if ((m = sel.match(/^\.([\w-]+)$/))) return this.classes.has(m[1]);
    if ((m = sel.match(/^\[data-index="(\d+)"\]$/))) return this.dataset.index === m[1];
    throw new Error(`selector not supported by the harness: ${sel}`);
  }
  querySelector(sel) { for (const e of this.walk()) if (e.matches(sel)) return e; return null; }
  querySelectorAll(sel) { return [...this.walk()].filter((e) => e.matches(sel)); }
}

class Video extends El {
  constructor() { super("video"); Object.assign(this, { duration: NaN, currentTime: 0, videoWidth: 0 }); }
  set src(v) { this.attrs.src = v; this.duration = NaN; }
  get src() { return this.attrs.src || ""; }
  load() { this.duration = NaN; }
  pause() {}
}

// Enough of an HTML parser for index.html: elements, attributes, nesting, void and self-closed tags.
function parseHtml(html) {
  const root = new El("#document");
  const VOID = new Set(["meta", "link", "input", "img", "br", "hr", "source"]);
  let cur = root;
  for (const [, close, open, attrs = "", selfClosed, text] of html.matchAll(
    /<!--[\s\S]*?-->|<!doctype[^>]*>|<\/(\w+)\s*>|<(\w+)((?:\s+[\w:-]+(?:="[^"]*")?)*)\s*(\/?)>|([^<]+)/gi)) {
    if (close) {
      for (let el = cur; el !== root; el = el.parent) if (el.tagName === close.toUpperCase()) { cur = el.parent; break; }
    } else if (open) {
      const el = open.toLowerCase() === "video" ? new Video() : new El(open);
      for (const [, k, v = ""] of attrs.matchAll(/([\w:-]+)(?:="([^"]*)")?/g)) {
        el.attrs[k] = v;
        if (k === "id") el.id = v;
        else if (k === "class") el.className = v;
        else if (k === "hidden") el.hidden = true;
      }
      cur.append(el);
      if (!VOID.has(open.toLowerCase()) && !selfClosed) cur = el;
    } else if (text && text.trim()) cur.text += text;
  }
  return root;
}

function boot({ narrow = false } = {}) {
  const doc = parseHtml(fs.readFileSync(path.join(STATIC, "index.html"), "utf8"));
  const byId = {};
  for (const el of doc.walk()) if (el.id) byId[el.id] = el;

  const clock = { now: 0, seq: 0, timers: new Map() };
  const setTimeout_ = (fn, ms = 0) => { clock.timers.set(++clock.seq, { at: clock.now + ms, fn }); return clock.seq; };
  const clearTimeout_ = (id) => clock.timers.delete(id);
  const settle = async () => { for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r)); };
  async function advance(ms) {
    const end = clock.now + ms;
    for (;;) {
      await settle();
      const due = [...clock.timers].filter(([, t]) => t.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
      if (!due) break;
      clock.timers.delete(due[0]);
      clock.now = due[1].at;
      due[1].fn();
    }
    clock.now = end;
    await settle();
  }

  // Requests stay pending until the scenario answers them; nothing settles them on abort, so a
  // scenario can also deliver a response after the abort and check that the page ignores it.
  const fetches = [];
  const fetch_ = (url, opts = {}) => new Promise((resolve, reject) => {
    const f = { url, signal: opts.signal, settled: false };
    f.reply = (status, body) => { f.settled = true; resolve({ ok: status >= 200 && status < 300, status, json: async () => body }); };
    f.fail = () => { f.settled = true; reject(new TypeError("Failed to fetch")); };
    fetches.push(f);
  });
  const xhrs = [];
  class XHR {
    constructor() { Object.assign(this, { upload: {}, status: 0, responseText: "", aborted: false }); xhrs.push(this); }
    open(method, url) { this.url = url; }
    send(body) { this.body = body; }
    abort() { this.aborted = true; }
    respond(status, body) { this.status = status; this.responseText = JSON.stringify(body); this.onload && this.onload(); }
  }
  const document = {
    getElementById: (id) => byId[id] || null,
    createElement: (tag) => (tag === "video" ? new Video() : new El(tag)),
    addEventListener() {},
  };
  let blobs = 0;
  const ctx = vm.createContext({
    document, fetch: fetch_, XMLHttpRequest: XHR, AbortController, console,
    window: { matchMedia: () => ({ matches: narrow }) },
    FormData: class { append() {} },
    URL: { createObjectURL: () => `blob:${++blobs}`, revokeObjectURL() {} },
    setTimeout: setTimeout_, clearTimeout: clearTimeout_,
  });
  vm.runInContext(fs.readFileSync(path.join(STATIC, "app.js"), "utf8"), ctx, { filename: "app.js" });

  const t = {
    ctx, byId, xhrs, fetches, advance, settle,
    polls: () => fetches.filter((f) => f.url.startsWith("/ui/jobs/")),
    lastPoll: () => t.polls().at(-1),
    visible: () => ["stateIdle", "stateWork", "stateFail", "stateDone"].filter((id) => !byId[id].hidden),
    pick(name, bytes = Buffer.alloc(16)) {
      byId.fileInput.files = [new File([bytes], name)];
      byId.fileInput.dispatch("change");
      return xhrs.at(-1);
    },
    async uploaded(xhr, jobId) { xhr.respond(200, { job_id: jobId, status: "processing" }); await settle(); },
    findingTypes: () => byId.findings.querySelectorAll(".finding").map((n) => n.querySelector(".finding__type").textContent),
    marks: () => byId.marks.children.map((m) => m.style.left),
    // what a browser fires once it has read the file header; width 0 means it found no video it can decode
    loadMetadata(sec, width = 464) {
      Object.assign(byId.video, { duration: sec, videoWidth: width });
      byId.video.dispatch("durationchange", { bubbles: false });
      byId.video.dispatch("loadedmetadata", { bubbles: false });
    },
    changeDuration(sec) { byId.video.duration = sec; byId.video.dispatch("durationchange", { bubbles: false }); },
  };
  return t;
}

const JPEG = "/9j/2Q==";
const doneJob = (id, n = 2, extra = {}) => ({
  job_id: id, status: "done", verdict: n ? "violation" : "no_violation", processing_sec: 12.3, ...extra,
  violations: Array.from({ length: n }, (_, i) => ({
    type: `${id}-type-${i}`, timecode_sec: 3 + i * 5, description: "d", confidence: "высокая",
    evidence_frames: [{ timestamp_sec: 3 + i * 5, image_base64: JPEG }],
    norm: { document: "Doc", clause: "п. 1", url: "https://adilet.zan.kz/x", quote: "q", interpretation: null },
  })),
});

// QuickTime file with the movie header after the media data, the way cameras write it.
function box(type, ...parts) {
  const body = Buffer.concat(parts);
  const head = Buffer.alloc(8);
  head.writeUInt32BE(8 + body.length);
  head.write(type, 4, "latin1");
  return Buffer.concat([head, body]);
}
function movBytes(seconds, { version = 0, scale = 600, moovFirst = false } = {}) {
  const mvhd = Buffer.alloc(version === 1 ? 112 : 100);
  mvhd[0] = version;
  if (version === 1) { mvhd.writeUInt32BE(scale, 20); mvhd.writeBigUInt64BE(BigInt(Math.round(seconds * scale)), 24); }
  else { mvhd.writeUInt32BE(scale, 12); mvhd.writeUInt32BE(Math.round(seconds * scale), 16); }
  const moov = box("moov", box("mvhd", mvhd), box("trak", Buffer.alloc(24)));
  const ftyp = box("ftyp", Buffer.from("qt  \0\0\0\0qt  ", "latin1"));
  const mdat = box("mdat", Buffer.alloc(4096, 7));
  return moovFirst ? Buffer.concat([ftyp, moov, mdat]) : Buffer.concat([ftyp, box("wide"), mdat, moov]);
}
const pct = (t, dur) => `${(t / dur) * 100}%`;

const scenarios = {
  async fmtTimeRollsOver() {
    const { ctx } = boot();
    assert.equal(ctx.fmtTime(59.96), "1:00.0");
    assert.equal(ctx.fmtTime(119.97), "2:00.0");
    assert.equal(ctx.fmtTime(299.97), "5:00.0");
    assert.equal(ctx.fmtTime(59.94), "0:59.9");
    assert.equal(ctx.fmtTime(61.3), "1:01.3");
    assert.equal(ctx.fmtTime(179.9), "2:59.9");
    assert.equal(ctx.fmtTime(0), "0:00.0");
  },

  async resetAbortsUpload() {
    const t = boot();
    const a = t.pick("a.mov");
    t.byId.resetBtn.click();
    assert.ok(a.aborted, "upload not aborted");
    await t.uploaded(a, "jobA");  // a load event that slipped through must change nothing
    await t.advance(30000);
    assert.equal(t.polls().length, 0);
    assert.deepEqual(t.visible(), ["stateIdle"]);
    assert.equal(t.byId.drop.hidden, false);
    assert.equal(log.focused, t.byId.pickBtn, "focus not returned to the file picker");
  },

  async abandonedUploadCannotHijackNextFile() {
    const t = boot();
    const a = t.pick("a.mov");
    t.byId.resetBtn.click();
    const b = t.pick("b.mov");
    a.upload.onprogress({ lengthComputable: true, loaded: 80, total: 100 });
    assert.equal(t.byId.workStage.textContent, "Загрузка видео на сервер — 0%");
    await t.uploaded(b, "jobB");
    await t.uploaded(a, "jobA");
    assert.deepEqual(t.polls().map((f) => f.url), ["/ui/jobs/jobB"]);
    t.lastPoll().reply(200, doneJob("jobB"));
    await t.advance(5000);
    assert.deepEqual(t.visible(), ["stateDone"]);
    assert.deepEqual(t.findingTypes(), ["jobB-type-0", "jobB-type-1"]);
    assert.equal(t.byId.verdictMeta.children[1].textContent, "b.mov");
    assert.equal(t.polls().length, 1);
  },

  async resetStopsPolling() {
    const t = boot();
    await t.uploaded(t.pick("a.mov"), "jobA");
    let inFlight = t.lastPoll();
    t.byId.resetBtn.click();
    assert.ok(inFlight.signal.aborted, "status request not aborted");
    inFlight.fail();  // how fetch ends once its signal is aborted
    await t.advance(60000);
    assert.equal(t.polls().length, 1, "polling went on after reset");

    await t.uploaded(t.pick("b.mov"), "jobB");
    inFlight = t.lastPoll();
    t.byId.resetBtn.click();
    inFlight.reply(200, doneJob("jobB"));  // an answer that was already on its way
    await t.advance(60000);
    assert.equal(t.polls().length, 2, "polling went on after reset");
    assert.deepEqual(t.visible(), ["stateIdle"]);
    assert.equal(t.byId.findings.children.length, 0);

    await t.uploaded(t.pick("c.mov"), "jobC");
    t.lastPoll().reply(200, { job_id: "jobC", status: "processing", stage: "analyzing", progress: 0.5 });
    await t.advance(1000);
    assert.deepEqual(t.polls().slice(2).map((f) => f.url), ["/ui/jobs/jobC", "/ui/jobs/jobC"]);
    assert.equal(t.byId.workStage.textContent, "Анализ видео — 50%");
  },

  async pollHttpErrorsSurface() {
    const t = boot();
    await t.uploaded(t.pick("a.mov"), "jobA");
    for (let i = 0; i < 6; i++) {
      assert.deepEqual(t.visible(), ["stateWork"]);
      t.lastPoll().reply(500, { detail: "internal error" });
      await t.settle();
      if (i === 0) assert.match(t.byId.workStage.textContent, /кодом 500 — повторяем/);
      await t.advance(20000);
    }
    assert.deepEqual(t.visible(), ["stateFail"]);
    assert.match(t.byId.failText.textContent, /Сервер ответил кодом 500/);
    await t.advance(120000);
    assert.equal(t.polls().length, 6, "polling went on after the error was shown");
  },

  async pollNetworkErrorsSurface() {
    const t = boot();
    await t.uploaded(t.pick("a.mov"), "jobA");
    for (let i = 0; i < 6; i++) { t.lastPoll().fail(); await t.advance(20000); }
    assert.deepEqual(t.visible(), ["stateFail"]);
    assert.match(t.byId.failText.textContent, /Сервер недоступен/);
  },

  async pollRecoversAndResetsRetries() {
    const t = boot();
    await t.uploaded(t.pick("a.mov"), "jobA");
    for (let i = 0; i < 3; i++) { t.lastPoll().reply(502, {}); await t.advance(20000); }
    t.lastPoll().reply(200, { job_id: "jobA", status: "processing", stage: "detecting", progress: 0.4 });
    await t.advance(1000);
    for (let i = 0; i < 5; i++) { t.lastPoll().fail(); await t.advance(20000); }
    assert.deepEqual(t.visible(), ["stateWork"], "retry budget was not reset by a good answer");
    t.lastPoll().reply(200, doneJob("jobA"));
    await t.settle();
    assert.deepEqual(t.visible(), ["stateDone"]);
  },

  async headerDuration() {
    const { ctx } = boot();
    const dur = (bytes) => ctx.headerDuration(new File([bytes], "v.mov")).catch(() => -1);
    assert.equal(await dur(movBytes(42)), 42);
    assert.equal(await dur(movBytes(26.43, { moovFirst: true })), 26.43);
    assert.equal(await dur(movBytes(299.5, { version: 1, scale: 30000 })), 299.5);
    assert.equal(await dur(Buffer.from("RIFF\x10\0\0\0AVI LIST", "latin1")), 0);
    assert.equal(await dur(Buffer.alloc(0)), 0);
    assert.ok(await dur(movBytes(42).subarray(0, 4150)) <= 0, "truncated file must not give a duration");
  },

  async undecodableVideoStillGetsTimeline() {
    const t = boot();
    const xhr = t.pick("cam.mov", movBytes(42));
    t.byId.video.dispatch("error", { bubbles: false });  // the browser cannot decode the codec
    assert.equal(t.byId.videoNote.hidden, false);
    await t.uploaded(xhr, "j1");
    t.lastPoll().reply(200, doneJob("j1"));
    await t.settle();
    assert.deepEqual(t.visible(), ["stateDone"]);
    assert.equal(t.byId.durLabel.textContent, "0:42");
    assert.deepEqual(t.marks(), [pct(3, 42), pct(8, 42)]);

    t.byId.resetBtn.click();
    assert.equal(t.byId.videoNote.hidden, true);
    assert.equal(t.byId.durLabel.textContent, "");

    t.pick("phone.mov");
    t.loadMetadata(30, 0);  // audio plays, the video track does not
    assert.equal(t.byId.videoNote.hidden, false);
    t.pick("ok.mov");
    t.loadMetadata(30);
    assert.equal(t.byId.videoNote.hidden, true);
  },

  async durationFromResult() {
    const t = boot();
    await t.uploaded(t.pick("cam.avi", Buffer.from("RIFF\x10\0\0\0AVI LIST", "latin1")), "j1");
    t.lastPoll().reply(200, doneJob("j1", 2, { video: { duration_sec: 50 } }));
    await t.settle();
    assert.equal(t.byId.durLabel.textContent, "0:50");
    assert.deepEqual(t.marks(), [pct(3, 50), pct(8, 50)]);
    t.loadMetadata(Infinity);  // a player that reports Infinity does not override a known duration
    assert.equal(t.byId.durLabel.textContent, "0:50");
    t.changeDuration(40);  // the player's own duration wins once it is known
    assert.deepEqual(t.marks(), [pct(3, 40), pct(8, 40)]);
  },

  async noStaleMarksAfterReset() {
    const t = boot();
    await t.uploaded(t.pick("a.avi"), "jobA");
    t.lastPoll().reply(200, doneJob("jobA"));
    await t.settle();
    assert.deepEqual(t.marks(), [], "no duration known yet");
    t.byId.resetBtn.click();
    const b = t.pick("b.mov");
    t.loadMetadata(60);
    assert.deepEqual(t.marks(), [], "marks of the previous result drawn on the next video");
    assert.equal(t.byId.durLabel.textContent, "1:00");
    b.respond(500, { detail: "boom" });
    assert.deepEqual(t.visible(), ["stateFail"]);
    assert.deepEqual(t.marks(), []);
  },

  async mobileTimecodeKeepsVideoInView() {
    const t = boot({ narrow: true });
    await t.uploaded(t.pick("a.mov"), "jobA");
    t.loadMetadata(60);
    t.lastPoll().reply(200, doneJob("jobA"));
    await t.settle();
    const second = t.byId.findings.querySelector('[data-index="1"]');
    second.querySelector(".tc").click();
    assert.equal(t.byId.video.currentTime, 8);
    assert.equal(log.scrolls.at(-1).el, t.byId.player, "timecode tap must bring the video into view");
    assert.ok(second.classes.has("is-active"));
    t.byId.marks.children[0].click();
    assert.equal(log.scrolls.at(-1).el, t.byId.findings.querySelector('[data-index="0"]'));

    const wide = boot();
    await wide.uploaded(wide.pick("a.mov"), "jobA");
    wide.lastPoll().reply(200, doneJob("jobA"));
    await wide.settle();
    const before = log.scrolls.length;
    wide.byId.findings.querySelector(".tc").click();
    assert.equal(log.scrolls.length, before, "the sticky desktop player needs no scrolling");
  },

  async filePickerWorksFromKeyboard() {
    const t = boot();
    const btn = t.byId.pickBtn;
    assert.equal(btn.tagName, "BUTTON");
    assert.equal(btn.attrs.type, "button");
    for (let el = btn; el; el = el.parent) {
      assert.ok(!el.hidden, `${el.tagName}#${el.id} hides the picker button on the start screen`);
      assert.notEqual(el.tagName, "LABEL", "a button inside a label is invalid and confuses screen readers");
    }
    btn.click();  // Enter and Space on a button fire click
    assert.equal(log.pickerOpened, 1);
    t.byId.drop.click();  // the whole drop zone still opens the picker for the mouse
    assert.equal(log.pickerOpened, 2);
    t.pick("a.mov");
    t.byId.resetBtn.click();
    assert.equal(log.focused, btn);
  },
};

const name = process.argv[2];
if (!scenarios[name]) {
  console.error(`unknown scenario ${name}; known: ${Object.keys(scenarios).join(", ")}`);
  process.exit(2);
}
scenarios[name]().then(() => console.log(`ok ${name}`), (err) => { console.error(err); process.exit(1); });
