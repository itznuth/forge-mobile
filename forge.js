/* F.O.R.G.E. Mobile — on-device tool detection + guidance for iPhone Safari.
 *
 * Pipeline: iPhone camera -> object detector -> tracker -> "in view" list / lock-on guidance / voice
 *
 * Detectors (first one that loads wins):
 *   1. model/best.onnx  — your own custom YOLOv11 (e.g. the FORGE tool model), via onnxruntime-web
 *   2. COCO-SSD         — pretrained, detects 80 everyday object types, no training needed
 *   3. simulation       — fake detections (FORGE_SIM=1 equivalent)
 *
 * Mirrors the original Pi runtime:
 *   laser dot + pan-tilt   -> on-screen laser reticle + direction arrow
 *   Vosk wake-word voice   -> Web Speech API (Safari)
 *   Piper / eSpeak TTS     -> speechSynthesis
 *   SQLite event tables    -> localStorage (tool_events, command_events)
 */
(() => {
  "use strict";

  // ---------- config ----------
  const CONFIG = {
    modelUrl: "model/best.onnx",
    classesUrl: "model/classes.json",
    cocoBase: "lite_mobilenet_v2",  // fastest COCO-SSD variant; "mobilenet_v2" is more accurate but slower
    inputSize: 640,
    confThreshold: 0.5,
    iouThreshold: 0.45,
    presentTimeoutMs: 2500,   // tool counts as "on bench" if seen within this window
    speakCooldownMs: 3500,
    maxLogRows: 200,
    recentMs: 60000,          // how long a vanished object stays in the "In view" list / chips
  };

  const COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear",
    "zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
    "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"];

  // spoken/typed words -> detector class names
  const ALIASES = {
    "phone": "cell phone", "iphone": "cell phone", "cellphone": "cell phone", "mobile": "cell phone",
    "sofa": "couch", "television": "tv", "monitor": "tv", "computer": "laptop", "macbook": "laptop",
    "mug": "cup", "glass": "cup", "plant": "potted plant", "table": "dining table", "desk": "dining table",
    "fridge": "refrigerator", "bag": "backpack", "purse": "handbag", "ball": "sports ball", "controller": "remote",
    "hair dryer": "hair drier", "hairdryer": "hair drier", "bike": "bicycle", "motorbike": "motorcycle",
    "water bottle": "bottle", "people": "person", "someone": "person", "me": "person", "doughnut": "donut",
    "screw driver": "screwdriver", "plier": "pliers",
    // FORGE workbench model (model/classes.json)
    "allen key": "allen_keys", "allen wrench": "allen_keys", "hex key": "allen_keys", "hex wrench": "allen_keys",
    "saw": "handsaw", "hand saw": "handsaw", "tape": "tape_measure", "measuring tape": "tape_measure",
    "tape measurer": "tape_measure", "spanner": "wrench",
  };

  // ---------- dom ----------
  const $ = (id) => document.getElementById(id);
  const video = $("video"), overlay = $("overlay"), octx = overlay.getContext("2d");
  const ui = {
    mode: $("modePill"), fps: $("fpsPill"), lock: $("lockPill"), say: $("say"),
    inv: $("inventory"), log: $("log"), tools: $("toolButtons"), mic: $("micBtn"),
    finds: $("statFinds"), cmds: $("statCmds"), inf: $("statInf"),
  };

  // ---------- state ----------
  const state = {
    classes: COCO.slice(),  // vocabulary the active detector knows
    backend: "none",        // "onnx" | "coco" | "sim"
    sim: false,
    session: null,
    customClasses: null,    // class list of model/best.onnx, when it loaded
    switching: false,
    coco: null,
    running: false,
    locked: null,          // tool name or null (auto mode)
    detections: [],
    tools: {},             // name -> {lastSeen, conf, box, present, count}
    chipKey: "",
    lastSpoken: 0,
    lastFoundAnnounce: {},
    finds: 0, cmds: 0,
  };

  // ---------- persistence (SQLite-style tables in localStorage) ----------
  const DB = {
    read(table) { try { return JSON.parse(localStorage.getItem("forge_" + table) || "[]"); } catch { return []; } },
    write(table, rows) { try { localStorage.setItem("forge_" + table, JSON.stringify(rows.slice(-CONFIG.maxLogRows))); } catch {} },
    insert(table, row) { const rows = DB.read(table); rows.push({ id: Date.now() + Math.random(), timestamp: new Date().toISOString(), ...row }); DB.write(table, rows); },
    clear() { ["tool_events", "command_events"].forEach((t) => { try { localStorage.removeItem("forge_" + t); } catch {} }); },
  };

  const esc = (x) => String(x).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // text is escaped; pass trusted markup via the html argument only
  function log(kind, text, html = "") {
    const div = document.createElement("div");
    const t = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    div.innerHTML = `${t} <b>${esc(kind)}</b> ${esc(text)}${html}`;
    ui.log.prepend(div);
    while (ui.log.children.length > 80) ui.log.lastChild.remove();
  }

  function restoreLog() {
    const rows = [
      ...DB.read("tool_events").map((r) => ({ ...r, kind: "tool" })),
      ...DB.read("command_events").map((r) => ({ ...r, kind: "cmd" })),
    ].filter((r) => typeof r.timestamp === "string").sort((a, b) => a.timestamp.localeCompare(b.timestamp)).slice(-40);
    for (const r of rows) {
      const div = document.createElement("div");
      const t = new Date(r.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        div.innerHTML = r.kind === "tool" ? `${t} <b>tool</b> ${esc(r.tool_name)} → ${esc(r.state)}` : `${t} <b>cmd</b> ${esc(r.command_text)}`;
      ui.log.prepend(div);
    }
  }

  // ---------- speech out ----------
  let speechUnlocked = false;
  let currentUtterance = null; // keep a reference: iOS can GC an utterance mid-speech and drop it
  let speechMuted = false;      // true while the mic is listening (iOS can't record and speak at once)

  function say(text, force = false) {
    ui.say.textContent = text;
    if (!("speechSynthesis" in window) || speechMuted) return;
    const now = performance.now();
    if (!force && now - state.lastSpoken < CONFIG.speakCooldownMs) return;
    state.lastSpoken = now;
    try {
      // iOS Safari: cancel() immediately followed by speak() often swallows the new utterance,
      // so only cancel when something is actually queued.
      // (the silent unlock utterance isn't tracked, so it never triggers a cancel)
      if (currentUtterance && (speechSynthesis.speaking || speechSynthesis.pending)) speechSynthesis.cancel();
      if (speechSynthesis.paused) speechSynthesis.resume();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = "en-US";
      u.rate = 1.05;
      u.onend = u.onerror = () => { if (currentUtterance === u) currentUtterance = null; };
      currentUtterance = u;
      speechSynthesis.speak(u);
      speechUnlocked = true; // a real speak() inside a tap unlocks iOS speech too
    } catch {}
  }

  // Must run synchronously inside a user tap. iOS only allows speech after a speak() call
  // made directly from a gesture handler; anything after an await no longer counts.
  function unlockSpeech() {
    if (speechUnlocked || !("speechSynthesis" in window)) return;
    speechUnlocked = true;
    try {
      const u = new SpeechSynthesisUtterance(" ");
      u.volume = 0;
      speechSynthesis.speak(u);
    } catch {}
  }

  // ---------- UI build ----------
  const pretty = (n) => n.replace(/_/g, " ");

  function tool(name) {
    return (state.tools[name] = state.tools[name] || { lastSeen: 0, conf: 0, box: null, present: false, count: 0 });
  }

  // objects seen in the last minute, most recent first
  function recentObjects() {
    const now = performance.now();
    return Object.entries(state.tools)
      .filter(([, t]) => t.lastSeen && now - t.lastSeen < CONFIG.recentMs)
      .sort((a, b) => (b[1].present - a[1].present) || (b[1].lastSeen - a[1].lastSeen))
      .map(([n]) => n);
  }

  function buildToolUI() { state.chipKey = ""; refreshUI(); }

  function refreshUI() {
    const now = performance.now();
    const recent = recentObjects();
    if (state.locked && !recent.includes(state.locked)) recent.unshift(state.locked);

    // chips: one tap to find anything currently/recently seen
    const key = recent.join("|") + "#" + state.locked;
    if (key !== state.chipKey) {
      state.chipKey = key;
      ui.tools.innerHTML = "";
      if (!recent.length) ui.tools.innerHTML = '<span class="hint">Objects FORGE sees will appear here. Tap one to find it.</span>';
      for (const name of recent) {
        const b = document.createElement("button");
        b.textContent = pretty(name);
        b.classList.toggle("active", name === state.locked);
        b.onclick = () => command(`find ${pretty(name)}`, "button");
        ui.tools.appendChild(b);
      }
    }

    // in-view list
    ui.inv.innerHTML = recent.length ? "" : '<span class="meta">Nothing detected yet.</span>';
    for (const name of recent) {
      const t = tool(name);
      const row = document.createElement("div");
      row.className = "tool";
      const meta = t.present
        ? `${t.count > 1 ? t.count + "× · " : ""}${(t.conf * 100).toFixed(0)}%`
        : t.lastSeen ? `gone · ${Math.round((now - t.lastSeen) / 1000)}s ago` : "not seen yet";
      row.innerHTML = `<span><span class="dot${t.present ? " on" : ""}"></span>${esc(pretty(name))}</span><span class="meta">${esc(meta)}</span>`;
      ui.inv.appendChild(row);
    }

    ui.lock.textContent = state.locked ? "LOCK: " + pretty(state.locked).toUpperCase() : "AUTO";
    ui.finds.textContent = state.finds;
    ui.cmds.textContent = state.cmds;
  }

  // ---------- commands (shared by buttons + voice) ----------
  const escapeRe = (x) => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const hasWord = (text, w) => new RegExp(`\\b${escapeRe(w)}(e?s)?\\b`).test(text);

  function matchTool(text) {
    const t = " " + text.toLowerCase().replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ") + " ";
    const vocab = state.classes.map((n) => [pretty(n), n]);
    const aliases = Object.entries(ALIASES).filter(([, n]) => state.classes.includes(n));
    // longest phrases first so "cell phone" beats "phone", "wine glass" beats "glass"
    const all = [...vocab, ...aliases].sort((a, b) => b[0].length - a[0].length);
    for (const [word, name] of all) if (hasWord(t, word)) return name;
    return null;
  }

  // the object the user asked for, even if the detector can't recognize it
  function askedFor(text) {
    const m = text.toLowerCase().match(/^(?:find|locate|where(?:'s| is| are)?|lock(?: on)?|show(?: me)?|track|get)\s+(?:my |the |a |an |me )*(.+?)\??$/);
    return m ? m[1].trim() : null;
  }

  function command(raw, source) {
    const text = raw.toLowerCase().replace(/^\s*(hey\s+)?forge[,\s]*/, "").trim();
    let ok = true, msg = "";
    const tool = matchTool(text);

    if (/^(find|locate|where|lock|show|track|get)\b/.test(text) || (tool && text.split(" ").length <= 3)) {
      if (!tool) {
        ok = false;
        const what = askedFor(text);
        msg = what
          ? `I can't recognize "${what}" yet. ${state.backend === "onnx" ? "Tools mode only knows workbench tools. Tap the mode button to switch to everyday objects." : "It's not one of the 80 object types I know."}${state.backend === "coco" && state.customClasses ? " Tap the mode button for workbench tools." : ""}`
          : "I didn't catch what to find.";
      } else {
        state.locked = tool;
        state.lastFoundAnnounce[tool] = 0;
        msg = tool && state.tools[tool] && state.tools[tool].present
          ? `Locking on ${pretty(tool)}.` : `Searching for ${pretty(tool)}. Pan the camera around.`;
      }
    } else if (/unlock|release|stop|cancel/.test(text)) {
      state.locked = null; msg = "Released. Back to auto mode.";
    } else if (/auto/.test(text)) {
      state.locked = null; msg = "Auto mode.";
    } else if (/what do you see|what.*see|scan/.test(text)) {
      const seen = Object.keys(state.tools).filter((n) => state.tools[n].present);
      msg = seen.length ? `I see ${listJoin(seen.map((n) => countName(n)))}.` : "I don't see anything I recognize right now.";
    } else if (/status|inventory|missing/.test(text)) {
      const now = performance.now();
      const gone = Object.keys(state.tools).filter((n) => !state.tools[n].present && state.tools[n].lastSeen > 0 && now - state.tools[n].lastSeen < CONFIG.recentMs);
      const seen = Object.keys(state.tools).filter((n) => state.tools[n].present);
      msg = `${seen.length} object type${seen.length === 1 ? "" : "s"} in view.` + (gone.length ? ` Recently gone: ${listJoin(gone.map(pretty))}.` : "");
    } else {
      ok = false; msg = `Unknown command: "${raw}". Try "find cup" or "what do you see".`;
    }

    state.cmds++;
    DB.insert("command_events", { command_type: source, command_text: raw, target_tool: tool, success: ok, message: msg });
    log("cmd", raw, ` <span>(${esc(source)})</span>`);
    say(msg, true);
    refreshUI();
  }

  const countName = (n) => { const c = state.tools[n].count; return c > 1 ? `${c} ${pretty(n)}s` : `a ${pretty(n)}`; };
  const listJoin = (a) => a.length < 2 ? a.join("") : a.slice(0, -1).join(", ") + " and " + a[a.length - 1];

  // ---------- voice in ----------
  function setupVoice() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      ui.mic.onclick = () => { say("Voice input isn't supported in this browser. Use the Find box or the buttons.", true); };
      return;
    }
    let rec = null, listening = false, heard = null, watchdog = 0;
    const resetMic = () => {
      clearTimeout(watchdog);
      listening = false; rec = null; speechMuted = false;
      ui.mic.classList.remove("listening"); ui.mic.textContent = "🎙 Voice";
    };
    ui.mic.onclick = () => {
      unlockSpeech();
      if (listening) { try { rec && rec.stop(); } catch {} return; }
      // stop TTS first: iOS shares one audio session, and speaking while recording
      // either cuts the recognizer off or gets transcribed as the command
      try { if ("speechSynthesis" in window) speechSynthesis.cancel(); } catch {}
      heard = null;
      const me = (rec = new SR());
      rec.lang = "en-US";
      rec.continuous = false;
      rec.interimResults = false;
      rec.maxAlternatives = 3;
      // set state now, not in onstart: a quick double tap would otherwise start a second recognizer
      listening = true; speechMuted = true;
      ui.mic.classList.add("listening"); ui.mic.textContent = "● Listening";
      rec.onresult = (e) => {
        if (rec !== me) return;
        const r = e.results[e.results.length - 1];
        if (!r) return;
        const alts = Array.from(r).map((a) => a.transcript.trim()).filter(Boolean);
        if (!alts.length) return;
        heard = alts.find((a) => matchTool(a)) || alts.find((a) => /unlock|auto|see|status|stop/.test(a.toLowerCase())) || alts[0];
        try { rec.stop(); } catch {}
      };
      rec.onerror = (e) => {
        if (rec !== me) return;
        const err = e.error;
        resetMic();
        if (err === "not-allowed" || err === "service-not-allowed")
          say("Voice is blocked. Allow the microphone for this site, and on iPhone turn on Siri and Dictation in Settings.", true);
        else if (err !== "no-speech" && err !== "aborted") say("Voice error: " + err, true);
      };
      // run the command after recognition ends so the spoken reply isn't recorded or muted
      rec.onend = () => {
        if (rec !== me) return;
        const cmd = heard;
        heard = null;
        resetMic();
        if (cmd) command(cmd, "voice");
      };
      // iOS sometimes never fires onend; don't leave the button stuck on "Listening"
      watchdog = setTimeout(() => { if (rec !== me) return; try { me.abort(); } catch {} const cmd = heard; heard = null; resetMic(); if (cmd) command(cmd, "voice"); }, 12000);
      try { rec.start(); } catch (err) { resetMic(); say("Couldn't start the microphone.", true); }
    };
  }

  // ---------- model ----------
  async function loadCustomModel() {
    if (!window.ort) throw new Error("onnxruntime-web not loaded");
    const head = await fetch(CONFIG.modelUrl, { method: "HEAD", cache: "no-store" });
    if (!head.ok) throw new Error("no custom model");
    const r = await fetch(CONFIG.classesUrl, { cache: "no-store" });
    if (!r.ok) throw new Error("model/classes.json missing");
    const c = await r.json();
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    ort.env.wasm.numThreads = 1; // iOS Safari: no SharedArrayBuffer on GitHub Pages
    state.session = await ort.InferenceSession.create(CONFIG.modelUrl, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    state.classes = Array.isArray(c) ? c : Object.keys(c).sort((a, b) => +a - +b).map((k) => c[k]);
    state.customClasses = state.classes.slice();
    state.backend = "onnx";
  }

  async function loadCoco() {
    if (!window.cocoSsd || !window.tf) throw new Error("COCO-SSD scripts failed to load");
    // setBackend() resolves to false (it doesn't throw) when WebGL init fails, which on iOS
    // can leave no usable backend. Check the result and fall back explicitly.
    let ok = false;
    try { ok = await tf.setBackend("webgl"); } catch {}
    if (!ok) {
      log("system", "WebGL unavailable, using CPU backend (slow)");
      ok = await tf.setBackend("cpu");
      if (!ok) throw new Error("no TensorFlow.js backend available");
    }
    await tf.ready();
    state.coco = await cocoSsd.load({ base: CONFIG.cocoBase });
    // warm-up pass so shader compilation doesn't stall the first live frame
    try { await state.coco.detect(video, 1, CONFIG.confThreshold); } catch {}
    state.classes = COCO.slice();
    state.backend = "coco";
  }

  function setModeLabel() {
    const canSwitch = !!state.customClasses;
    ui.mode.textContent = state.backend === "onnx" ? "LIVE · TOOLS" : "LIVE · 80 OBJECTS";
    if (canSwitch) ui.mode.textContent += " ⇄";
    ui.mode.className = "pill live" + (canSwitch ? " switch" : "");
  }

  // tap the mode pill: workbench tools (custom YOLO) <-> 80 everyday objects (COCO-SSD)
  async function switchModel() {
    if (!state.running || state.switching || !state.customClasses) return;
    state.switching = true;
    const toCoco = state.backend === "onnx";
    ui.mode.textContent = "SWITCHING…";
    try {
      if (toCoco) {
        if (!state.coco) await loadCoco();
        state.backend = "coco"; state.classes = COCO.slice();
      } else {
        state.backend = "onnx"; state.classes = state.customClasses.slice();
      }
      state.tools = {}; state.detections = []; state.locked = null;
      buildToolUI();
      say(toCoco ? "Everyday objects mode." : "Workbench tools mode.", true);
      log("system", toCoco ? "switched to COCO-SSD (80 everyday objects)" : `switched to workbench tools (${state.classes.length} types)`);
    } catch (e) {
      log("error", "couldn't switch model: " + e.message);
      say("Couldn't load the other model.", true);
    }
    setModeLabel();
    state.switching = false;
  }

  async function loadModel() {
    try { await loadCustomModel(); return "custom YOLO"; }
    catch (e) { if (e.message !== "no custom model") log("system", "custom model skipped: " + e.message); }
    await loadCoco();
    return "COCO-SSD";
  }

  // letterbox into an offscreen canvas and build CHW float tensor
  const prep = document.createElement("canvas");
  prep.width = prep.height = CONFIG.inputSize;
  const pctx = prep.getContext("2d", { willReadFrequently: true });

  function preprocess() {
    const S = CONFIG.inputSize, vw = video.videoWidth, vh = video.videoHeight;
    const scale = Math.min(S / vw, S / vh);
    const nw = Math.round(vw * scale), nh = Math.round(vh * scale);
    const dx = Math.floor((S - nw) / 2), dy = Math.floor((S - nh) / 2);
    pctx.fillStyle = "rgb(114,114,114)";
    pctx.fillRect(0, 0, S, S);
    pctx.drawImage(video, dx, dy, nw, nh);
    const px = pctx.getImageData(0, 0, S, S).data;
    const area = S * S, data = new Float32Array(3 * area);
    for (let i = 0, j = 0; i < area; i++, j += 4) {
      data[i] = px[j] / 255;
      data[i + area] = px[j + 1] / 255;
      data[i + 2 * area] = px[j + 2] / 255;
    }
    return { tensor: new ort.Tensor("float32", data, [1, 3, S, S]), scale, dx, dy };
  }

  // YOLOv8/11 ONNX output: [1, 4 + nc, N] (cx, cy, w, h, class scores...)
  function postprocess(output, { scale, dx, dy }) {
    const [, rows, n] = output.dims;
    const nc = rows - 4, d = output.data, boxes = [];
    for (let i = 0; i < n; i++) {
      let best = 0, cls = -1;
      for (let c = 0; c < nc; c++) { const s = d[(4 + c) * n + i]; if (s > best) { best = s; cls = c; } }
      if (best < CONFIG.confThreshold) continue;
      const cx = d[i], cy = d[n + i], w = d[2 * n + i], h = d[3 * n + i];
      boxes.push({
        cls, conf: best,
        x1: (cx - w / 2 - dx) / scale, y1: (cy - h / 2 - dy) / scale,
        x2: (cx + w / 2 - dx) / scale, y2: (cy + h / 2 - dy) / scale,
      });
    }
    return nms(boxes, CONFIG.iouThreshold).map((b) => ({ ...b, name: state.classes[b.cls] || "class" + b.cls }));
  }

  function iou(a, b) {
    const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
    const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
    const inter = ix * iy, u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return u > 0 ? inter / u : 0;
  }

  function nms(boxes, thr) {
    boxes.sort((a, b) => b.conf - a.conf);
    const keep = [];
    for (const b of boxes) if (!keep.some((k) => k.cls === b.cls && iou(k, b) > thr)) keep.push(b);
    return keep;
  }

  async function cocoDetect() {
    const preds = await state.coco.detect(video, 20, CONFIG.confThreshold);
    return preds.map((p) => ({ cls: COCO.indexOf(p.class), name: p.class, conf: p.score,
      x1: p.bbox[0], y1: p.bbox[1], x2: p.bbox[0] + p.bbox[2], y2: p.bbox[1] + p.bbox[3] }));
  }

  // ---------- simulation (FORGE_SIM=1 equivalent) ----------
  const SIM_OBJECTS = ["cup", "cell phone", "scissors", "laptop"];
  function simDetect(t) {
    const vw = video.videoWidth || 720, vh = video.videoHeight || 960, out = [];
    SIM_OBJECTS.forEach((name, i) => {
      // each tool periodically "leaves" the bench so inventory changes are visible
      if (Math.sin(t / 6000 + i * 2.1) < -0.75) return;
      const cx = vw * (0.2 + 0.2 * i) + Math.sin(t / 1500 + i) * vw * 0.05;
      const cy = vh * (0.35 + 0.15 * (i % 2)) + Math.cos(t / 1700 + i) * vh * 0.04;
      const w = vw * 0.18, h = vh * 0.22;
      out.push({ cls: i, name, conf: 0.82 + 0.1 * Math.sin(t / 900 + i), x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 });
    });
    return out;
  }

  // ---------- tracking / inventory ----------
  function updateTracker(dets) {
    const now = performance.now();
    const bestByTool = {}, counts = {};
    for (const d of dets) {
      counts[d.name] = (counts[d.name] || 0) + 1;
      if (!bestByTool[d.name] || d.conf > bestByTool[d.name].conf) bestByTool[d.name] = d;
    }
    for (const name of Object.keys(bestByTool)) tool(name);

    for (const name of Object.keys(state.tools)) {
      const t = state.tools[name], d = bestByTool[name];
      if (d) { t.lastSeen = now; t.conf = d.conf; t.box = d; t.count = counts[name]; }
      const present = t.lastSeen > 0 && now - t.lastSeen < CONFIG.presentTimeoutMs;
      if (present !== t.present) {
        t.present = present;
        DB.insert("tool_events", { tool_name: name, state: present ? "present" : "missing", confidence: +t.conf.toFixed(3), source: state.sim ? "sim" : "camera" });
        log("tool", `${pretty(name)} → ${present ? "present" : "missing"}`);
      }
    }

    // lock-on guidance (replaces the laser pan-tilt)
    if (state.locked) {
      const t = state.tools[state.locked];
      if (t && bestByTool[state.locked]) {
        if (now - (state.lastFoundAnnounce[state.locked] || 0) > 8000) {
          state.lastFoundAnnounce[state.locked] = now;
          state.finds++;
          say(`${cap(pretty(state.locked))} found, ${where(t.box)}.`, true);
        }
      }
    }
  }

  const cap = (s) => s[0].toUpperCase() + s.slice(1);

  function where(b) {
    // positions relative to what's visible on screen: object-fit: cover crops the frame
    const { s, ox, oy } = coverMap();
    const W = overlay.width || 1, H = overlay.height || 1;
    const cx = (((b.x1 + b.x2) / 2) * s + ox) / W, cy = (((b.y1 + b.y2) / 2) * s + oy) / H;
    const h = cx < 0.35 ? "left" : cx > 0.65 ? "right" : "center";
    const v = cy < 0.35 ? "top" : cy > 0.65 ? "bottom" : "";
    return v ? `${v} ${h}` : h === "center" ? "dead center" : `${h} side`;
  }

  // ---------- drawing ----------
  // object-fit: cover mapping from video pixels to canvas pixels
  function coverMap() {
    const W = overlay.width || 1, H = overlay.height || 1;
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return { s: 1, ox: 0, oy: 0 };
    const s = Math.max(W / vw, H / vh);
    return { s, ox: (W - vw * s) / 2, oy: (H - vh * s) / 2 };
  }

  function fitOverlay() {
    // the canvas backing store must match its CSS box exactly (same aspect), otherwise the
    // canvas's own object-fit would rescale it and boxes would drift
    const r = overlay.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (overlay.width !== w || overlay.height !== h) { overlay.width = w; overlay.height = h; }
    return { ...coverMap(), dpr };
  }

  function draw(dets, t) {
    const { s, ox, oy, dpr } = fitOverlay();
    const W = overlay.width, H = overlay.height;
    octx.clearRect(0, 0, W, H);
    octx.lineWidth = 2 * dpr;
    octx.font = `${600} ${13 * dpr}px -apple-system, system-ui, sans-serif`;

    for (const d of dets) {
      const x = d.x1 * s + ox, y = d.y1 * s + oy, w = (d.x2 - d.x1) * s, h = (d.y2 - d.y1) * s;
      const isTarget = d.name === state.locked;
      if (state.locked && !isTarget) octx.globalAlpha = 0.35;
      octx.strokeStyle = isTarget ? "#ff3b30" : "#2ecc71";
      octx.strokeRect(x, y, w, h);
      const label = `${pretty(d.name)} ${(d.conf * 100).toFixed(0)}%`;
      const tw = octx.measureText(label).width + 10 * dpr;
      octx.fillStyle = isTarget ? "#ff3b30" : "#2ecc71";
      const ly = y - 20 * dpr < 0 ? y : y - 20 * dpr; // flip inside the box at the top edge
      const lx = Math.min(Math.max(0, x), W - tw);
      octx.fillRect(lx, ly, tw, 20 * dpr);
      octx.fillStyle = "#000";
      octx.fillText(label, lx + 5 * dpr, ly + 14 * dpr);
      octx.globalAlpha = 1;

      if (isTarget) drawLaser(x + w / 2, y + h / 2, dpr, t);
    }

    // target not in frame -> arrow toward last known position
    if (state.locked && !dets.some((d) => d.name === state.locked)) {
      const tool = state.tools[state.locked];
      octx.fillStyle = "rgba(0,0,0,.55)";
      octx.fillRect(0, H / 2 - 22 * dpr, W, 44 * dpr);
      octx.fillStyle = "#fff";
      octx.textAlign = "center";
      octx.fillText(`Searching for ${pretty(state.locked)}…`, W / 2, H / 2 + 5 * dpr);
      octx.textAlign = "left";
      if (tool && tool.box) {
        const bx = ((tool.box.x1 + tool.box.x2) / 2) * s + ox, by = ((tool.box.y1 + tool.box.y2) / 2) * s + oy;
        drawArrow(W / 2, H / 2 + 50 * dpr, bx, by, dpr);
      }
    }
  }

  function drawLaser(x, y, dpr, t) {
    const pulse = 6 + 3 * Math.sin(t / 120);
    const g = octx.createRadialGradient(x, y, 0, x, y, 28 * dpr);
    g.addColorStop(0, "rgba(255,40,40,.95)");
    g.addColorStop(0.25, "rgba(255,40,40,.5)");
    g.addColorStop(1, "rgba(255,40,40,0)");
    octx.fillStyle = g;
    octx.beginPath(); octx.arc(x, y, 28 * dpr, 0, Math.PI * 2); octx.fill();
    octx.fillStyle = "#fff";
    octx.beginPath(); octx.arc(x, y, pulse * dpr * 0.4, 0, Math.PI * 2); octx.fill();
    octx.strokeStyle = "rgba(255,59,48,.9)";
    octx.lineWidth = 1.5 * dpr;
    const r = 40 * dpr;
    for (const [a, b] of [[0, 1], [Math.PI, 1]]) { octx.beginPath(); octx.arc(x, y, r, a - 0.6, a + 0.6); octx.stroke(); }
  }

  function drawArrow(x0, y0, x1, y1, dpr) {
    const ang = Math.atan2(y1 - y0, x1 - x0), len = 46 * dpr;
    const ex = x0 + Math.cos(ang) * len, ey = y0 + Math.sin(ang) * len;
    octx.strokeStyle = "#ff6a1a"; octx.fillStyle = "#ff6a1a"; octx.lineWidth = 4 * dpr;
    octx.beginPath(); octx.moveTo(x0, y0); octx.lineTo(ex, ey); octx.stroke();
    octx.beginPath();
    octx.moveTo(ex + Math.cos(ang) * 12 * dpr, ey + Math.sin(ang) * 12 * dpr);
    octx.lineTo(ex + Math.cos(ang + 2.5) * 14 * dpr, ey + Math.sin(ang + 2.5) * 14 * dpr);
    octx.lineTo(ex + Math.cos(ang - 2.5) * 14 * dpr, ey + Math.sin(ang - 2.5) * 14 * dpr);
    octx.fill();
  }

  // ---------- main loop ----------
  let frames = 0, fpsT = performance.now(), busy = false, uiT = 0;

  async function loop(t) {
    if (!state.running) return;
    if (!busy && !state.switching && video.readyState >= 2 && video.videoWidth > 0) {
      busy = true;
      const t0 = performance.now();
      const be = state.backend;
      try {
        let dets;
        if (be === "sim") dets = simDetect(t);
        else if (be === "coco") dets = await cocoDetect();
        else {
          const pre = preprocess();
          const out = await state.session.run({ [state.session.inputNames[0]]: pre.tensor });
          dets = postprocess(out[state.session.outputNames[0]], pre);
        }
        // the model was switched while this frame ran: drop its results (wrong class list)
        if (state.backend === be && !state.switching) {
          state.detections = dets;
          ui.inf.textContent = Math.round(performance.now() - t0);
          updateTracker(dets);
          frames++;
        }
      } catch (e) {
        console.error(e);
        log("error", e.message);
      }
      busy = false;
    }
    try {
      draw(state.detections, t);
      if (t - uiT > 400) { refreshUI(); uiT = t; }
    } catch (e) { console.error(e); }
    if (t - fpsT > 1000) { ui.fps.textContent = `${frames} fps`; frames = 0; fpsT = t; }
    requestAnimationFrame(loop);
  }

  let starting = false;
  async function start() {
    if (starting || state.running) return; // ignore double taps
    starting = true;
    const btn = $("startBtn"), msg = $("startMsg");
    btn.disabled = true;
    msg.textContent = "Starting camera…";
    say("FORGE online.", true);   // unlocks iOS speech; must stay synchronous in the tap (before any await)

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
        throw new Error("camera API unavailable (the page must be opened over https)");
      // iOS needs these set as properties too, before srcObject, or play() can go fullscreen / reject
      video.muted = true;
      video.playsInline = true;
      video.setAttribute("playsinline", "");
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false,
        });
      } catch (e) {
        if (e.name !== "OverconstrainedError") throw e;
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }
      video.srcObject = stream;
      // wait for dimensions; videoWidth is 0 until metadata arrives
      if (video.readyState < 1) await new Promise((res) => video.addEventListener("loadedmetadata", res, { once: true }));
      await video.play();
    } catch (e) {
      const why = e.name === "NotAllowedError" ? "permission denied" : (e.message || e.name);
      msg.textContent = "Camera blocked: " + why + ". Open this page over https in Safari and allow camera access (Settings › Safari › Camera).";
      btn.disabled = false;
      starting = false;
      return;
    }

    msg.textContent = "Loading model… (first time takes a few seconds)";
    try {
      const which = await loadModel();
      state.sim = false;
      setModeLabel();
      log("system", `${which} loaded (${state.classes.length} object types)`);
      buildToolUI();
    } catch (e) {
      state.sim = true;
      state.backend = "sim";
      state.classes = COCO.slice();
      buildToolUI();
      ui.mode.textContent = "SIMULATION";
      ui.mode.className = "pill sim";
      log("system", `simulation mode — ${e.message}`);
      say("Couldn't load a detection model. Running in simulation mode.", true);
    }

    $("startScreen").style.display = "none";
    state.running = true;
    starting = false;
    requestAnimationFrame(loop);
  }

  // iOS pauses the camera when the tab/app is backgrounded; resume when we come back
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && state.running && video.paused) video.play().catch(() => {});
  });

  // ---------- wire up ----------
  buildToolUI();
  restoreLog();
  setupVoice();
  $("startBtn").onclick = start;
  ui.mode.onclick = switchModel;
  $("findForm").onsubmit = (e) => {
    e.preventDefault();
    const v = $("findInput").value.trim();
    if (!v) return;
    command(/^(find|locate|where|show|track)\b/i.test(v) ? v : "find " + v, "typed");
    $("findInput").value = "";
    $("findInput").blur();
  };
  $("autoBtn").onclick = () => command("auto", "button");
  $("seeBtn").onclick = () => command("what do you see", "button");
  $("clearLog").onclick = () => { DB.clear(); ui.log.innerHTML = ""; log("system", "history cleared"); };

  // expose for tests
  window.__forge = { postprocess, nms, iou, matchTool, askedFor, command, updateTracker, state, CONFIG };
})();
