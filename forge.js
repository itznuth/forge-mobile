/* F.O.R.G.E. Mobile: on-device tool detection, guidance and bench tracking for phone browsers.
 *
 * Pipeline: camera -> detector -> smoothing -> tracker -> guidance (reticle, hints, sonar) / voice / bench watch
 *
 * Detectors:
 *   1. model/best.onnx: the FORGE workbench model (YOLO11n, 7 tools), via onnxruntime-web (WASM)
 *   2. COCO-SSD:        pretrained, 80 everyday object types, via TensorFlow.js (WebGL)
 *   3. simulation:      fake detections when neither model can load (FORGE_SIM=1 equivalent)
 *   Tap the mode pill to switch between 1 and 2.
 *
 * Mirrors the original Pi runtime:
 *   laser dot + pan-tilt   -> laser reticle, "pan left/up" hints, sonar beeps
 *   Vosk voice             -> Web Speech API
 *   Piper / eSpeak TTS     -> speechSynthesis
 *   SQLite event tables    -> localStorage (tool_events, command_events, bench checkouts)
 */
(() => {
  "use strict";

  // ---------- config ----------
  const CONFIG = {
    modelUrl: "model/best.onnx",
    classesUrl: "model/classes.json",
    ortBase: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/",
    cocoBase: "lite_mobilenet_v2",  // fastest COCO-SSD variant
    inputSize: 640,
    confThreshold: 0.5,             // overwritten from settings
    iouThreshold: 0.45,
    presentTimeoutMs: 2500,         // object counts as "in view" if seen within this window
    benchOutMs: 5000,               // bench watch: unseen this long = checked out (filters flicker)
    speakCooldownMs: 3500,
    maxLogRows: 200,
    recentMs: 60000,                // how long a vanished object stays in the "In view" list / chips
    centerTolerance: 0.15,          // |offset| below this (of half the screen) counts as on target
  };

  const COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear",
    "zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
    "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"];

  // spoken/typed words -> detector class names (only used when the active model has that class)
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

  // ---------- settings (persisted) ----------
  const readJSON = (k, d) => { try { const v = JSON.parse(localStorage.getItem(k)); return v == null ? d : v; } catch { return d; } };
  const writeJSON = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} };
  const settings = Object.assign({ conf: 50, voice: true, sonar: true, smooth: true, overdueMin: 5, facing: "environment" },
    readJSON("forge_settings", {}));
  const saveSettings = () => { CONFIG.confThreshold = settings.conf / 100; writeJSON("forge_settings", settings); };
  CONFIG.confThreshold = settings.conf / 100;

  // ---------- dom ----------
  const $ = (id) => document.getElementById(id);
  const video = $("video"), overlay = $("overlay"), octx = overlay.getContext("2d");
  const ui = {
    mode: $("modePill"), fps: $("fpsPill"), lock: $("lockPill"), say: $("say"), guide: $("guide"),
    inv: $("inventory"), log: $("log"), tools: $("toolButtons"), mic: $("micBtn"),
    finds: $("statFinds"), cmds: $("statCmds"), inf: $("statInf"), fpsStat: $("statFps"), backend: $("statBackend"),
    bench: $("bench"), benchState: $("benchState"), benchHelp: $("benchHelp"), benchBtn: $("benchBtn"),
    benchStop: $("benchStop"), watchBadge: $("watchBadge"),
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
    startedWall: 0,
    stream: null, track: null, torchOK: false,
    locked: null,           // object name or null (auto mode)
    detections: [],
    tools: {},              // name -> {lastSeen, lastSeenWall, conf, box, present, count}
    chipKey: "",
    guideText: "",
    lastSpoken: 0,
    lastFoundAnnounce: {},
    finds: 0, cmds: 0,
    snapUrl: "",
  };

  // ---------- persistence (SQLite-style tables in localStorage) ----------
  const DB = {
    read(table) { const v = readJSON("forge_" + table, []); return Array.isArray(v) ? v : []; },
    write(table, rows) { writeJSON("forge_" + table, rows.slice(-CONFIG.maxLogRows)); },
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

  // ---------- small helpers ----------
  const pretty = (n) => n.replace(/_/g, " ");
  const cap = (s) => s[0].toUpperCase() + s.slice(1);
  const plural = (n) => (/s$/.test(pretty(n)) ? pretty(n) : pretty(n) + "s");
  const listJoin = (a) => a.length < 2 ? a.join("") : a.slice(0, -1).join(", ") + " and " + a[a.length - 1];
  const countName = (n) => { const c = state.tools[n].count; return c > 1 ? `${c} ${plural(n)}` : `a ${pretty(n)}`; };
  const fmtShort = (ms) => { const s = Math.max(0, Math.round(ms / 1000)); return s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${Math.floor(s / 3600)}h ${Math.floor(s / 60) % 60}m`; };
  const fmtSpoken = (ms) => { const s = Math.max(1, Math.round(ms / 1000)); if (s < 60) return `${s} second${s === 1 ? "" : "s"}`; const m = Math.round(s / 60); if (m < 60) return `${m} minute${m === 1 ? "" : "s"}`; const h = Math.floor(m / 60); return `${h} hour${h === 1 ? "" : "s"}${m % 60 ? ` ${m % 60} minutes` : ""}`; };
  const clock = (t) => new Date(t).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

  // ---------- speech out ----------
  let speechUnlocked = false;
  let currentUtterance = null; // keep a reference: iOS can GC an utterance mid-speech and drop it
  let speechMuted = false;      // true while the mic is listening (iOS can't record and speak at once)

  function say(text, force = false) {
    ui.say.textContent = text;
    if (!settings.voice || !("speechSynthesis" in window) || speechMuted) return;
    const now = performance.now();
    if (!force && now - state.lastSpoken < CONFIG.speakCooldownMs) return;
    state.lastSpoken = now;
    try {
      // iOS Safari: cancel() immediately followed by speak() often swallows the new utterance,
      // so only cancel when one of our utterances is actually queued.
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

  // Must run synchronously inside a user tap: iOS only allows speech after a speak() call made
  // directly from a gesture handler; anything after an await no longer counts.
  function unlockSpeech() {
    if (speechUnlocked || !("speechSynthesis" in window)) return;
    speechUnlocked = true;
    try { const u = new SpeechSynthesisUtterance(" "); u.volume = 0; speechSynthesis.speak(u); } catch {}
  }

  // ---------- sonar (Web Audio) ----------
  let actx = null, nextBeep = 0;
  // also needs a tap: iOS starts AudioContexts suspended
  function unlockAudio() {
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      if (!actx) actx = new AC();
      if (actx.state !== "running") actx.resume().catch(() => {});
      const src = actx.createBufferSource();
      src.buffer = actx.createBuffer(1, 1, 22050);
      src.connect(actx.destination);
      src.start(0);
    } catch {}
  }

  function beep(freq, dur = 0.05, vol = 0.18) {
    if (!actx) return;
    if (actx.state !== "running") { actx.resume().catch(() => {}); return; }
    const t = actx.currentTime, o = actx.createOscillator(), g = actx.createGain();
    o.type = "sine"; o.frequency.value = freq;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(vol, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g); g.connect(actx.destination);
    o.start(t); o.stop(t + dur + 0.02);
  }

  // like a parking sensor: beeps speed up and rise in pitch as the target nears the centre
  function sonar(off, now) {
    if (!settings.sonar || !off || speechMuted || now < nextBeep) return;
    if ("speechSynthesis" in window && speechSynthesis.speaking) return; // don't talk over FORGE
    const d = Math.min(1, Math.hypot(off.dx, off.dy));
    const centered = d < CONFIG.centerTolerance;
    nextBeep = now + (centered ? 120 : 160 + d * 800);
    beep(centered ? 1320 : 440 + (1 - d) * 520);
  }

  const buzz = (ms) => { try { navigator.vibrate && navigator.vibrate(ms); } catch {} }; // Android only

  // ---------- UI ----------
  function tool(name) {
    return (state.tools[name] = state.tools[name] || { lastSeen: 0, lastSeenWall: 0, conf: 0, box: null, present: false, count: 0 });
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
      if (!recent.length) {
        const hint = state.backend === "onnx" ? "Show FORGE a tool. Tap a name or a box to find it." : "Objects FORGE sees will appear here. Tap one to find it.";
        ui.tools.innerHTML = `<span class="hint">${esc(hint)}</span>`;
      }
      for (const name of recent) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = pretty(name);
        b.classList.toggle("active", name === state.locked);
        b.onclick = () => { unlockAudio(); command(`find ${pretty(name)}`, "button"); };
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
    renderBench();
  }

  // ---------- bench watch (the original workbench's tool check-out log) ----------
  // bench = { active, backend, savedAt, items: {name: {status: "in"|"out", since, warned}}, history: [{tool, out, back}] }
  let bench = readJSON("forge_bench", null);
  if (!bench || typeof bench !== "object" || !bench.items) bench = { active: false, backend: "", savedAt: 0, items: {}, history: [] };
  if (!Array.isArray(bench.history)) bench.history = [];
  const saveBenchState = () => writeJSON("forge_bench", { ...bench, history: bench.history.slice(-50) });

  let wakeLock = null;
  async function keepAwake(on) {
    try {
      if (on && !wakeLock && "wakeLock" in navigator) {
        wakeLock = await navigator.wakeLock.request("screen");
        wakeLock.addEventListener("release", () => { wakeLock = null; });
      } else if (!on && wakeLock) { await wakeLock.release(); wakeLock = null; }
    } catch {}
  }

  function saveBench() {
    if (!state.running) return "Start the camera first, then save your bench.";
    const present = Object.keys(state.tools).filter((n) => state.tools[n].present);
    if (!present.length) return "I don't see anything to save yet. Point the camera at your bench and try again.";
    const now = Date.now();
    bench = { active: true, backend: state.backend, savedAt: now, items: {}, history: bench.history || [] };
    for (const n of present) bench.items[n] = { status: "in", since: now, warned: false };
    saveBenchState();
    keepAwake(true);
    DB.insert("tool_events", { tool_name: present.join(", "), state: "bench saved", confidence: 1, source: state.backend });
    log("bench", `saved: ${present.map(pretty).join(", ")}`);
    return `Bench saved with ${listJoin(present.map((n) => countName(n)))}. Keep the phone still and I'll log anything that leaves.`;
  }

  function stopBench() {
    if (!bench.active) return "Bench watch is already off.";
    bench.active = false;
    saveBenchState();
    keepAwake(false);
    log("bench", "watch stopped");
    return "Stopped watching the bench.";
  }

  function updateBench() {
    if (!bench.active || !state.running || bench.backend !== state.backend || state.switching) return;
    const now = Date.now();
    if (now - state.startedWall < CONFIG.benchOutMs) return; // grace period after (re)start
    let changed = false;
    for (const [name, it] of Object.entries(bench.items)) {
      const t = state.tools[name];
      const seenAt = (t && t.lastSeenWall) || 0;
      const seen = now - seenAt < CONFIG.benchOutMs;
      if (it.status === "in" && !seen) {
        it.status = "out"; it.since = Math.max(seenAt, state.startedWall, bench.savedAt); it.warned = false; changed = true;
        bench.history.push({ tool: name, out: it.since, back: 0 });
        DB.insert("tool_events", { tool_name: name, state: "checked out", confidence: 0, source: "bench" });
        log("bench", `${pretty(name)} checked out`);
        say(`${cap(pretty(name))} checked out.`, true);
        beep(330, 0.12);
      } else if (it.status === "out" && seen) {
        const away = now - it.since;
        it.status = "in"; it.since = now; changed = true;
        const h = [...bench.history].reverse().find((x) => x.tool === name && !x.back);
        if (h) h.back = now;
        DB.insert("tool_events", { tool_name: name, state: "returned", confidence: t.conf, source: "bench" });
        log("bench", `${pretty(name)} returned after ${fmtShort(away)}`);
        say(`${cap(pretty(name))} returned after ${fmtSpoken(away)}.`, true);
        beep(880, 0.08);
        buzz(40);
      } else if (it.status === "out" && !it.warned && now - it.since > settings.overdueMin * 60000) {
        it.warned = true; changed = true;
        log("bench", `${pretty(name)} overdue`);
        say(`Reminder: the ${pretty(name)} has been out for ${fmtSpoken(now - it.since)}.`, true);
        beep(660, 0.1); setTimeout(() => beep(660, 0.1), 180);
      }
    }
    if (changed) saveBenchState();
  }

  function benchOut() { return Object.keys(bench.items).filter((n) => bench.items[n].status === "out"); }

  function renderBench() {
    const names = Object.keys(bench.items);
    const mismatch = bench.active && state.running && bench.backend && bench.backend !== state.backend;
    ui.benchState.textContent = bench.active ? (mismatch ? "paused (other mode)" : `watching · ${names.length} item${names.length === 1 ? "" : "s"}`) : "off";
    ui.watchBadge.style.display = bench.active && state.running && !mismatch ? "block" : "none";
    ui.benchBtn.textContent = bench.active ? "Re-save bench" : "Save bench";
    ui.benchStop.disabled = !bench.active;
    ui.benchHelp.style.display = bench.active ? "none" : "";
    const now = Date.now();
    let html = "";
    if (bench.active) {
      for (const n of names.sort((a, b) => (bench.items[b].status === "out") - (bench.items[a].status === "out"))) {
        const it = bench.items[n], out = it.status === "out";
        const overdue = out && now - it.since > settings.overdueMin * 60000;
        html += `<div class="tool"><span><span class="dot${out ? (overdue ? "" : " warn") : " on"}"></span>${esc(pretty(n))}</span>` +
          `<span class="meta">${out ? `<span class="${overdue ? "bad" : ""}">out ${fmtShort(now - it.since)}</span>` : "on bench"}</span></div>`;
      }
    }
    const hist = (bench.history || []).slice(-5).reverse();
    if (hist.length) {
      html += `<div class="meta" style="margin-top:6px">Recent check-outs</div>`;
      for (const h of hist) {
        html += `<div class="tool meta"><span>${esc(pretty(h.tool))}</span><span>${clock(h.out)} → ${h.back ? `${clock(h.back)} (${fmtShort(h.back - h.out)})` : "still out"}</span></div>`;
      }
    }
    if (ui.bench.innerHTML !== html) ui.bench.innerHTML = html;
  }

  // ---------- commands (shared by buttons, taps, typing and voice) ----------
  const escapeRe = (x) => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const hasWord = (text, w) => new RegExp(`\\b${escapeRe(w)}(e?s)?\\b`).test(text);

  function matchTool(text) {
    const t = " " + text.toLowerCase().replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ") + " ";
    const vocab = state.classes.map((n) => [pretty(n), n]);
    const aliases = Object.entries(ALIASES).filter(([, n]) => state.classes.includes(n));
    // longest phrases first so "cell phone" beats "phone", "allen wrench" beats "wrench"
    const all = [...vocab, ...aliases].sort((a, b) => b[0].length - a[0].length);
    for (const [word, name] of all) if (hasWord(t, word)) return name;
    return null;
  }

  // the object the user asked for, even if the detector can't recognize it
  function askedFor(text) {
    const m = text.toLowerCase().match(/^(?:find|locate|where(?:'s| is| are)?|lock(?: on)?|show(?: me)?|track|get)\s+(?:my |the |a |an |me )*(.+?)\??$/);
    return m ? m[1].trim() : null;
  }

  const HELP = 'Try: "find wrench", "how many screwdrivers", "what do you see", "save bench", "what\'s missing", "snapshot", "everyday mode", "flashlight on", or "unlock".';

  function command(raw, source) {
    const text = raw.toLowerCase().replace(/^\s*(hey\s+|ok\s+)?forge[,\s]*/, "").replace(/[.!]+$/, "").trim();
    let ok = true, msg = "";
    const tool = matchTool(text);

    if (/^(help|commands|what can (you|i) (do|say))\b/.test(text)) {
      msg = HELP;
    } else if (/\b(save|start|watch|remember)\b.*\bbench\b|^start watching|^watch( the)? (bench|tools)/.test(text)) {
      msg = saveBench(); ok = bench.active;
    } else if (/stop watching|\b(stop|end|clear)\b.*\b(bench|watch)/.test(text)) {
      msg = stopBench();
    } else if (/missing|checked out|what'?s (out|gone)|what is (out|gone)/.test(text)) {
      if (bench.active) {
        const out = benchOut(), now = Date.now();
        msg = out.length ? `Out right now: ${listJoin(out.map((n) => `the ${pretty(n)} for ${fmtSpoken(now - bench.items[n].since)}`))}.` : "Everything is on the bench.";
      } else {
        const now = performance.now();
        const gone = Object.keys(state.tools).filter((n) => !state.tools[n].present && state.tools[n].lastSeen > 0 && now - state.tools[n].lastSeen < CONFIG.recentMs);
        msg = (gone.length ? `Recently gone: ${listJoin(gone.map(pretty))}.` : "Nothing has gone missing recently.") + " Tip: save your bench to track check-outs.";
      }
    } else if (/^how many\b/.test(text)) {
      if (!tool) { ok = false; msg = "How many of what? Try \"how many wrenches\"."; }
      else {
        const t = state.tools[tool], n = t && t.present ? t.count : 0;
        msg = n ? `I see ${n === 1 ? "one" : n} ${n === 1 ? pretty(tool) : plural(tool)}.` : `I don't see any ${plural(tool)} right now.`;
      }
    } else if (/\b(snap(shot)?|photo|picture|screenshot)\b/.test(text)) {
      ok = takeSnapshot(); msg = ok ? "Snapshot taken." : "Start the camera first.";
    } else if (/\b(tools?|workbench) mode\b|\beveryday\b|\b80 objects\b|\bswitch (mode|model)s?\b/.test(text)) {
      const target = /everyday|80 objects/.test(text) ? "coco" : /tools?|workbench/.test(text) ? "onnx" : null;
      switchModel(target);
      return finish(raw, source, tool, true, ""); // switchModel speaks for itself
    } else if (/\b(flash ?light|torch|light)\b/.test(text) && /\b(on|off)\b/.test(text)) {
      const on = /\bon\b/.test(text);
      ok = setTorch(on); msg = ok ? `Flashlight ${on ? "on" : "off"}.` : "This camera doesn't have a flashlight I can control.";
    } else if (/^(find|locate|where|lock|show|track|get)\b/.test(text) || (tool && text.split(" ").length <= 3)) {
      if (!tool) {
        ok = false;
        const what = askedFor(text);
        msg = what
          ? `I can't recognize "${what}" yet. ${state.backend === "onnx" ? "Tools mode only knows workbench tools. Say \"everyday mode\" or tap the mode button." : "It's not one of the 80 object types I know."}${state.backend === "coco" && state.customClasses ? " Say \"tools mode\" for workbench tools." : ""}`
          : "I didn't catch what to find.";
      } else {
        state.locked = tool;
        state.lastFoundAnnounce[tool] = 0;
        msg = state.tools[tool] && state.tools[tool].present ? `Locking on ${pretty(tool)}.` : `Searching for ${pretty(tool)}. Pan the camera around.`;
      }
    } else if (/unlock|release|\bstop\b|cancel/.test(text)) {
      state.locked = null; msg = "Released. Back to auto mode.";
    } else if (/\bauto\b/.test(text)) {
      state.locked = null; msg = "Auto mode.";
    } else if (/what do you see|what.*see|what'?s here|scan/.test(text)) {
      const seen = Object.keys(state.tools).filter((n) => state.tools[n].present);
      msg = seen.length ? `I see ${listJoin(seen.map((n) => countName(n)))}.` : "I don't see anything I recognize right now.";
    } else if (/status|inventory/.test(text)) {
      const seen = Object.keys(state.tools).filter((n) => state.tools[n].present);
      msg = `${seen.length} object type${seen.length === 1 ? "" : "s"} in view.` + (bench.active ? ` Bench watch is on, ${benchOut().length} out.` : "");
    } else {
      ok = false; msg = `I didn't understand "${raw}". ${HELP}`;
    }
    finish(raw, source, tool, ok, msg);
  }

  function finish(raw, source, tool, ok, msg) {
    state.cmds++;
    DB.insert("command_events", { command_type: source, command_text: raw, target_tool: tool, success: ok, message: msg });
    log("cmd", raw, ` <span>(${esc(source)})</span>`);
    if (msg) say(msg, true);
    refreshUI();
  }

  // ---------- voice in ----------
  function setupVoice() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      ui.mic.onclick = () => say("Voice input isn't supported in this browser. Use the Find box or the buttons.", true);
      return;
    }
    let rec = null, listening = false, heard = null, watchdog = 0;
    const resetMic = () => {
      clearTimeout(watchdog);
      listening = false; rec = null; speechMuted = false;
      ui.mic.classList.remove("listening"); ui.mic.textContent = "🎙 Voice";
    };
    ui.mic.onclick = () => {
      unlockSpeech(); unlockAudio();
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
        heard = alts.find((a) => matchTool(a)) ||
          alts.find((a) => /unlock|auto|see|status|stop|bench|missing|snap|photo|picture|how many|mode|light|help/.test(a.toLowerCase())) || alts[0];
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

  // ---------- models ----------
  const progressEl = $("progress"), progressBar = $("progressBar");
  function showProgress(f) {
    if (f == null) { progressEl.style.display = "none"; return; }
    progressEl.style.display = "block";
    progressBar.style.width = Math.round(Math.min(1, f) * 100) + "%";
  }

  async function fetchBytes(url, onProgress) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(r.status === 404 ? "no custom model" : `model download failed (HTTP ${r.status})`);
    const total = +r.headers.get("content-length") || 0;
    if (!r.body || !r.body.getReader) return new Uint8Array(await r.arrayBuffer());
    const reader = r.body.getReader(), chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); got += value.length;
      if (total) onProgress(got / total);
    }
    const out = new Uint8Array(got);
    let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.length; }
    return out;
  }

  async function loadCustomModel(msg) {
    if (!window.ort) throw new Error("onnxruntime-web not loaded");
    const r = await fetch(CONFIG.classesUrl, { cache: "no-cache" });
    if (!r.ok) throw new Error("no custom model");
    const c = await r.json();
    msg.textContent = "Downloading tools model…";
    const bytes = await fetchBytes(CONFIG.modelUrl, (f) => {
      showProgress(f);
      msg.textContent = `Downloading tools model… ${Math.round(f * 100)}%`;
    });
    msg.textContent = "Starting tools model…";
    ort.env.wasm.wasmPaths = CONFIG.ortBase;
    ort.env.wasm.numThreads = 1; // no SharedArrayBuffer on GitHub Pages (no COOP/COEP headers)
    state.session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
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
    ui.mode.textContent = (state.backend === "onnx" ? "LIVE · TOOLS" : "LIVE · 80 OBJECTS") + (canSwitch ? " ⇄" : "");
    ui.mode.className = "pill live" + (canSwitch ? " switch" : "");
    ui.backend.textContent = state.backend === "onnx"
      ? `Tools mode · YOLO11n · ONNX Runtime Web (WASM) · ${state.classes.length} classes`
      : `Everyday mode · COCO-SSD lite · TensorFlow.js (${window.tf ? tf.getBackend() : "?"}) · 80 classes`;
  }

  // workbench tools (custom YOLO) <-> 80 everyday objects (COCO-SSD); target: "onnx" | "coco" | null = toggle
  async function switchModel(target) {
    if (!state.running || state.switching) return;
    if (!state.customClasses) { say("Only the everyday objects model is available.", true); return; }
    const toCoco = target ? target === "coco" : state.backend === "onnx";
    if ((toCoco ? "coco" : "onnx") === state.backend) { say(toCoco ? "Already in everyday mode." : "Already in tools mode.", true); return; }
    state.switching = true;
    ui.mode.textContent = "SWITCHING…";
    try {
      if (toCoco) {
        if (!state.coco) await loadCoco();
        state.backend = "coco"; state.classes = COCO.slice();
      } else {
        state.backend = "onnx"; state.classes = state.customClasses.slice();
      }
      state.tools = {}; state.detections = []; state.locked = null;
      state.startedWall = Date.now(); // bench grace period
      say(toCoco ? "Everyday objects mode." : "Workbench tools mode.", true);
      log("system", toCoco ? "switched to COCO-SSD (80 everyday objects)" : `switched to workbench tools (${state.classes.length} types)`);
    } catch (e) {
      log("error", "couldn't switch model: " + e.message);
      say("Couldn't load the other model.", true);
    }
    setModeLabel();
    state.switching = false;
    buildToolUI();
  }

  async function loadModel(msg) {
    try { await loadCustomModel(msg); return "Tools model (YOLO11n)"; }
    catch (e) { if (e.message !== "no custom model") log("system", "tools model skipped: " + e.message); }
    msg.textContent = "Loading everyday objects model…";
    showProgress(null);
    await loadCoco();
    return "COCO-SSD";
  }

  // letterbox into an offscreen canvas and build a CHW float tensor
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

  // blend each box toward its match in the previous frame (same class, overlapping) to cut jitter
  function smooth(prev, dets, a = 0.55) {
    if (!settings.smooth || !prev.length) return dets;
    const used = new Set();
    return dets.map((d) => {
      let bi = -1, bv = 0.3;
      prev.forEach((p, i) => { if (p.name === d.name && !used.has(i)) { const v = iou(p, d); if (v > bv) { bv = v; bi = i; } } });
      if (bi < 0) return d;
      used.add(bi);
      const p = prev[bi];
      return { ...d, x1: p.x1 + (d.x1 - p.x1) * a, y1: p.y1 + (d.y1 - p.y1) * a, x2: p.x2 + (d.x2 - p.x2) * a, y2: p.y2 + (d.y2 - p.y2) * a };
    });
  }

  // ---------- simulation (FORGE_SIM=1 equivalent) ----------
  const SIM_OBJECTS = ["cup", "cell phone", "scissors", "laptop"];
  function simDetect(t) {
    const vw = video.videoWidth || 720, vh = video.videoHeight || 960, out = [];
    SIM_OBJECTS.forEach((name, i) => {
      if (Math.sin(t / 6000 + i * 2.1) < -0.75) return; // objects periodically "leave" so changes are visible
      const cx = vw * (0.2 + 0.2 * i) + Math.sin(t / 1500 + i) * vw * 0.05;
      const cy = vh * (0.35 + 0.15 * (i % 2)) + Math.cos(t / 1700 + i) * vh * 0.04;
      const w = vw * 0.18, h = vh * 0.22;
      out.push({ cls: i, name, conf: 0.82 + 0.1 * Math.sin(t / 900 + i), x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 });
    });
    return out;
  }

  // ---------- tracking ----------
  function updateTracker(dets) {
    const now = performance.now(), wall = Date.now();
    const bestByTool = {}, counts = {};
    for (const d of dets) {
      counts[d.name] = (counts[d.name] || 0) + 1;
      if (!bestByTool[d.name] || d.conf > bestByTool[d.name].conf) bestByTool[d.name] = d;
    }
    for (const name of Object.keys(bestByTool)) tool(name);

    for (const name of Object.keys(state.tools)) {
      const t = state.tools[name], d = bestByTool[name];
      if (d) { t.lastSeen = now; t.lastSeenWall = wall; t.conf = d.conf; t.box = d; t.count = counts[name]; }
      const present = t.lastSeen > 0 && now - t.lastSeen < CONFIG.presentTimeoutMs;
      if (present !== t.present) {
        t.present = present;
        DB.insert("tool_events", { tool_name: name, state: present ? "present" : "missing", confidence: +t.conf.toFixed(3), source: state.sim ? "sim" : "camera" });
        log("tool", `${pretty(name)} → ${present ? "present" : "missing"}`);
      }
    }

    // lock-on announcement (replaces the laser pan-tilt)
    if (state.locked) {
      const t = state.tools[state.locked];
      if (t && bestByTool[state.locked] && now - (state.lastFoundAnnounce[state.locked] || 0) > 8000) {
        state.lastFoundAnnounce[state.locked] = now;
        state.finds++;
        say(`${cap(pretty(state.locked))} found, ${where(t.box)}.`, true);
        buzz(60);
      }
    }
  }

  // ---------- geometry ----------
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

  // a box's centre relative to the visible screen centre, in [-1, 1] (cover crops the frame)
  function screenOffset(b) {
    const { s, ox, oy } = coverMap();
    const W = overlay.width || 1, H = overlay.height || 1;
    return { dx: ((((b.x1 + b.x2) / 2) * s + ox) / W) * 2 - 1, dy: ((((b.y1 + b.y2) / 2) * s + oy) / H) * 2 - 1 };
  }

  function where(b) {
    const { dx, dy } = screenOffset(b);
    const h = dx < -0.3 ? "left" : dx > 0.3 ? "right" : "center";
    const v = dy < -0.3 ? "top" : dy > 0.3 ? "bottom" : "";
    return v ? `${v} ${h}` : h === "center" ? "dead center" : `${h} side`;
  }

  // ---------- guidance ----------
  function setGuide(text, on) {
    const key = text + on;
    if (key === state.guideText) return;
    state.guideText = key;
    ui.guide.textContent = text;
    ui.guide.style.display = text ? "block" : "none";
    ui.guide.classList.toggle("on", !!on);
  }

  function guide(dets, now) {
    const target = state.locked && dets.filter((d) => d.name === state.locked).sort((a, b) => b.conf - a.conf)[0];
    if (!target) { setGuide("", false); return; }
    const off = screenOffset(target), tol = CONFIG.centerTolerance;
    if (Math.hypot(off.dx, off.dy) < tol) setGuide("● ON TARGET", true);
    else {
      // the object is left of centre -> pan the camera left to centre it
      const hz = off.dx < -tol ? "◀ LEFT" : off.dx > tol ? "RIGHT ▶" : "";
      const vt = off.dy < -tol ? "▲ UP" : off.dy > tol ? "▼ DOWN" : "";
      setGuide("PAN " + [hz, vt].filter(Boolean).join(" · "), false);
    }
    sonar(off, now);
  }

  // ---------- drawing ----------
  function draw(dets, t) {
    const { s, ox, oy, dpr } = fitOverlay();
    const W = overlay.width, H = overlay.height;
    octx.clearRect(0, 0, W, H);
    octx.lineWidth = 2 * dpr;
    octx.font = `600 ${13 * dpr}px -apple-system, system-ui, sans-serif`;

    for (const d of dets) {
      const x = d.x1 * s + ox, y = d.y1 * s + oy, w = (d.x2 - d.x1) * s, h = (d.y2 - d.y1) * s;
      const isTarget = d.name === state.locked;
      const color = isTarget ? "#ff3b30" : state.backend === "onnx" ? "#ff6a1a" : "#2ecc71";
      if (state.locked && !isTarget) octx.globalAlpha = 0.35;
      octx.strokeStyle = color;
      octx.strokeRect(x, y, w, h);
      const label = `${pretty(d.name)} ${(d.conf * 100).toFixed(0)}%`;
      const tw = octx.measureText(label).width + 10 * dpr;
      octx.fillStyle = color;
      const ly = y - 20 * dpr < 0 ? y : y - 20 * dpr; // flip inside the box at the top edge
      const lx = Math.min(Math.max(0, x), W - tw);
      octx.fillRect(lx, ly, tw, 20 * dpr);
      octx.fillStyle = "#000";
      octx.fillText(label, lx + 5 * dpr, ly + 14 * dpr);
      octx.globalAlpha = 1;
      if (isTarget) drawLaser(x + w / 2, y + h / 2, dpr, t);
    }

    // centre crosshair while locked, so "pan left/right" has a reference
    if (state.locked) {
      octx.strokeStyle = "rgba(255,255,255,.55)"; octx.lineWidth = 1.5 * dpr;
      const c = 10 * dpr;
      octx.beginPath(); octx.moveTo(W / 2 - c, H / 2); octx.lineTo(W / 2 + c, H / 2); octx.moveTo(W / 2, H / 2 - c); octx.lineTo(W / 2, H / 2 + c); octx.stroke();
    }

    // target not in frame -> arrow toward last known position
    if (state.locked && !dets.some((d) => d.name === state.locked)) {
      const tl = state.tools[state.locked];
      octx.fillStyle = "rgba(0,0,0,.55)";
      octx.fillRect(0, H / 2 - 22 * dpr, W, 44 * dpr);
      octx.fillStyle = "#fff";
      octx.textAlign = "center";
      octx.fillText(`Searching for ${pretty(state.locked)}…`, W / 2, H / 2 + 5 * dpr);
      octx.textAlign = "left";
      if (tl && tl.box) {
        const bx = ((tl.box.x1 + tl.box.x2) / 2) * s + ox, by = ((tl.box.y1 + tl.box.y2) / 2) * s + oy;
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
    const r = 40 * dpr, spin = t / 600;
    for (const a of [spin, spin + Math.PI]) { octx.beginPath(); octx.arc(x, y, r, a - 0.6, a + 0.6); octx.stroke(); }
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

  // ---------- tap a box to lock on ----------
  overlay.addEventListener("click", (e) => {
    if (!state.running) return;
    unlockAudio();
    const r = overlay.getBoundingClientRect();
    const px = (e.clientX - r.left) * (overlay.width / r.width), py = (e.clientY - r.top) * (overlay.height / r.height);
    const { s, ox, oy } = coverMap();
    const vx = (px - ox) / s, vy = (py - oy) / s, pad = 12 / s; // a little slack for fingers
    // current boxes plus each object's last box while it still counts as present (boxes flicker)
    const cands = [...state.detections, ...Object.entries(state.tools).filter(([, t]) => t.present && t.box).map(([name, t]) => ({ ...t.box, name }))];
    const hit = cands
      .filter((d) => vx >= d.x1 - pad && vx <= d.x2 + pad && vy >= d.y1 - pad && vy <= d.y2 + pad)
      .sort((a, b) => (a.x2 - a.x1) * (a.y2 - a.y1) - (b.x2 - b.x1) * (b.y2 - b.y1))[0]; // smallest box wins
    if (hit) command(hit.name === state.locked ? "unlock" : `find ${pretty(hit.name)}`, "tap");
  });

  // ---------- snapshot ----------
  const snapDialog = $("snapDialog"), snapImg = $("snapImg");
  const openDialog = (d) => { if (d.showModal) { if (!d.open) d.showModal(); } else d.setAttribute("open", ""); };
  const closeDialog = (d) => { if (d.close) d.close(); else d.removeAttribute("open"); };

  function takeSnapshot() {
    if (!state.running || !video.videoWidth) return false;
    const W = overlay.width, H = overlay.height;
    const c = document.createElement("canvas");
    c.width = W; c.height = H;
    const x = c.getContext("2d");
    const { s, ox, oy } = coverMap();
    x.drawImage(video, ox, oy, video.videoWidth * s, video.videoHeight * s);
    x.drawImage(overlay, 0, 0);
    const k = W / (overlay.getBoundingClientRect().width || W);
    const stamp = `F.O.R.G.E. Mobile · ${new Date().toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}`;
    x.font = `700 ${12 * k}px -apple-system, system-ui, sans-serif`;
    const tw = x.measureText(stamp).width;
    x.fillStyle = "rgba(0,0,0,.6)";
    x.fillRect(8 * k, H - 30 * k, tw + 16 * k, 22 * k);
    x.fillStyle = "#ff6a1a";
    x.fillText(stamp, 16 * k, H - 14 * k);
    state.snapUrl = c.toDataURL("image/jpeg", 0.9);
    snapImg.src = state.snapUrl;
    openDialog(snapDialog);
    log("system", "snapshot taken");
    return true;
  }

  function dataUrlToFile(url, name) {
    const [head, b64] = url.split(",");
    const bin = atob(b64), arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new File([arr], name, { type: head.slice(5, head.indexOf(";")) });
  }

  $("snapShare").onclick = () => {
    if (!state.snapUrl) return;
    const name = `forge-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.jpg`;
    const file = dataUrlToFile(state.snapUrl, name);
    // navigator.share must run synchronously inside this tap on iOS
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      navigator.share({ files: [file], title: "F.O.R.G.E. Mobile" }).catch(() => {});
    } else {
      const a = document.createElement("a");
      a.href = state.snapUrl; a.download = name;
      document.body.appendChild(a); a.click(); a.remove();
    }
  };
  $("snapClose").onclick = () => closeDialog(snapDialog);

  // ---------- camera ----------
  async function openCamera(facing) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
      throw new Error("camera API unavailable (the page must be opened over https)");
    if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
    // iOS needs these as properties too, before srcObject, or play() can go fullscreen / reject
    video.muted = true;
    video.playsInline = true;
    video.setAttribute("playsinline", "");
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false,
      });
    } catch (e) {
      if (e.name !== "OverconstrainedError") throw e;
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
    state.stream = stream;
    state.track = stream.getVideoTracks()[0] || null;
    video.srcObject = stream;
    // wait for dimensions; videoWidth is 0 until metadata arrives
    if (video.readyState < 1) await new Promise((res) => video.addEventListener("loadedmetadata", res, { once: true }));
    await video.play();
    let caps = {};
    try { caps = (state.track && state.track.getCapabilities && state.track.getCapabilities()) || {}; } catch {}
    state.torchOK = !!caps.torch;
    $("torchRow").hidden = !state.torchOK;
    $("setTorch").checked = false;
  }

  function setTorch(on) {
    if (!state.torchOK || !state.track) return false;
    state.track.applyConstraints({ advanced: [{ torch: on }] }).catch(() => {});
    $("setTorch").checked = on;
    return true;
  }

  async function flipCamera() {
    if (!state.running) return;
    const next = settings.facing === "user" ? "environment" : "user";
    state.detections = [];
    try {
      await openCamera(next);
      settings.facing = next; saveSettings();
      log("system", `camera: ${next === "user" ? "front" : "back"}`);
    } catch (e) {
      log("error", "couldn't switch camera: " + (e.message || e.name));
      try { await openCamera(settings.facing); } catch {}
    }
  }

  // ---------- settings dialog ----------
  const settingsDialog = $("settings");
  function syncSettingsUI() {
    $("setConf").value = settings.conf; $("setConfVal").textContent = settings.conf + "%";
    $("setVoice").checked = settings.voice;
    $("setSonar").checked = settings.sonar;
    $("setSmooth").checked = settings.smooth;
    $("setOverdue").value = settings.overdueMin; $("setOverdueVal").textContent = settings.overdueMin + " min";
  }
  $("settingsBtn").onclick = () => { unlockAudio(); syncSettingsUI(); openDialog(settingsDialog); };
  $("closeSettings").onclick = () => closeDialog(settingsDialog);
  $("setConf").oninput = (e) => { settings.conf = +e.target.value; saveSettings(); syncSettingsUI(); };
  $("setOverdue").oninput = (e) => { settings.overdueMin = +e.target.value; saveSettings(); syncSettingsUI(); };
  $("setVoice").onchange = (e) => { settings.voice = e.target.checked; saveSettings(); if (settings.voice) say("Voice on.", true); };
  $("setSonar").onchange = (e) => { settings.sonar = e.target.checked; saveSettings(); if (settings.sonar) beep(880, 0.08); };
  $("setSmooth").onchange = (e) => { settings.smooth = e.target.checked; saveSettings(); };
  $("setTorch").onchange = (e) => setTorch(e.target.checked);
  $("flipBtn").onclick = flipCamera;

  // ---------- main loop ----------
  let frames = 0, fpsT = performance.now(), busy = false, uiT = 0, benchT = 0;

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
          state.detections = smooth(state.detections, dets);
          ui.inf.textContent = Math.round(performance.now() - t0);
          updateTracker(state.detections);
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
      guide(state.detections, t);
      if (t - benchT > 500) { updateBench(); benchT = t; }
      if (t - uiT > 400) { refreshUI(); uiT = t; }
    } catch (e) { console.error(e); }
    if (t - fpsT > 1000) { ui.fps.textContent = `${frames} fps`; ui.fpsStat.textContent = frames; frames = 0; fpsT = t; }
    requestAnimationFrame(loop);
  }

  let starting = false;
  async function start() {
    if (starting || state.running) return; // ignore double taps
    starting = true;
    const btn = $("startBtn"), msg = $("startMsg");
    btn.disabled = true;
    msg.textContent = "Starting camera…";
    unlockAudio();                // both must stay synchronous in the tap (before any await)
    say("FORGE online.", true);

    try {
      await openCamera(settings.facing);
    } catch (e) {
      const why = e.name === "NotAllowedError" ? "permission denied" : (e.message || e.name);
      msg.textContent = "Camera blocked: " + why + ". Open this page over https in Safari and allow camera access (Settings › Safari › Camera).";
      btn.disabled = false;
      starting = false;
      return;
    }

    msg.textContent = "Loading model… (first time takes a few seconds)";
    try {
      const which = await loadModel(msg);
      state.sim = false;
      setModeLabel();
      log("system", `${which} loaded (${state.classes.length} object types)`);
    } catch (e) {
      state.sim = true;
      state.backend = "sim";
      state.classes = COCO.slice();
      ui.mode.textContent = "SIMULATION";
      ui.mode.className = "pill sim";
      ui.backend.textContent = "Simulation · no model could load";
      log("system", `simulation mode: ${e.message}`);
      say("Couldn't load a detection model. Running in simulation mode.", true);
    }
    showProgress(null);
    buildToolUI();

    $("startScreen").style.display = "none";
    state.running = true;
    state.startedWall = Date.now();
    starting = false;
    if (bench.active) { keepAwake(true); log("bench", "watch resumed"); }
    requestAnimationFrame(loop);
  }

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState !== "visible" || !state.running) return;
    // iOS pauses the camera when the tab/app is backgrounded; resume, and give bench watch a grace period
    if (video.paused) video.play().catch(() => {});
    state.startedWall = Date.now();
    if (bench.active) keepAwake(true); // the wake lock is dropped when hidden
  });

  // ---------- wire up ----------
  buildToolUI();
  restoreLog();
  setupVoice();
  $("startBtn").onclick = start;
  ui.mode.onclick = () => switchModel(null);
  $("findForm").onsubmit = (e) => {
    e.preventDefault();
    unlockAudio();
    const v = $("findInput").value.trim();
    if (!v) return;
    const isCommand = /^(find|locate|where|show|track|how many|help|save|start|stop|watch|what|snap|take|flash|torch|light|unlock|auto|tools|everyday|switch|status)\b/i.test(v);
    command(isCommand ? v : "find " + v, "typed");
    $("findInput").value = "";
    $("findInput").blur();
  };
  $("autoBtn").onclick = () => command("auto", "button");
  $("seeBtn").onclick = () => command("what do you see", "button");
  $("snapBtn").onclick = () => command("snapshot", "button");
  $("benchBtn").onclick = () => { unlockAudio(); command("save bench", "button"); };
  $("benchStop").onclick = () => command("stop watching", "button");
  $("clearLog").onclick = () => {
    DB.clear(); bench.history = []; saveBenchState();
    ui.log.innerHTML = ""; log("system", "history cleared"); refreshUI();
  };

  // offline support + installable app
  if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost")) {
    window.addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(() => {}));
  }

  // exposed for tests
  window.__forge = { postprocess, nms, iou, smooth, matchTool, askedFor, command, updateTracker, updateBench, screenOffset,
    state, CONFIG, settings, get bench() { return bench; } };
})();
