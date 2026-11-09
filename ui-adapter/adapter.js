#!/usr/bin/env node
"use strict";
/*
 * PotatoAmika adapter for the ORIGINAL OpenAmika Use page.
 *
 * Serves the original page files (pristine, except use.html / use.js /
 * common.js which carry the documented removals in ./build — see
 * REMOVALS.md) and translates the endpoints the page expects to the
 * PotatoAmika scheduler (API.md) and the ASR gateway. Standalone speech
 * (the wizard's lines, name previews) goes through the scheduler's
 * POST /v1/say, which uses the one resident tts_host — this adapter
 * spawns no TTS process of its own (the phone cannot afford a second
 * ~375 MB host; see scheduler API.md for the say contract, incl. its
 * 409-while-a-turn-is-running rule).
 *
 * Capabilities PotatoAmika definitively does not have are NOT faked here:
 * their page parts were removed, and their endpoints return explicit
 * 501/404 (see REMOVALS.md). Nothing in this file invents data.
 */
const http = require("http");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const net = require("net");

const PORT = parseInt(process.env.PORT || "8372", 10);
const UI_PRISTINE = process.env.UI_PRISTINE || path.join(__dirname, "pristine");
const UI_PATCHED = process.env.UI_PATCHED || path.join(__dirname, "patched");
const MODELS_DIR = process.env.MODELS_DIR || path.join(__dirname, "models-live2d");
const STATIC_DIR = process.env.STATIC_DIR || path.join(__dirname, "static");
const DATA_DIR = process.env.DATA_DIR || path.join(__dirname, "data");
const SCHED = process.env.SCHED || "http://127.0.0.1:18100";
const ASR_WS = process.env.ASR_WS || "ws://127.0.0.1:18120/ws/asr";

fs.mkdirSync(DATA_DIR, { recursive: true });

const log = (...a) => console.log(new Date().toISOString(), ...a);

/* ------------------------------------------------------------------ *
 * Emotion vocabularies (copied from the original speech/voice_adapter.py)
 * ------------------------------------------------------------------ */
const DEFAULT_MAP = {
  polite_neutral: "neutral1", polite_smile: "happy1", polite_happy: "happy1",
  polite_caring: "caring1", polite_sad: "sad1", polite_worried: "worried1",
  polite_cold: "cold1", polite_angry: "annoyed1", polite_embarrassed: "embarrassed1",
  polite_surprise: "surprise1", polite_fear: "fear1",
  casual_neutral: "neutral1", casual_smile: "happy1", casual_happy: "happy2",
  casual_very_happy: "happy3", casual_laughing: "playful3", casual_playful: "playful2",
  casual_caring: "caring2", casual_relieved: "relieved1", casual_shy: "shy2",
  casual_sad: "sad2", casual_worried: "worried1", casual_frustrated: "frustrated1",
  casual_angry: "angry2", casual_jealous: "jealous2", casual_embarrassed: "embarrassed1",
  casual_surprise: "surprise1", casual_fear: "fear2", casual_tired: "tired1",
  casual_mumbling: "mumbling1",
  serious_neutral: "firm1", serious_protective: "protective1", serious_cold: "cold3",
  serious_worried: "worried2", serious_anxious: "anxiety1", serious_disappointed: "disappointed1",
  serious_sad: "sad2", serious_angry: "angry3", serious_very_angry: "angry4",
  serious_fear: "fear3", serious_surprise: "surprise2",
  intimate_neutral: "neutral1", intimate_smile: "happy1", intimate_happy: "happy2",
  intimate_very_happy: "happy3", intimate_caring: "caring3", intimate_very_caring: "caring3",
  intimate_playful: "playful1", intimate_shy: "shy2", intimate_very_shy: "shy3",
  intimate_relieved: "relieved2", intimate_sad: "sad2", intimate_very_sad: "sad3",
  intimate_tearful: "sad3", intimate_worried: "worried1", intimate_jealous: "jealous1",
  intimate_tired: "tired2", intimate_sleepy: "sleepy1", intimate_mumbling: "mumbling1",
  whisper_neutral: "whisper1", whisper_smile: "whisper2", whisper_happy: "happy1",
  whisper_caring: "caring1", whisper_relieved: "relieved2", whisper_shy: "shy1",
  whisper_sad: "sad1", whisper_tearful: "sad1", whisper_worried: "worried1",
  whisper_sleepy: "sleepy1",
};
const EMOTIONS = Object.keys(DEFAULT_MAP); // 69 V2 keys
const V1_IDS = new Set(Object.values(DEFAULT_MAP));
["awe1", "lonely1", "awkwardness1", "sarcastic1", "pout1", "proud1", "breathy1", "monotone1"].forEach((t) => V1_IDS.add(t));
// V1 -> V2, preferring the casual register, then polite/intimate/serious/whisper.
const V1_TO_V2 = {};
for (const reg of ["casual", "polite", "intimate", "serious", "whisper"]) {
  for (const [v2, v1] of Object.entries(DEFAULT_MAP)) {
    if (v2.startsWith(reg + "_") && !(v1 in V1_TO_V2)) V1_TO_V2[v1] = v2;
  }
}
const v1toV2 = (tag) => V1_TO_V2[tag] || "casual_neutral";

/* ------------------------------------------------------------------ *
 * Small helpers
 * ------------------------------------------------------------------ */
const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
  ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
  ".webp": "image/webp", ".gif": "image/gif", ".svg": "image/svg+xml",
  ".moc3": "application/octet-stream", ".wav": "audio/wav", ".mp3": "audio/mpeg",
  ".ogg": "audio/ogg", ".txt": "text/plain; charset=utf-8", ".md": "text/plain; charset=utf-8",
};
function sendJSON(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { "Content-Type": "application/json; charset=utf-8", "Content-Length": Buffer.byteLength(body) });
  res.end(body);
}
function sendFile(res, file) {
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) return sendJSON(res, 404, { detail: "Not found." });
    res.writeHead(200, { "Content-Type": MIME[path.extname(file).toLowerCase()] || "application/octet-stream", "Content-Length": st.size });
    fs.createReadStream(file).pipe(res);
  });
}
function readBody(req, cap = 30 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    const chunks = []; let n = 0;
    req.on("data", (c) => { n += c.length; if (n > cap) { reject(new Error("body too large")); req.destroy(); } else chunks.push(c); });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}
async function readJSON(req) {
  const buf = await readBody(req);
  if (!buf.length) return {};
  return JSON.parse(buf.toString("utf-8"));
}
function schedJSON(method, p, body) {
  return new Promise((resolve, reject) => {
    const data = body ? JSON.stringify(body) : null;
    const req = http.request(SCHED + p, { method, headers: data ? { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(data) } : {} }, (res) => {
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => {
        let parsed = null;
        try { parsed = JSON.parse(Buffer.concat(chunks).toString("utf-8")); } catch { /* keep null */ }
        resolve({ status: res.statusCode, body: parsed });
      });
    });
    req.on("error", reject);
    if (data) req.write(data);
    req.end();
  });
}
let healthzCache = { at: 0, body: null };
async function schedulerHealth() {
  if (Date.now() - healthzCache.at < 1000 && healthzCache.body) return healthzCache.body;
  const r = await schedJSON("GET", "/healthz");
  if (r.status !== 200 || !r.body) throw new Error("scheduler healthz failed");
  healthzCache = { at: Date.now(), body: r.body };
  return r.body;
}

/* ------------------------------------------------------------------ *
 * Adapter-local persisted state: persona identity + stage
 * ------------------------------------------------------------------ */
function loadState(name, dflt) {
  try { return { ...dflt, ...JSON.parse(fs.readFileSync(path.join(DATA_DIR, name), "utf-8")) }; }
  catch { return { ...dflt }; }
}
function saveState(name, obj) {
  fs.writeFileSync(path.join(DATA_DIR, name), JSON.stringify(obj, null, 2));
}
const personaState = loadState("persona.json", { user_name: "", city: "", timezone: "", onboarded: false, say_as: "" });
const stageState = loadState("stage.json", { blur: 12, bg: null }); // bg: filename inside DATA_DIR

/* ------------------------------------------------------------------ *
 * Avatar models (port of the original speech/api.py behaviour)
 * ------------------------------------------------------------------ */
function walkFiles(dir, exts, base = dir, out = []) {
  let entries = [];
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return out; }
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) walkFiles(full, exts, base, out);
    else if (exts.some((x) => e.name.toLowerCase().endsWith(x))) out.push(path.relative(base, full).split(path.sep).join("/"));
  }
  return out.sort();
}
function modelDir(id) {
  const d = path.join(MODELS_DIR, id);
  return path.basename(id) === id && fs.existsSync(d) && fs.statSync(d).isDirectory() ? d : null;
}
function banksDoc(id) {
  const dir = modelDir(id); if (!dir) return null;
  let doc = { config: {}, banks: {} };
  try { doc = JSON.parse(fs.readFileSync(path.join(dir, "banks.json"), "utf-8")); } catch { /* fresh */ }
  doc.config = doc.config || {}; doc.banks = doc.banks || {};
  const stems = new Set(walkFiles(dir, [".motion3.json"]).map((r) => path.basename(r, ".motion3.json")));
  const fill = (key) => {
    const names = [...stems].filter((s) => s === key || s.startsWith(key + ".")).sort();
    return { emotes: [], movements: names.map((name) => ({ name, likelihood: 100 })) };
  };
  for (const key of EMOTIONS) if (!(key in doc.banks)) doc.banks[key] = fill(key);
  if (!("__idle__" in doc.banks)) {
    const names = [...stems].filter((s) => s.startsWith("idle")).sort();
    doc.banks.__idle__ = { emotes: [], movements: names.map((name) => ({ name, likelihood: 100 })) };
  }
  return doc;
}
function avatarModelsList() {
  const models = [];
  let ids = [];
  try { ids = fs.readdirSync(MODELS_DIR, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name).sort(); } catch { /* none */ }
  for (const id of ids) {
    const dir = modelDir(id);
    const model3 = walkFiles(dir, [".model3.json"])[0];
    if (!model3) continue;
    const cfg = (banksDoc(id) || {}).config || {};
    models.push({
      id,
      name: cfg.display_name || id,
      model3_url: `/avatar_models/${id}/${model3}`,
      motions_url: `/api/avatar/models/${id}/motions`,
      banks_url: `/api/avatar/models/${id}/banks`,
      assets_prefix: `/avatar_models/${id}/`,
      mouth_open_param: cfg.mouth_open_param || "ParamMouthOpenY",
      mouth_form_param: cfg.mouth_form_param || "ParamMouthForm",
      role: cfg.role || "character",
      stage: {
        scale: cfg.scale != null ? cfg.scale : null,
        nx: cfg.nx != null ? cfg.nx : null,
        ny: cfg.ny != null ? cfg.ny : null,
        head_y: cfg.head_y != null ? cfg.head_y : null,
        busy_expression: cfg.busy_expression || null,
      },
    });
  }
  return models;
}
function versionedModel3(id, rel) {
  const dir = modelDir(id);
  const file = path.join(dir, rel);
  const data = JSON.parse(fs.readFileSync(file, "utf-8"));
  const refs = data.FileReferences;
  if (!refs) return data;
  const base = path.dirname(rel);
  const stamp = (r) => {
    let v = "0";
    try { v = String(Math.floor(fs.statSync(path.join(dir, base, r)).mtimeMs / 1000)); } catch { /* missing */ }
    return `${r}?v=${v}`;
  };
  if (Array.isArray(refs.Textures)) refs.Textures = refs.Textures.map(stamp);
  for (const k of ["Moc", "Physics", "DisplayInfo"]) if (typeof refs[k] === "string") refs[k] = stamp(refs[k]);
  const nonEmpty = (o) => o && typeof o === "object" && Object.values(o).some((v) => (Array.isArray(v) ? v.length : !!v));
  if (!nonEmpty(refs.Motions)) {
    const files = walkFiles(dir, [".motion3.json"]);
    refs.Motions = { Default: files.map((f) => ({ File: path.relative(base, f).split(path.sep).join("/") || f, FadeInTime: 0.5, FadeOutTime: 0.5 })) };
  }
  if (!nonEmpty(refs.Expressions)) {
    refs.Expressions = walkFiles(dir, [".exp3.json"]).map((f) => ({ Name: path.basename(f, ".exp3.json"), File: path.relative(base, f).split(path.sep).join("/") || f }));
  }
  return data;
}

/* ------------------------------------------------------------------ *
 * tts_say: forwarded to the scheduler's POST /v1/say, which speaks
 * through the one resident tts_host. No private host here (it cost
 * ~375 MB on the preview; the phone has no room for a second one).
 * The scheduler refuses with 409 while a dialogue turn is running;
 * that surfaces to the page as a skipped line, never a hang.
 * Concurrent say calls are serialized through a local chain so the
 * page's back-to-back wizard lines keep their order.
 * ------------------------------------------------------------------ */
let sayChain = Promise.resolve();
function sayOnce(text, tag) {
  const run = async () => {
    const r = await schedJSON("POST", "/v1/say", { text, tag });
    if (r.status === 200 && r.body && r.body.file) return r.body;
    const err = (r.body && r.body.error) || `scheduler /v1/say returned ${r.status}`;
    const e = new Error(err);
    e.status = r.status;
    throw e;
  };
  const p = sayChain.then(run, run);
  sayChain = p.catch(() => {});
  return p;
}

/* ------------------------------------------------------------------ *
 * Minimal RFC 6455: frame codec shared by the server and the client.
 * ------------------------------------------------------------------ */
function wsAcceptKey(key) {
  return crypto.createHash("sha1").update(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").digest("base64");
}
function wsEncode(payload, opcode, mask) {
  const data = Buffer.isBuffer(payload) ? payload : Buffer.from(String(payload), "utf-8");
  const len = data.length;
  let head;
  if (len < 126) { head = Buffer.alloc(2); head[1] = len; }
  else if (len < 65536) { head = Buffer.alloc(4); head[1] = 126; head.writeUInt16BE(len, 2); }
  else { head = Buffer.alloc(10); head[1] = 127; head.writeBigUInt64BE(BigInt(len), 2); }
  head[0] = 0x80 | opcode;
  if (!mask) return Buffer.concat([head, data]);
  head[1] |= 0x80;
  const key = crypto.randomBytes(4);
  const masked = Buffer.from(data);
  for (let i = 0; i < masked.length; i++) masked[i] ^= key[i % 4];
  return Buffer.concat([head, key, masked]);
}
class WsParser {
  constructor(onMessage, onBinary, onClose) {
    this.buf = Buffer.alloc(0); this.frag = null; this.fragOp = 0;
    this.onMessage = onMessage; this.onBinary = onBinary; this.onClose = onClose;
  }
  push(chunk) {
    this.buf = Buffer.concat([this.buf, chunk]);
    for (;;) {
      if (this.buf.length < 2) return;
      const b0 = this.buf[0], b1 = this.buf[1];
      const fin = !!(b0 & 0x80), op = b0 & 0x0f, masked = !!(b1 & 0x80);
      let len = b1 & 0x7f, off = 2;
      if (len === 126) { if (this.buf.length < 4) return; len = this.buf.readUInt16BE(2); off = 4; }
      else if (len === 127) { if (this.buf.length < 10) return; len = Number(this.buf.readBigUInt64BE(2)); off = 10; }
      const maskLen = masked ? 4 : 0;
      if (this.buf.length < off + maskLen + len) return;
      let payload = this.buf.slice(off + maskLen, off + maskLen + len);
      if (masked) {
        const key = this.buf.slice(off, off + 4);
        payload = Buffer.from(payload);
        for (let i = 0; i < payload.length; i++) payload[i] ^= key[i % 4];
      }
      this.buf = this.buf.slice(off + maskLen + len);
      if (op === 8) { this.onClose && this.onClose(); return; }
      if (op === 9) { this.onPing && this.onPing(payload); continue; }
      if (op === 10) continue;
      if (op === 0) { if (this.frag) this.frag.push(payload); }
      else { this.frag = [payload]; this.fragOp = op; }
      if (fin && this.frag) {
        const full = Buffer.concat(this.frag); const fop = this.fragOp; this.frag = null;
        if (fop === 2) this.onBinary && this.onBinary(full);
        else this.onMessage && this.onMessage(full.toString("utf-8"));
      }
    }
  }
}

/* WebSocket client (to the ASR gateway). */
function wsConnect(url) {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const key = crypto.randomBytes(16).toString("base64");
    const sock = net.connect(parseInt(u.port, 10), u.hostname, () => {
      sock.write(
        `GET ${u.pathname} HTTP/1.1\r\nHost: ${u.host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n` +
        `Sec-WebSocket-Key: ${key}\r\nSec-WebSocket-Version: 13\r\n\r\n`);
    });
    let head = Buffer.alloc(0); let done = false;
    const client = {
      sock, closed: false, onMessage: null, onBinary: null, onClose: null,
      sendText(s) { if (!this.closed) sock.write(wsEncode(s, 1, true)); },
      sendBinary(b) { if (!this.closed) sock.write(wsEncode(b, 2, true)); },
      close() { this.closed = true; try { sock.end(); } catch { /* already gone */ } },
    };
    const parser = new WsParser(
      (s) => client.onMessage && client.onMessage(s),
      (b) => client.onBinary && client.onBinary(b),
      () => { client.closed = true; client.onClose && client.onClose(); });
    parser.onPing = (p) => sock.write(wsEncode(p, 10, true));
    sock.on("data", (d) => {
      if (!done) {
        head = Buffer.concat([head, d]);
        const idx = head.indexOf("\r\n\r\n");
        if (idx < 0) return;
        const statusLine = head.slice(0, idx).toString("utf-8").split("\r\n")[0];
        if (!statusLine.includes("101")) { reject(new Error("ASR gateway handshake: " + statusLine)); sock.destroy(); return; }
        done = true; resolve(client);
        const rest = head.slice(idx + 4);
        if (rest.length) parser.push(rest);
        return;
      }
      parser.push(d);
    });
    sock.on("error", (e) => { if (!done) reject(e); client.closed = true; client.onClose && client.onClose(); });
    sock.on("close", () => { client.closed = true; client.onClose && client.onClose(); });
  });
}

/* SSE consumer for the scheduler event stream. */
function openSSE(sessionId, onEvent) {
  const req = http.get(`${SCHED}/v1/events?session=${encodeURIComponent(sessionId)}`, (res) => {
    let buf = "";
    res.on("data", (d) => {
      buf += d.toString("utf-8");
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
        let ev = "message", data = "";
        for (const line of raw.split("\n")) {
          if (line.startsWith("event:")) ev = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (data) { try { const parsed = JSON.parse(data); onEvent(parsed.type || ev, parsed); } catch { /* malformed event */ } }
      }
    });
  });
  req.on("error", () => {});
  return req;
}

/* ------------------------------------------------------------------ *
 * /ws/dialog translation.
 *
 * Page protocol (original) <-> scheduler API.md:
 *   text            -> commit echo + POST /v1/chat; SSE tone/sentence_audio
 *                      become tts_audio chunks; turn_done becomes
 *                      ai_surface_done + ai_end.
 *   stop            -> POST /v1/interrupt.
 *   context_inject  -> queued, prepended to the next turn's text.
 *   settings        -> voice_output only gates tts_audio forwarding.
 *   binary PCM      -> ASR gateway; each asr_final starts a voice turn.
 * The gateway emits no partials and no speech_start, so the page's live
 * STT line stays empty and barge-in lands at commit time (the page
 * flushes playback on commit) instead of at speech start.
 * ------------------------------------------------------------------ */
class Dialog {
  constructor(sock, sendFrame) {
    this.sock = sock; this.sendFrame = sendFrame;
    this.session = null; this.sse = null;
    this.turn = null; this.turnSeq = 0;
    this.voiceOutput = true; this.ctxQueue = "";
    this.asr = null; this.asrOpening = null; this.asrActive = false; this.pcmIdle = null;
    this.dead = false;
  }
  send(obj) { if (!this.dead) this.sendFrame(wsEncode(JSON.stringify(obj), 1, false)); }
  async init() {
    const r = await schedJSON("POST", "/v1/sessions", {});
    this.session = (r.body && r.body.session_id) || "default";
    this.sse = openSSE(this.session, (ev, data) => this.onSSE(ev, data));
    this.send({ type: "ready", sample_rate: 16000, frame_samples: 512, format: "pcm_s16le_mono", brain: true, tts: true });
  }
  async rotateSession() { // page "reset": abandon the session, start a fresh one
    if (this.sse) this.sse.destroy();
    const r = await schedJSON("POST", "/v1/sessions", {});
    this.session = (r.body && r.body.session_id) || "default";
    this.sse = openSSE(this.session, (ev, data) => this.onSSE(ev, data));
  }
  onSSE(ev, data) {
    const turn = this.turn;
    if (ev === "tone" && turn && !turn.closed) turn.toneV2 = v1toV2(data.tag);
    else if (ev === "sentence_audio" && turn && !turn.audioClosed && !turn.stopped) {
      const s = data.sentence || {};
      if (this.voiceOutput && s.wav) {
        fs.readFile(s.wav, (err, buf) => {
          if (err || turn.audioClosed || turn.stopped) return;
          this.send({
            type: "tts_audio", turn_id: turn.id, idx: s.index != null ? s.index : turn.idx++,
            text: s.text || "", audio_b64: buf.toString("base64"),
            emotion: v1toV2(s.tag) || turn.toneV2 || "casual_neutral",
            kind: "surface", surface_last: false, pause_after: 0,
          });
        });
      }
    } else if (ev === "turn_done" && turn) {
      turn.sseDone = true;
      this.finalize(turn, data.reply != null ? data.reply : (turn.result || {}).reply, data.tag || (turn.result || {}).tag);
    }
    else if (ev === "interrupted" && turn) { turn.closed = true; turn.audioClosed = true; clearTimeout(turn.grace); if (this.turn === turn) this.turn = null; }
    else if (ev === "error" && turn && !turn.closed) {
      turn.closed = true; if (this.turn === turn) this.turn = null;
      this.send({ type: "ai_error", message: String(data.error || "turn failed") });
    }
  }
  finalize(turn, reply, tagV1) {
    if (turn.closed) return;
    turn.closed = true; turn.audioClosed = true;
    clearTimeout(turn.grace);
    if (this.turn === turn) this.turn = null;
    if (turn.stopped) { this.send({ type: "ai_end" }); return; }
    this.send({
      type: "ai_surface_done", reply: reply || "", tone: turn.toneV2 || v1toV2(tagV1),
      mode: "say", surface_tts_done: true,
    });
    this.send({ type: "ai_end" });
  }
  startTurn(text, input) {
    if (this.dead || !text) return;
    if (this.turn && !this.turn.closed) return; // one turn at a time, like the page
    const turn = { id: `t${++this.turnSeq}-${Date.now().toString(36)}`, closed: false, stopped: false, toneV2: null, idx: 0 };
    this.turn = turn;
    this.send({ type: "commit", text, turn_id: turn.id, input: input || "text" });
    let full = text;
    if (this.ctxQueue) { full = `${this.ctxQueue}\n${text}`; this.ctxQueue = ""; }
    schedJSON("POST", "/v1/chat", { session_id: this.session, text: full }).then((r) => {
      // The chat response can land before the last SSE audio events; keep
      // it as the fallback result and let turn_done (or a grace timer)
      // close the turn, so no sentence_audio is dropped.
      if (r.status === 200 && r.body) turn.result = { reply: r.body.reply, tag: r.body.tag };
      else if (r.body && r.body.partial && r.body.partial.reply) turn.result = { reply: r.body.partial.reply, tag: r.body.partial.tag };
      if (turn.result) {
        if (turn.sseDone || turn.stopped) this.finalize(turn, turn.result.reply, turn.result.tag);
        else turn.grace = setTimeout(() => this.finalize(turn, turn.result.reply, turn.result.tag), 1500);
      } else if (turn.stopped) this.finalize(turn, "", null);
      else if (!turn.closed) {
        turn.closed = true; if (this.turn === turn) this.turn = null;
        this.send({ type: "ai_error", message: (r.body && r.body.error) || `scheduler returned ${r.status}` });
      }
    }).catch((e) => {
      if (!turn.closed) {
        turn.closed = true; if (this.turn === turn) this.turn = null;
        this.send({ type: "ai_error", message: String(e.message || e) });
      }
    });
  }
  async stop() {
    const turn = this.turn;
    if (turn) turn.stopped = true;
    try { await schedJSON("POST", "/v1/interrupt", { session_id: this.session }); } catch { /* best effort */ }
    if (turn && !turn.closed) this.finalize(turn, "", null);
  }
  async ensureAsr() {
    if (this.asr && !this.asr.closed) return this.asr;
    if (this.asrOpening) return this.asrOpening;
    this.asrOpening = wsConnect(ASR_WS).then((c) => {
      this.asr = c; this.asrOpening = null;
      c.onMessage = (s) => {
        let m; try { m = JSON.parse(s); } catch { return; }
        if (m.type === "asr_final" && m.text) this.startTurn(m.text, "voice");
      };
      c.onClose = () => { if (this.asr === c) this.asr = null; };
      c.sendText(JSON.stringify({ type: "settings" }));
      return c;
    }).catch((e) => { this.asrOpening = null; log("ASR gateway connect failed:", e.message); return null; });
    return this.asrOpening;
  }
  onBinary(buf) {
    this.ensureAsr().then((c) => { if (c) c.sendBinary(buf); });
    // The page streams only while its mic is on and sends no mic-off
    // signal; a >1 s gap in frames flushes the gateway's current segment.
    if (!this.asrActive) { this.asrActive = true; this.ensureAsr().then((c) => { if (c) c.sendText(JSON.stringify({ type: "ptt_start" })); }); }
    clearTimeout(this.pcmIdle);
    this.pcmIdle = setTimeout(() => {
      this.asrActive = false;
      if (this.asr && !this.asr.closed) this.asr.sendText(JSON.stringify({ type: "ptt_end" }));
    }, 1000);
  }
  async onText(s) {
    let m; try { m = JSON.parse(s); } catch { return; }
    switch (m.type) {
      case "settings": if (typeof m.voice_output === "boolean") this.voiceOutput = m.voice_output; break;
      case "hello": this.ensureAsr(); break;
      case "text": this.startTurn(String(m.text || ""), "text"); break;
      case "context_inject": this.ctxQueue = String(m.text || ""); break;
      case "stop": this.stop(); break;
      case "reset": this.rotateSession().catch(() => {}); break;
      case "tts_playing": case "tts_finished": case "timeline_reset": break; // bookkeeping only in the original too
      default: break; // inject: removed from the page; unknown types ignored
    }
  }
  destroy() {
    this.dead = true;
    clearTimeout(this.pcmIdle);
    if (this.sse) this.sse.destroy();
    if (this.asr) this.asr.close();
  }
}

/* ------------------------------------------------------------------ *
 * HTTP routing
 * ------------------------------------------------------------------ */
const PATCHED_FILES = new Set(["use.html", "use.js", "common.js"]);
const NOT_SUPPORTED = (what) => ({ detail: `Not supported in PotatoAmika: ${what}` });

function timezones() {
  try { return Intl.supportedValuesOf("timeZone"); } catch { return ["UTC"]; }
}

async function handleAPI(req, res, u) {
  const p = u.pathname;
  const method = req.method;

  /* ---- avatar ---- */
  if (p === "/api/avatar/models" && method === "GET") {
    return sendJSON(res, 200, { ok: true, models: avatarModelsList(), emotions: EMOTIONS });
  }
  let m = p.match(/^\/api\/avatar\/models\/([^/]+)\/motions$/);
  if (m && method === "GET") {
    const id = decodeURIComponent(m[1]); const dir = modelDir(id);
    if (!dir) return sendJSON(res, 404, { detail: "Unknown model." });
    const out = [];
    for (const rel of walkFiles(dir, [".motion3.json", ".exp3.json"])) {
      const isMotion = rel.endsWith(".motion3.json");
      out.push({
        type: isMotion ? "motion" : "expression",
        name: path.basename(rel, isMotion ? ".motion3.json" : ".exp3.json"),
        file: path.basename(rel),
        path: `/avatar_models/${id}/${rel}`,
      });
    }
    return sendJSON(res, 200, out);
  }
  m = p.match(/^\/api\/avatar\/models\/([^/]+)\/banks$/);
  if (m && method === "GET") {
    const id = decodeURIComponent(m[1]); const doc = banksDoc(id);
    if (!doc) return sendJSON(res, 404, { detail: "Unknown model." });
    return sendJSON(res, 200, doc);
  }
  if (m && method === "POST") {
    const id = decodeURIComponent(m[1]); const dir = modelDir(id);
    if (!dir) return sendJSON(res, 404, { detail: "Unknown model." });
    const body = await readJSON(req);
    if (!body || typeof body.banks !== "object" || body.banks === null) {
      return sendJSON(res, 400, { detail: "Body must carry a banks object." });
    }
    const cur = banksDoc(id);
    const clean = {};
    for (const [key, bank] of Object.entries(body.banks)) {
      if (!bank || typeof bank !== "object") continue;
      clean[key] = {
        emotes: Array.isArray(bank.emotes) ? bank.emotes.filter((x) => typeof x === "string") : [],
        movements: Array.isArray(bank.movements) ? bank.movements.filter((x) => x && typeof x.name === "string")
          .map((x) => ({ name: x.name, likelihood: Math.max(0, Math.min(100, parseInt(x.likelihood, 10) || 0)) })) : [],
      };
    }
    const EDITABLE_CFG = ["display_name", "mouth_open_param", "mouth_form_param", "scale", "nx", "ny", "head_y", "busy_expression"];
    const cfg = { ...cur.config };
    if (body.config && typeof body.config === "object") {
      for (const k of EDITABLE_CFG) if (k in body.config) cfg[k] = body.config[k];
    }
    const doc = { config: cfg, banks: clean };
    fs.writeFileSync(path.join(dir, "banks.json"), JSON.stringify(doc, null, 2));
    return sendJSON(res, 200, { ok: true, banks: doc });
  }

  /* ---- avatar model files (with the original's model3 transform) ---- */
  m = p.match(/^\/avatar_models\/([^/]+)\/(.+)$/);
  if (m && method === "GET") {
    const id = decodeURIComponent(m[1]); const rel = decodeURIComponent(m[2]);
    const dir = modelDir(id);
    if (!dir || rel.includes("..")) return sendJSON(res, 404, { detail: "Not found." });
    const file = path.join(dir, rel);
    if (!file.startsWith(dir) || !fs.existsSync(file)) return sendJSON(res, 404, { detail: "Not found." });
    if (rel.endsWith(".model3.json")) {
      try { return sendJSON(res, 200, versionedModel3(id, rel)); }
      catch { return sendFile(res, file); }
    }
    return sendFile(res, file);
  }

  /* ---- stage ---- */
  if (p === "/api/stage" && method === "GET") {
    let bg = null;
    if (stageState.bg) {
      try { bg = `/api/stage/background?v=${Math.floor(fs.statSync(path.join(DATA_DIR, stageState.bg)).mtimeMs / 1000)}`; } catch { bg = null; }
    }
    return sendJSON(res, 200, { ok: true, background: bg, blur: stageState.blur });
  }
  if (p === "/api/stage" && method === "POST") {
    const body = await readJSON(req);
    if (body.clear_background) {
      if (stageState.bg) { try { fs.unlinkSync(path.join(DATA_DIR, stageState.bg)); } catch { /* gone */ } }
      stageState.bg = null;
    }
    if (typeof body.blur === "number") stageState.blur = Math.max(0, Math.min(40, body.blur));
    saveState("stage.json", stageState);
    return sendJSON(res, 200, { ok: true, blur: stageState.blur, background: stageState.bg ? "/api/stage/background" : null });
  }
  if (p === "/api/stage/background" && method === "GET") {
    if (!stageState.bg) return sendJSON(res, 404, { detail: "No background set." });
    return sendFile(res, path.join(DATA_DIR, stageState.bg));
  }
  if (p === "/api/stage/background" && method === "POST") {
    const ct = req.headers["content-type"] || "";
    const bm = ct.match(/boundary=(.+)$/);
    if (!bm) return sendJSON(res, 400, { detail: "Expected multipart form data." });
    const buf = await readBody(req, 26 * 1024 * 1024);
    const boundary = Buffer.from("--" + bm[1].trim());
    // Minimal multipart scan: first part carrying a filename wins.
    let pos = 0; let saved = null;
    while ((pos = buf.indexOf(boundary, pos)) >= 0) {
      const headEnd = buf.indexOf("\r\n\r\n", pos);
      if (headEnd < 0) break;
      const head = buf.slice(pos, headEnd).toString("utf-8");
      const next = buf.indexOf(boundary, headEnd + 4);
      if (next < 0) break;
      const dataEnd = next - 2; // CRLF before boundary
      const fm = head.match(/filename="([^"]*)"/);
      if (fm && fm[1]) {
        const ext = (path.extname(fm[1]) || ".png").toLowerCase();
        if (![".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif"].includes(ext)) {
          return sendJSON(res, 400, { detail: "Unsupported image type." });
        }
        const name = "stage-bg" + ext;
        fs.writeFileSync(path.join(DATA_DIR, name), buf.slice(headEnd + 4, dataEnd));
        stageState.bg = name; saveState("stage.json", stageState);
        saved = name; break;
      }
      pos = next;
    }
    if (!saved) return sendJSON(res, 400, { detail: "No image file in upload." });
    return sendJSON(res, 200, { ok: true, background: "/api/stage/background" });
  }

  /* ---- health / status / probe ---- */
  if (p === "/health" && method === "GET") {
    return sendJSON(res, 200, { status: "ok", app: "PotatoAmika", model: "Minico-M1-1105 (llama.cpp, local)", device: "cpu", cuda_available: false, gpu_name: null });
  }
  if (p === "/api/build_f/status" && method === "GET") {
    try {
      const hz = await schedulerHealth();
      return sendJSON(res, 200, {
        ok: true, app: "PotatoAmika scheduler",
        template_check: {}, memory: { events: hz.memory_n || 0, degraded: false },
      });
    } catch { return sendJSON(res, 502, { detail: "Scheduler unreachable." }); }
  }
  if (p === "/api/tts_probe" && method === "GET") {
    try {
      const hz = await schedulerHealth();
      return sendJSON(res, 200, { ok: !!hz.tts, url: "scheduler tts_host (local)", character: "MekaHime_Amika", status: hz.tts ? 200 : 503 });
    } catch { return sendJSON(res, 200, { ok: false, url: "scheduler tts_host (local)", character: "MekaHime_Amika", error: "scheduler unreachable" }); }
  }

  /* ---- tts_say (first-run wizard + name preview) ---- */
  if (p === "/api/tts_say" && method === "POST") {
    const body = await readJSON(req);
    const text = String(body.text || "").trim();
    if (!text) return sendJSON(res, 400, { detail: "Empty text." });
    const tag = V1_IDS.has(body.voice) ? body.voice : (DEFAULT_MAP[body.emotion] || "neutral1");
    try {
      const r = await sayOnce(text, tag); // scheduler /v1/say; throws on 409 (turn running) and other failures
      const audio = fs.readFileSync(r.file).toString("base64");
      return sendJSON(res, 200, {
        ok: true, audio, audio_format: "wav", duration: r.duration_s || 0,
        spoken: text, emotion: body.emotion || v1toV2(r.tag_used || tag),
        voice_tag: r.tag_used || tag, language: "en",
      });
    } catch (e) { return sendJSON(res, 200, { ok: false, error: String(e.message || e) }); }
  }

  /* ---- build_f: models / persona / pronunciation ---- */
  if (p === "/api/build_f/models" && method === "GET") {
    const opt = { key: "local-minico", label: "Minico-M1-1105 · llama.cpp (local)", transport: "llama.cpp", model: "Minico-M1-1105-Q4_K_M" };
    const roles = ["surface", "deep", "reflection", "memory_write"];
    return sendJSON(res, 200, {
      options: Object.fromEntries(roles.map((r) => [r, [opt]])),
      selections: Object.fromEntries(roles.map((r) => [r, "local-minico"])),
    });
  }
  if (p === "/api/build_f/persona" && method === "GET") {
    return sendJSON(res, 200, {
      ok: true, onboarded: !!personaState.onboarded, persona_name: "Amika",
      user_name: personaState.user_name || "", city: personaState.city || "",
      timezone: personaState.timezone || "", timezones: timezones(),
    });
  }
  if (p === "/api/build_f/persona/identity" && method === "POST") {
    const body = await readJSON(req);
    if (typeof body.user_name === "string") personaState.user_name = body.user_name;
    if (typeof body.city === "string") personaState.city = body.city;
    if (typeof body.timezone === "string") personaState.timezone = body.timezone;
    if (typeof body.onboarded === "boolean") personaState.onboarded = body.onboarded;
    saveState("persona.json", personaState);
    return sendJSON(res, 200, { ok: true });
  }
  if (p === "/api/name_pronunciation" && method === "GET") {
    return sendJSON(res, 200, { ok: true, name: personaState.user_name || "", say_as: personaState.say_as || "" });
  }
  if (p === "/api/name_pronunciation" && method === "POST") {
    const body = await readJSON(req);
    if (typeof body.say_as === "string") personaState.say_as = body.say_as;
    if (typeof body.name === "string") personaState.user_name = body.name;
    saveState("persona.json", personaState);
    return sendJSON(res, 200, { ok: true });
  }
  if (p === "/api/voice_map" && method === "GET") {
    return sendJSON(res, 200, { ok: true, enabled: true, voice: "MekaHime_Amika", map: DEFAULT_MAP });
  }

  /* ---- shaped, truthful empties: features that exist but are off/empty ---- */
  if (p === "/api/preflight" && method === "GET") {
    return sendJSON(res, 200, { at: new Date().toISOString(), checks: [], summary: {} });
  }
  if (p === "/api/build_f/messengers" && method === "GET") {
    return sendJSON(res, 200, { ok: true, enabled: false, messengers: [] });
  }
  if (p === "/api/build_f/tools" && method === "GET") {
    // The tool layer is disabled in PotatoAmika v1 (user decision); the
    // page parts that displayed it were removed, this stays for the
    // health poll's sake and reports the true state.
    return sendJSON(res, 200, { ok: true, enabled: false, servers: {}, dialogue: [], jobs: [], pending: [] });
  }
  if (p === "/api/build_f/openrouter_key" && method === "GET") {
    return sendJSON(res, 200, { ok: true, present: false, hint: "" });
  }
  if (p === "/api/build_f/ollama" && method === "GET") {
    return sendJSON(res, 200, { ok: true, reachable: false, base_url: "", live_version: null, models: [], in_use: {} });
  }
  if (p === "/api/vision" && method === "GET") {
    // UNDECIDED (reported to the user, not deleted): PotatoAmika's vision
    // is the scheduler's one-shot /v1/look, not this page's live camera
    // feed. Until the user decides, the endpoint reports the page's own
    // disabled state, so the camera button gives the page's own toast.
    return sendJSON(res, 200, {
      ok: true, enabled: false, url: null, service: null, description: null,
      last_description: null, description_age_s: null, camera_clients: 0,
      subscribers: 0, her_line: null, language: "en",
    });
  }
  if (p === "/sfx_manifest" && method === "GET") {
    // Genuinely empty: no SFX audio files are installed (the upstream
    // speech/static/sfx holds only a README).
    return sendJSON(res, 200, { dir: "/static/sfx/", files: [] });
  }

  /* ---- explicit unsupported: capabilities PotatoAmika does not have ---- */
  const unsupported = [
    [/^\/api\/build_f\/models$/, "switching brain providers from the page (the brain is the local Minico)"],
    [/^\/api\/build_f\/openrouter_key$/, "OpenRouter accounts (all inference is local)"],
    [/^\/api\/build_f\/ollama(\/.*)?$/, "Ollama (the brain runs in llama.cpp)"],
    [/^\/api\/build_f\/preset$/, "brain presets"],
    [/^\/api\/build_f\/meter\/reset$/, "a usage meter (local inference is not metered)"],
    [/^\/api\/build_f\/persona\/save$/, "editing persona files from the page"],
    [/^\/api\/build_f\/persona\/simple$/, "the simple persona editor"],
    [/^\/api\/build_f\/messengers$/, "messenger bots (cut from PotatoAmika)"],
    [/^\/api\/build_f\/tools(\/.*)?$/, "the tool layer (disabled in v1)"],
    [/^\/api\/build_f\/wipe$/, "memory wipe from the page"],
    [/^\/api\/build_f\/reset$/, "full reset from the page"],
    [/^\/api\/build_f\/iot(\/.*)?$/, "IoT devices (not part of PotatoAmika)"],
    [/^\/api\/voice_map(\/reset)?$/, "editing the voice map (it ships fixed)"],
  ];
  for (const [re, what] of unsupported) {
    if (re.test(p)) return sendJSON(res, 501, NOT_SUPPORTED(what));
  }
  return sendJSON(res, 404, { detail: "Not found." });
}

const server = http.createServer(async (req, res) => {
  try {
    const u = new URL(req.url, "http://x");
    const p = u.pathname;
    if (p === "/" || p === "/use") return sendFile(res, path.join(UI_PATCHED, "use.html"));
    if (p === "/about") return sendFile(res, path.join(UI_PRISTINE, "about.html"));
    if (p === "/setup" || p === "/tools") return sendJSON(res, 404, { detail: "This page is not part of the PotatoAmika build (see REMOVALS.md)." });
    if (p === "/flow") return sendFile(res, path.join(STATIC_DIR, "flow.html"));
    if (p.startsWith("/ui/")) {
      const rel = p.slice(4);
      if (rel.includes("..")) return sendJSON(res, 404, { detail: "Not found." });
      if (PATCHED_FILES.has(rel)) return sendFile(res, path.join(UI_PATCHED, rel));
      return sendFile(res, path.join(UI_PRISTINE, rel));
    }
    if (p.startsWith("/static/")) {
      const rel = p.slice(8);
      if (rel.includes("..")) return sendJSON(res, 404, { detail: "Not found." });
      return sendFile(res, path.join(STATIC_DIR, rel));
    }
    if (p.startsWith("/api/") || p.startsWith("/avatar_models/") || p === "/health" || p === "/sfx_manifest") {
      return await handleAPI(req, res, u);
    }
    return sendJSON(res, 404, { detail: "Not found." });
  } catch (e) {
    log("request error", req.url, e.message);
    if (!res.headersSent) sendJSON(res, 500, { detail: String(e.message || e) });
  }
});

server.on("upgrade", (req, sock, head) => {
  const u = new URL(req.url, "http://x");
  const key = req.headers["sec-websocket-key"];
  if (u.pathname === "/ws/dialog" && key) {
    sock.write(
      "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n" +
      `Sec-WebSocket-Accept: ${wsAcceptKey(key)}\r\n\r\n`);
    const dialog = new Dialog(sock, (frame) => { if (!sock.destroyed) sock.write(frame); });
    const parser = new WsParser(
      (s) => dialog.onText(s),
      (b) => dialog.onBinary(b),
      () => { dialog.destroy(); sock.end(); });
    parser.onPing = (pl) => { if (!sock.destroyed) sock.write(wsEncode(pl, 10, false)); };
    dialog.init().catch((e) => { log("dialog init failed:", e.message); dialog.send({ type: "ai_error", message: "scheduler unreachable" }); });
    sock.on("data", (d) => parser.push(d));
    sock.on("close", () => dialog.destroy());
    sock.on("error", () => dialog.destroy());
    if (head && head.length) parser.push(head);
  } else {
    // /ws/camera and anything else: no such channel in this build.
    sock.destroy();
  }
});

server.listen(PORT, "0.0.0.0", () => log(`orig-ui adapter on :${PORT} (ui=${UI_PRISTINE} patched=${UI_PATCHED} models=${MODELS_DIR})`));
