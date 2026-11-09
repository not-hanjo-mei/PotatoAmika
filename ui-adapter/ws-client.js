// Hand-rolled WS acceptance client: one text turn over /ws/dialog (ws + wss).
const net = require("net"), crypto = require("crypto"), tls = require("tls");
const u = new URL(process.argv[2]);
const text = process.argv[3] || "Hello there.";
const key = crypto.randomBytes(16).toString("base64");
const handshake = `GET ${u.pathname} HTTP/1.1\r\nHost: ${u.host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: ${key}\r\nSec-WebSocket-Version: 13\r\n\r\n`;
const sock = u.protocol === "wss:"
  ? tls.connect(parseInt(u.port || "443", 10), u.hostname, { servername: u.hostname }, () => sock.write(handshake))
  : net.connect(parseInt(u.port, 10), u.hostname, () => sock.write(handshake));
function encode(payload, op) {
  const data = Buffer.from(payload, "utf-8"); const len = data.length; let head;
  if (len < 126) { head = Buffer.alloc(2); head[1] = len; } else if (len < 65536) { head = Buffer.alloc(4); head[1] = 126; head.writeUInt16BE(len, 2); } else { head = Buffer.alloc(10); head[1] = 127; head.writeBigUInt64BE(BigInt(len), 2); }
  head[0] = 0x80 | op; head[1] |= 0x80;
  const k = crypto.randomBytes(4); const m = Buffer.from(data);
  for (let i = 0; i < m.length; i++) m[i] ^= k[i % 4];
  return Buffer.concat([head, k, m]);
}
let buf = Buffer.alloc(0), handshaken = false; const seen = [];
const timer = setTimeout(() => { console.log("TIMEOUT; seen:", seen.join(",")); process.exit(1); }, 240000);
function onMsg(s) {
  const m = JSON.parse(s); seen.push(m.type);
  if (m.type === "tts_audio") {
    const head = Buffer.from(m.audio_b64, "base64").slice(0, 4).toString();
    console.log("tts_audio idx", m.idx, "emotion", m.emotion, "kind", m.kind, "riff:", head, "text:", JSON.stringify(m.text));
    if (head !== "RIFF") { console.log("FAIL: audio not RIFF"); process.exit(1); }
  } else if (m.type === "ai_surface_done") console.log("ai_surface_done tone", m.tone, "reply:", JSON.stringify(m.reply));
  else console.log("msg", m.type, m.turn_id ? `turn=${m.turn_id}` : "", m.input ? `input=${m.input}` : "", m.text ? JSON.stringify(m.text) : "");
  if (m.type === "ready") { sock.write(encode(JSON.stringify({ type: "settings", voice_output: true }), 1)); sock.write(encode(JSON.stringify({ type: "text", text }), 1)); }
  if (m.type === "ai_end") {
    clearTimeout(timer);
    const ok = seen.includes("commit") && seen.includes("tts_audio") && seen.includes("ai_surface_done");
    console.log(ok ? "TURN-OK" : "TURN-INCOMPLETE", seen.join(","));
    process.exit(ok ? 0 : 1);
  }
  if (m.type === "ai_error") { console.log("FAIL ai_error", m.message); process.exit(1); }
}
sock.on("data", (d) => {
  buf = Buffer.concat([buf, d]);
  if (!handshaken) {
    const idx = buf.indexOf("\r\n\r\n"); if (idx < 0) return;
    handshaken = true; buf = buf.slice(idx + 4);
  }
  for (;;) {
    if (buf.length < 2) return;
    const b0 = buf[0], b1 = buf[1]; const op = b0 & 0x0f; let len = b1 & 0x7f, off = 2;
    if (len === 126) { if (buf.length < 4) return; len = buf.readUInt16BE(2); off = 4; }
    else if (len === 127) { if (buf.length < 10) return; len = Number(buf.readBigUInt64BE(2)); off = 10; }
    if (buf.length < off + len) return;
    const payload = buf.slice(off, off + len); buf = buf.slice(off + len);
    if (op === 1) onMsg(payload.toString("utf-8"));
    if (op === 9) sock.write(encode("", 10));
  }
});
sock.on("error", (e) => { console.log("SOCK ERROR", e.message); process.exit(1); });
