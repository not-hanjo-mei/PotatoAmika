// asr_gateway: the small backend adapter behind the UI's /ws/asr socket.
//
// It owns the whole ASR-facing shape so the page server does not have to:
// the page server only pipes the WebSocket upgrade through byte-for-byte
// (same arrangement as its /api/sched proxy), and this process speaks the
// protocol the page already knows:
//
//	up:   binary frames = 16 kHz mono s16le PCM (512 samples per frame),
//	      text {"type":"settings"} -> {"type":"ready","role":"asr-gateway"},
//	      text {"type":"ptt_start"} / {"type":"ptt_end"}
//	down: {"type":"asr_final","text": ...} per recognized sentence
//
// PCM is forwarded verbatim into asr_host's stdin in --stream mode;
// asr_host does Silero VAD segmentation in-process (800 ms silence rule)
// and prints one JSON line per closed segment. ptt_end is translated to
// SIGUSR1, which makes asr_host flush the open segment immediately.
//
// Run: ./asr_gateway [-listen 127.0.0.1:18120] [-asr-bin ../build/asr_host]
//      [-model-dir ../models] [-threads 2]
package main

import (
	"bufio"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"os/exec"
	"strings"
	"sync"
	"syscall"
)

// ---------- asr_host child ----------

type segLine struct {
	Text   string  `json:"text"`
	StartS float64 `json:"start_s"`
	EndS   float64 `json:"end_s"`
}

type child struct {
	mu     sync.Mutex
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	bin    string
	models string
	thread int
	onSeg  func(segLine)
}

func (c *child) start() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cmd != nil && c.cmd.ProcessState == nil {
		return nil
	}
	cmd := exec.Command(c.bin, "--model-dir", c.models, "--stream",
		"--quant", "--threads", fmt.Sprint(c.thread), "--opt", "basic")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	cmd.Stderr = &logWriter{prefix: "[asr_host] "}
	if err := cmd.Start(); err != nil {
		return err
	}
	c.cmd = cmd
	c.stdin = stdin
	log.Printf("asr_host started pid=%d", cmd.Process.Pid)
	go func() {
		sc := bufio.NewScanner(stdout)
		sc.Buffer(make([]byte, 64*1024), 1024*1024)
		for sc.Scan() {
			var s segLine
			if err := json.Unmarshal(sc.Bytes(), &s); err != nil {
				log.Printf("asr_host bad line: %q", sc.Text())
				continue
			}
			log.Printf("segment [%.3f %.3f] %q", s.StartS, s.EndS, s.Text)
			if c.onSeg != nil {
				c.onSeg(s)
			}
		}
		log.Printf("asr_host stdout closed")
	}()
	go func() {
		err := cmd.Wait()
		log.Printf("asr_host exited: %v", err)
		c.mu.Lock()
		if c.cmd == cmd {
			c.cmd = nil
			c.stdin = nil
		}
		c.mu.Unlock()
	}()
	return nil
}

func (c *child) writePCM(p []byte) {
	c.mu.Lock()
	w := c.stdin
	c.mu.Unlock()
	if w == nil {
		if err := c.start(); err != nil {
			log.Printf("asr_host restart failed: %v", err)
			return
		}
		c.mu.Lock()
		w = c.stdin
		c.mu.Unlock()
	}
	if _, err := w.Write(p); err != nil {
		log.Printf("asr_host stdin write: %v", err)
	}
}

func (c *child) flush() {
	c.mu.Lock()
	cmd := c.cmd
	c.mu.Unlock()
	if cmd != nil && cmd.Process != nil {
		if err := cmd.Process.Signal(syscall.SIGUSR1); err != nil {
			log.Printf("flush signal: %v", err)
		}
	}
}

type logWriter struct{ prefix string }

func (w *logWriter) Write(p []byte) (int, error) {
	for _, line := range strings.Split(strings.TrimRight(string(p), "\n"), "\n") {
		if line != "" {
			log.Printf("%s%s", w.prefix, line)
		}
	}
	return len(p), nil
}

// ---------- minimal WebSocket server (RFC 6455) ----------

const wsGUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

type wsConn struct {
	conn   net.Conn
	br     *bufio.Reader
	wmu    sync.Mutex
	closed bool
}

func wsAcceptConn(nc net.Conn) (*wsConn, error) {
	br := bufio.NewReader(nc)
	// request line: only /ws/asr is served here
	reqLine, err := br.ReadString('\n')
	if err != nil {
		return nil, err
	}
	parts := strings.Fields(strings.TrimRight(reqLine, "\r\n"))
	if len(parts) < 2 || (parts[1] != "/ws/asr" && !strings.HasPrefix(parts[1], "/ws/asr?")) {
		return nil, fmt.Errorf("unexpected path %q", reqLine)
	}
	var key string
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			break
		}
		if k, v, ok := strings.Cut(line, ":"); ok &&
			strings.EqualFold(strings.TrimSpace(k), "Sec-WebSocket-Key") {
			key = strings.TrimSpace(v)
		}
	}
	if key == "" {
		return nil, fmt.Errorf("no Sec-WebSocket-Key")
	}
	h := sha1.New()
	h.Write([]byte(key + wsGUID))
	accept := base64.StdEncoding.EncodeToString(h.Sum(nil))
	_, err = fmt.Fprintf(nc, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n\r\n", accept)
	if err != nil {
		return nil, err
	}
	return &wsConn{conn: nc, br: br}, nil
}

// readMessage returns (opcode, payload); opcode 1=text, 2=binary.
func (w *wsConn) readMessage() (int, []byte, error) {
	var msgOp int
	var msg []byte
	for {
		hdr := make([]byte, 2)
		if _, err := io.ReadFull(w.br, hdr); err != nil {
			return 0, nil, err
		}
		fin := hdr[0]&0x80 != 0
		op := int(hdr[0] & 0x0f)
		masked := hdr[1]&0x80 != 0
		length := uint64(hdr[1] & 0x7f)
		if length == 126 {
			b := make([]byte, 2)
			if _, err := io.ReadFull(w.br, b); err != nil {
				return 0, nil, err
			}
			length = uint64(binary.BigEndian.Uint16(b))
		} else if length == 127 {
			b := make([]byte, 8)
			if _, err := io.ReadFull(w.br, b); err != nil {
				return 0, nil, err
			}
			length = binary.BigEndian.Uint64(b)
		}
		var mask [4]byte
		if masked {
			if _, err := io.ReadFull(w.br, mask[:]); err != nil {
				return 0, nil, err
			}
		}
		payload := make([]byte, length)
		if _, err := io.ReadFull(w.br, payload); err != nil {
			return 0, nil, err
		}
		if masked {
			for i := range payload {
				payload[i] ^= mask[i%4]
			}
		}
		switch op {
		case 8: // close
			w.sendFrame(8, nil)
			return 0, nil, io.EOF
		case 9: // ping
			w.sendFrame(10, payload)
			continue
		case 10: // pong
			continue
		}
		if msgOp == 0 {
			msgOp = op
		}
		msg = append(msg, payload...)
		if fin {
			return msgOp, msg, nil
		}
	}
}

func (w *wsConn) sendFrame(op int, payload []byte) {
	w.wmu.Lock()
	defer w.wmu.Unlock()
	if w.closed {
		return
	}
	var hdr []byte
	switch {
	case len(payload) < 126:
		hdr = []byte{0x80 | byte(op), byte(len(payload))}
	case len(payload) < 65536:
		hdr = make([]byte, 4)
		hdr[0] = 0x80 | byte(op)
		hdr[1] = 126
		binary.BigEndian.PutUint16(hdr[2:], uint16(len(payload)))
	default:
		hdr = make([]byte, 10)
		hdr[0] = 0x80 | byte(op)
		hdr[1] = 127
		binary.BigEndian.PutUint64(hdr[2:], uint64(len(payload)))
	}
	if _, err := w.conn.Write(append(hdr, payload...)); err != nil {
		w.closed = true
	}
}

func (w *wsConn) sendJSON(v any) {
	b, _ := json.Marshal(v)
	w.sendFrame(1, b)
}

// ---------- gateway ----------

var (
	currentMu sync.Mutex
	current   *wsConn
)

func setCurrent(w *wsConn) {
	currentMu.Lock()
	current = w
	currentMu.Unlock()
}

func getCurrent() *wsConn {
	currentMu.Lock()
	defer currentMu.Unlock()
	return current
}

func main() {
	listen := flag.String("listen", "127.0.0.1:18120", "listen address")
	asrBin := flag.String("asr-bin", "../build/asr_host", "asr_host binary")
	modelDir := flag.String("model-dir", "../models", "model dir")
	threads := flag.Int("threads", 2, "asr_host threads")
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	ch := &child{bin: *asrBin, models: *modelDir, thread: *threads}
	ch.onSeg = func(s segLine) {
		if s.Text == "" {
			log.Printf("segment with empty text, not forwarded")
			return
		}
		if w := getCurrent(); w != nil {
			w.sendJSON(map[string]string{"type": "asr_final", "text": s.Text})
		} else {
			log.Printf("segment with no client connected, dropped")
		}
	}
	if err := ch.start(); err != nil {
		log.Fatalf("start asr_host: %v", err)
	}

	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		log.Fatalf("listen: %v", err)
	}
	log.Printf("asr gateway on ws://%s/ws/asr (asr-bin=%s model-dir=%s)", *listen, *asrBin, *modelDir)
	for {
		nc, err := ln.Accept()
		if err != nil {
			log.Printf("accept: %v", err)
			continue
		}
		go handleConn(nc, ch)
	}
}

func handleConn(nc net.Conn, ch *child) {
	defer nc.Close()
	w, err := wsAcceptConn(nc)
	if err != nil {
		log.Printf("handshake: %v", err)
		return
	}
	// wsAcceptConn built its own reader over nc; buffered bytes beyond the
	// handshake would be lost, but a WebSocket client sends nothing before
	// the 101 response, so nothing can be buffered yet.
	setCurrent(w)
	log.Printf("ws client connected")
	defer func() {
		if getCurrent() == w {
			setCurrent(nil)
		}
		ch.flush() // close any open segment; its line drops with no client
		log.Printf("ws client gone")
	}()
	for {
		op, payload, err := w.readMessage()
		if err != nil {
			return
		}
		switch op {
		case 2: // PCM frame
			ch.writePCM(payload)
		case 1: // control text
			var msg struct {
				Type string `json:"type"`
			}
			if err := json.Unmarshal(payload, &msg); err != nil {
				continue
			}
			switch msg.Type {
			case "settings":
				w.sendJSON(map[string]string{"type": "ready", "role": "asr-gateway"})
			case "ptt_start":
				log.Printf("ptt_start")
			case "ptt_end":
				log.Printf("ptt_end -> flush")
				ch.flush()
			default:
				log.Printf("msg %s (ignored)", msg.Type)
			}
		}
	}
}
