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
// Backend history: this gateway used to forward PCM into asr_host
// (Moonshine Tiny + Silero VAD, --stream mode). Since the Omni
// integration (2026-11-11) the backend is the resident LFM3-Omni brain
// server: the gateway segments the PCM stream itself (energy VAD below;
// the 800 ms silence rule is the same one asr_host used) and sends each
// closed segment to the brain's /v1/chat/completions as an input_audio
// content part, exactly the request shape the spike's ASR gate scored
// CER/WER 0 with. ptt_end closes the open segment immediately (the role
// SIGUSR1 played for asr_host). The wire contract above is unchanged.
//
// The energy VAD is simpler than Silero: soft onsets can clip a first
// syllable the neural VAD would have caught, and steady noise above the
// threshold holds a segment open. A 200 ms pre-roll is kept so speech
// onsets survive the threshold crossing.
//
// Run: ./asr_gateway [-listen 127.0.0.1:18120]
//
//	[-brain-url http://127.0.0.1:18091] [-vad-threshold 500]
//
// (The legacy -asr-bin/-model-dir/-threads flags are still accepted so
// existing launch scripts keep working; they are ignored.)
package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

// ---------- segmentation + Omni transcription ----------

const (
	sampleRate   = 16000
	frameSamples = 512 // 32 ms, the page worklet's frame

	silenceCloseSamples = 12800  // 800 ms of silence closes a segment
	preRollSamples      = 3200   // 200 ms kept before speech onset
	tailSamples         = 1600   // 100 ms kept after speech end
	minSpeechSamples    = 1600   // shorter blips are dropped as clicks
	maxSegSamples       = 480000 // 30 s hard cap per segment
)

type segLine struct {
	Text   string  `json:"text"`
	StartS float64 `json:"start_s"`
	EndS   float64 `json:"end_s"`
}

type segment struct {
	pcm    []int16
	startS float64
	endS   float64
}

type asrBackend struct {
	brainURL  string
	threshold float64
	client    *http.Client
	onSeg     func(segLine)
	queue     chan segment

	mu             sync.Mutex
	pending        []byte // PCM bytes not yet consumed into a frame
	clock          int64  // samples consumed so far (stream clock)
	inSpeech       bool
	seg            []int16 // open segment, pre-roll included
	segStartSample int64   // stream position of seg[0]
	speechStart    int64   // stream position of the first speech frame
	lastSpeech     int64   // stream position just past the last speech frame
	silenceRun     int     // consecutive below-threshold samples in speech
	preRoll        []int16
}

func newASRBackend(brainURL string, threshold float64, onSeg func(segLine)) *asrBackend {
	b := &asrBackend{
		brainURL:  strings.TrimRight(brainURL, "/"),
		threshold: threshold,
		client:    &http.Client{}, // deadlines via ctx per request
		onSeg:     onSeg,
		queue:     make(chan segment, 4),
	}
	go b.worker()
	return b
}

func (b *asrBackend) writePCM(p []byte) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.pending = append(b.pending, p...)
	for len(b.pending) >= frameSamples*2 {
		frame := make([]int16, frameSamples)
		for i := range frame {
			frame[i] = int16(binary.LittleEndian.Uint16(b.pending[i*2:]))
		}
		b.pending = b.pending[frameSamples*2:]
		b.processFrame(frame)
	}
}

// processFrame consumes one 32 ms frame. Caller holds b.mu.
func (b *asrBackend) processFrame(f []int16) {
	frameStart := b.clock
	b.clock += frameSamples
	var sum float64
	for _, s := range f {
		sum += float64(s) * float64(s)
	}
	rms := math.Sqrt(sum / frameSamples)
	if rms >= b.threshold {
		if !b.inSpeech {
			b.inSpeech = true
			b.seg = append(b.seg[:0], b.preRoll...)
			b.segStartSample = frameStart - int64(len(b.preRoll))
			b.speechStart = frameStart
			b.preRoll = nil
		}
		b.seg = append(b.seg, f...)
		b.lastSpeech = b.clock
		b.silenceRun = 0
		if len(b.seg) >= maxSegSamples {
			b.closeSegmentLocked()
		}
		return
	}
	// Below threshold.
	if b.inSpeech {
		b.seg = append(b.seg, f...)
		b.silenceRun += frameSamples
		if b.silenceRun >= silenceCloseSamples {
			b.closeSegmentLocked()
		}
		return
	}
	b.preRoll = append(b.preRoll, f...)
	if len(b.preRoll) > preRollSamples {
		b.preRoll = append([]int16(nil), b.preRoll[len(b.preRoll)-preRollSamples:]...)
	}
}

// closeSegmentLocked trims the open segment to speech + tail and queues
// it for transcription. Caller holds b.mu.
func (b *asrBackend) closeSegmentLocked() {
	if !b.inSpeech {
		return
	}
	b.inSpeech = false
	b.silenceRun = 0
	keep := int(b.lastSpeech-b.segStartSample) + tailSamples
	if keep > len(b.seg) {
		keep = len(b.seg)
	}
	pcm := append([]int16(nil), b.seg[:keep]...)
	b.seg = nil
	b.preRoll = nil
	if b.lastSpeech-b.speechStart < minSpeechSamples {
		log.Printf("segment dropped: %.0f ms of speech is below the minimum",
			float64(b.lastSpeech-b.speechStart)*1000/sampleRate)
		return
	}
	s := segment{
		pcm:    pcm,
		startS: float64(b.speechStart) / sampleRate,
		endS:   float64(b.lastSpeech) / sampleRate,
	}
	select {
	case b.queue <- s:
	default:
		log.Printf("segment dropped: transcription queue full")
	}
}

// flush closes the open segment now (ptt_end, client disconnect).
func (b *asrBackend) flush() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.closeSegmentLocked()
}

func (b *asrBackend) worker() {
	for s := range b.queue {
		t0 := time.Now()
		text, err := b.transcribe(s)
		if err != nil {
			log.Printf("transcribe [%.2f %.2f]: %v", s.startS, s.endS, err)
			continue
		}
		log.Printf("segment [%.3f %.3f] %q (%.2fs audio, %d ms wall)",
			s.startS, s.endS, text, float64(len(s.pcm))/sampleRate,
			time.Since(t0).Milliseconds())
		if b.onSeg != nil {
			b.onSeg(segLine{Text: text, StartS: s.startS, EndS: s.endS})
		}
	}
}

// transcribe sends one segment to the brain server as an input_audio
// chat completion — the exact request shape the spike's ASR gate used
// (asr_server.py against llama-server, enable_thinking off, temp 0).
func (b *asrBackend) transcribe(s segment) (string, error) {
	dur := float64(len(s.pcm)) / sampleRate
	maxTokens := 80 + int(dur*20)
	if maxTokens > 512 {
		maxTokens = 512
	}
	body := map[string]any{
		"messages": []any{map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "input_audio", "input_audio": map[string]any{
					"data":   base64.StdEncoding.EncodeToString(encodeWAV(s.pcm)),
					"format": "wav",
				}},
				map[string]any{"type": "text", "text": "Transcribe the audio exactly, word for word. Output only the transcription."},
			},
		}},
		"temperature":          0,
		"max_tokens":           maxTokens,
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
	}
	data, err := json.Marshal(body)
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "POST", b.brainURL+"/v1/chat/completions", bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := b.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return "", fmt.Errorf("brain asr: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("brain asr: no choices")
	}
	return stripThink(out.Choices[0].Message.Content), nil
}

// encodeWAV wraps 16 kHz mono s16le PCM in a minimal WAV header.
func encodeWAV(pcm []int16) []byte {
	var buf bytes.Buffer
	dataLen := len(pcm) * 2
	buf.WriteString("RIFF")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(36+dataLen))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(16))    // fmt size
	_ = binary.Write(&buf, binary.LittleEndian, uint16(1))     // PCM
	_ = binary.Write(&buf, binary.LittleEndian, uint16(1))     // mono
	_ = binary.Write(&buf, binary.LittleEndian, uint32(16000)) // rate
	_ = binary.Write(&buf, binary.LittleEndian, uint32(32000)) // byte rate
	_ = binary.Write(&buf, binary.LittleEndian, uint16(2))     // block align
	_ = binary.Write(&buf, binary.LittleEndian, uint16(16))    // bits
	buf.WriteString("data")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(dataLen))
	_ = binary.Write(&buf, binary.LittleEndian, pcm)
	return buf.Bytes()
}

// stripThink mirrors the scheduler's helper: the brain may wrap output
// in a think block even with thinking nominally off.
func stripThink(s string) string {
	for {
		i := strings.Index(s, "<think>")
		if i < 0 {
			break
		}
		j := strings.Index(s[i:], "</think>")
		if j < 0 {
			s = s[:i]
			break
		}
		s = s[:i] + s[i+j+len("</think>"):]
	}
	if i := strings.LastIndex(s, "</think>"); i >= 0 {
		s = s[i+len("</think>"):]
	}
	return strings.Trim(s, " \t\r\n")
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
	brainURL := flag.String("brain-url", "http://127.0.0.1:18091", "brain llama-server base URL (Omni ASR backend)")
	threshold := flag.Float64("vad-threshold", 500, "energy VAD threshold, int16 RMS per 32 ms frame")
	// Legacy Moonshine-era flags: accepted so existing launch scripts
	// (deploy.sh) keep starting the gateway; the backend no longer
	// spawns asr_host, so they have no effect.
	_ = flag.String("asr-bin", "", "ignored (legacy asr_host path)")
	_ = flag.String("model-dir", "", "ignored (legacy Moonshine model dir)")
	_ = flag.Int("threads", 0, "ignored (legacy asr_host threads)")
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	legacy := []string{}
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "asr-bin" || f.Name == "model-dir" || f.Name == "threads" {
			legacy = append(legacy, f.Name)
		}
	})
	if len(legacy) > 0 {
		log.Printf("legacy flags ignored (%s): ASR backend is the Omni brain server", strings.Join(legacy, ", "))
	}

	be := newASRBackend(*brainURL, *threshold, func(s segLine) {
		if s.Text == "" {
			log.Printf("segment with empty text, not forwarded")
			return
		}
		if w := getCurrent(); w != nil {
			w.sendJSON(map[string]string{"type": "asr_final", "text": s.Text})
		} else {
			log.Printf("segment with no client connected, dropped")
		}
	})

	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		log.Fatalf("listen: %v", err)
	}
	log.Printf("asr gateway on ws://%s/ws/asr (backend: omni brain at %s, vad-threshold %.0f)", *listen, *brainURL, *threshold)
	for {
		nc, err := ln.Accept()
		if err != nil {
			log.Printf("accept: %v", err)
			continue
		}
		go handleConn(nc, be)
	}
}

func handleConn(nc net.Conn, be *asrBackend) {
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
		be.flush() // close any open segment; its line drops with no client
		log.Printf("ws client gone")
	}()
	for {
		op, payload, err := w.readMessage()
		if err != nil {
			return
		}
		switch op {
		case 2: // PCM frame
			be.writePCM(payload)
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
				be.flush()
			default:
				log.Printf("msg %s (ignored)", msg.Type)
			}
		}
	}
}
