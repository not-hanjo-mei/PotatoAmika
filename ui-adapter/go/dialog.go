package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

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

type chatResult struct {
	reply string
	tag   string
}

type turnT struct {
	id          string
	closed      bool
	stopped     bool
	audioClosed bool
	sseDone     bool
	toneV2      string
	idx         int
	result      *chatResult
	grace       *time.Timer
}

type sseSentence struct {
	Wav   *string `json:"wav"`
	Index *int    `json:"index"`
	Text  *string `json:"text"`
	Tag   *string `json:"tag"`
}

type sseData struct {
	Type     *string      `json:"type"`
	Tag      *string      `json:"tag"`
	Sentence *sseSentence `json:"sentence"`
	Reply    *string      `json:"reply"`
	Error    *string      `json:"error"`
}

type dialog struct {
	sendFrame func([]byte)
	mu        sync.Mutex

	session    string
	sseCancel  context.CancelFunc
	turn       *turnT
	turnSeq    int
	voiceOut   bool
	ctxQueue   string
	asr        *wsClient
	asrOpening bool
	asrWaiters []func(*wsClient)
	asrActive  bool
	pcmIdle    *time.Timer
	dead       bool
}

func newDialog(sendFrame func([]byte)) *dialog {
	return &dialog{sendFrame: sendFrame, voiceOut: true}
}

func (d *dialog) send(obj interface{}) {
	d.mu.Lock()
	dead := d.dead
	d.mu.Unlock()
	if dead {
		return
	}
	d.sendFrame(wsEncode(marshalJSON(obj), 1, false))
}

func (d *dialog) init() error {
	r, err := schedJSON("POST", "/v1/sessions", orderedJSON{})
	if err != nil {
		return err
	}
	session := "default"
	if r.body != nil {
		if s, ok := rawStrField(r.body, "session_id"); ok && s != "" {
			session = s
		}
	}
	d.mu.Lock()
	d.session = session
	d.mu.Unlock()
	d.openSSE(session)
	d.send(orderedJSON{
		{"type", "ready"}, {"sample_rate", 16000}, {"frame_samples", 512},
		{"format", "pcm_s16le_mono"}, {"brain", true}, {"tts", true},
	})
	return nil
}

func (d *dialog) rotateSession() {
	d.mu.Lock()
	if d.sseCancel != nil {
		d.sseCancel()
		d.sseCancel = nil
	}
	d.mu.Unlock()
	r, err := schedJSON("POST", "/v1/sessions", orderedJSON{})
	if err != nil {
		return // JS: a failed rotate leaves the old session id and no SSE
	}
	session := "default"
	if r.body != nil {
		if s, ok := rawStrField(r.body, "session_id"); ok && s != "" {
			session = s
		}
	}
	d.mu.Lock()
	d.session = session
	d.mu.Unlock()
	d.openSSE(session)
}

func (d *dialog) openSSE(session string) {
	ctx, cancel := context.WithCancel(context.Background())
	d.mu.Lock()
	d.sseCancel = cancel
	d.mu.Unlock()
	go func() {
		req, err := http.NewRequestWithContext(ctx, "GET",
			schedBase+"/v1/events?session="+url.QueryEscape(session), nil)
		if err != nil {
			return
		}
		res, err := httpClient.Do(req)
		if err != nil {
			return
		}
		defer res.Body.Close()
		var buf []byte
		chunk := make([]byte, 8192)
		for {
			n, err := res.Body.Read(chunk)
			if n > 0 {
				buf = append(buf, chunk[:n]...)
				for {
					idx := bytes.Index(buf, []byte("\n\n"))
					if idx < 0 {
						break
					}
					raw := buf[:idx]
					buf = buf[idx+2:]
					ev := "message"
					data := ""
					for _, line := range strings.Split(string(raw), "\n") {
						if strings.HasPrefix(line, "event:") {
							ev = strings.TrimSpace(line[len("event:"):])
						} else if strings.HasPrefix(line, "data:") {
							data += strings.TrimSpace(line[len("data:"):])
						}
					}
					if data == "" {
						continue
					}
					var parsed sseData
					if err := json.Unmarshal([]byte(data), &parsed); err != nil {
						continue
					}
					name := ev
					if parsed.Type != nil && *parsed.Type != "" {
						name = *parsed.Type
					}
					d.onSSE(name, &parsed)
				}
			}
			if err != nil {
				return
			}
		}
	}()
}

func strOr(p *string, dflt string) string {
	if p == nil {
		return dflt
	}
	return *p
}

func (d *dialog) onSSE(ev string, data *sseData) {
	d.mu.Lock()
	turn := d.turn
	d.mu.Unlock()
	switch ev {
	case "tone":
		if turn == nil {
			return
		}
		d.mu.Lock()
		if !turn.closed {
			turn.toneV2 = v1toV2(strOr(data.Tag, ""))
		}
		d.mu.Unlock()
	case "sentence_audio":
		if turn == nil {
			return
		}
		d.mu.Lock()
		skip := turn.audioClosed || turn.stopped
		voice := d.voiceOut
		d.mu.Unlock()
		if skip {
			return
		}
		s := data.Sentence
		if s == nil {
			s = &sseSentence{}
		}
		if !voice || s.Wav == nil || *s.Wav == "" {
			return
		}
		wavPath := *s.Wav
		go func() {
			buf, err := os.ReadFile(wavPath)
			if err != nil {
				return
			}
			d.mu.Lock()
			if turn.audioClosed || turn.stopped {
				d.mu.Unlock()
				return
			}
			idx := turn.idx
			if s.Index != nil {
				idx = *s.Index
			} else {
				turn.idx++
			}
			d.mu.Unlock()
			emotion := v1toV2(strOr(s.Tag, ""))
			d.send(orderedJSON{
				{"type", "tts_audio"}, {"turn_id", turn.id}, {"idx", idx},
				{"text", strOr(s.Text, "")}, {"audio_b64", base64.StdEncoding.EncodeToString(buf)},
				{"emotion", emotion}, {"kind", "surface"}, {"surface_last", false},
				{"pause_after", 0},
			})
		}()
	case "turn_done":
		if turn == nil {
			return
		}
		d.mu.Lock()
		turn.sseDone = true
		var res *chatResult
		if turn.result != nil {
			cp := *turn.result
			res = &cp
		}
		d.mu.Unlock()
		reply := ""
		tag := ""
		if data.Reply != nil {
			reply = *data.Reply
		} else if res != nil {
			reply = res.reply
		}
		if data.Tag != nil && *data.Tag != "" {
			tag = *data.Tag
		} else if res != nil {
			tag = res.tag
		}
		d.finalize(turn, reply, tag)
	case "interrupted":
		if turn == nil {
			return
		}
		d.mu.Lock()
		turn.closed = true
		turn.audioClosed = true
		if turn.grace != nil {
			turn.grace.Stop()
		}
		if d.turn == turn {
			d.turn = nil
		}
		d.mu.Unlock()
	case "error":
		if turn == nil {
			return
		}
		d.mu.Lock()
		if turn.closed {
			d.mu.Unlock()
			return
		}
		turn.closed = true
		if d.turn == turn {
			d.turn = nil
		}
		d.mu.Unlock()
		msg := "turn failed"
		if data.Error != nil && *data.Error != "" {
			msg = *data.Error
		}
		d.send(orderedJSON{{"type", "ai_error"}, {"message", msg}})
	}
}

func (d *dialog) finalize(turn *turnT, reply, tagV1 string) {
	d.mu.Lock()
	if turn.closed {
		d.mu.Unlock()
		return
	}
	turn.closed = true
	turn.audioClosed = true
	if turn.grace != nil {
		turn.grace.Stop()
	}
	if d.turn == turn {
		d.turn = nil
	}
	stopped := turn.stopped
	tone := turn.toneV2
	d.mu.Unlock()
	if stopped {
		d.send(orderedJSON{{"type", "ai_end"}})
		return
	}
	if tone == "" {
		tone = v1toV2(tagV1)
	}
	d.send(orderedJSON{
		{"type", "ai_surface_done"}, {"reply", reply}, {"tone", tone},
		{"mode", "say"}, {"surface_tts_done", true},
	})
	d.send(orderedJSON{{"type", "ai_end"}})
}

func (d *dialog) startTurn(text, input string) {
	d.mu.Lock()
	if d.dead || text == "" {
		d.mu.Unlock()
		return
	}
	if d.turn != nil && !d.turn.closed {
		d.mu.Unlock()
		return
	}
	d.turnSeq++
	turn := &turnT{id: fmt.Sprintf("t%d-%s", d.turnSeq, strconv.FormatInt(time.Now().UnixMilli(), 36))}
	d.turn = turn
	session := d.session
	full := text
	if d.ctxQueue != "" {
		full = d.ctxQueue + "\n" + text
		d.ctxQueue = ""
	}
	d.mu.Unlock()

	d.send(orderedJSON{
		{"type", "commit"}, {"text", text}, {"turn_id", turn.id}, {"input", input},
	})
	go func() {
		r, err := schedJSON("POST", "/v1/chat", orderedJSON{
			{"session_id", session}, {"text", full},
		})
		d.chatDone(turn, r, err)
	}()
}

func (d *dialog) chatDone(turn *turnT, r schedResp, err error) {
	if err != nil {
		d.mu.Lock()
		closed := turn.closed
		if !closed {
			turn.closed = true
			if d.turn == turn {
				d.turn = nil
			}
		}
		d.mu.Unlock()
		if !closed {
			d.send(orderedJSON{{"type", "ai_error"}, {"message", err.Error()}})
		}
		return
	}
	var result *chatResult
	if r.status == 200 && r.body != nil {
		result = &chatResult{reply: strOr2(r.body, "reply"), tag: strOr2(r.body, "tag")}
	} else if r.body != nil {
		if praw, ok := r.body["partial"]; ok {
			var p rawObj
			if json.Unmarshal(praw, &p) == nil && p != nil {
				if rep, ok := rawStrField(p, "reply"); ok && rep != "" {
					result = &chatResult{reply: rep, tag: strOr2(p, "tag")}
				}
			}
		}
	}
	d.mu.Lock()
	if result != nil {
		turn.result = result
		if turn.sseDone || turn.stopped {
			d.mu.Unlock()
			d.finalize(turn, result.reply, result.tag)
			return
		}
		res := result
		turn.grace = time.AfterFunc(1500*time.Millisecond, func() {
			d.finalize(turn, res.reply, res.tag)
		})
		d.mu.Unlock()
		return
	}
	stopped := turn.stopped
	closed := turn.closed
	if !stopped && !closed {
		turn.closed = true
		if d.turn == turn {
			d.turn = nil
		}
	}
	d.mu.Unlock()
	if stopped {
		d.finalize(turn, "", "")
		return
	}
	if !closed {
		msg := fmt.Sprintf("scheduler returned %d", r.status)
		if r.body != nil {
			if e, ok := rawStrField(r.body, "error"); ok && e != "" {
				msg = e
			}
		}
		d.send(orderedJSON{{"type", "ai_error"}, {"message", msg}})
	}
}

func strOr2(obj rawObj, key string) string {
	if obj == nil {
		return ""
	}
	s, _ := rawStrField(obj, key)
	return s
}

func (d *dialog) stop() {
	d.mu.Lock()
	turn := d.turn
	if turn != nil {
		turn.stopped = true
	}
	session := d.session
	d.mu.Unlock()
	_, _ = schedJSON("POST", "/v1/interrupt", orderedJSON{{"session_id", session}})
	if turn != nil {
		d.mu.Lock()
		closed := turn.closed
		d.mu.Unlock()
		if !closed {
			d.finalize(turn, "", "")
		}
	}
}

func (d *dialog) ensureAsr(cb func(*wsClient)) {
	d.mu.Lock()
	if d.asr != nil && !d.asr.closed {
		c := d.asr
		d.mu.Unlock()
		cb(c)
		return
	}
	if d.asrOpening {
		d.asrWaiters = append(d.asrWaiters, cb)
		d.mu.Unlock()
		return
	}
	d.asrOpening = true
	d.asrWaiters = []func(*wsClient){cb}
	d.mu.Unlock()
	go func() {
		c, err := wsConnect(asrWS)
		d.mu.Lock()
		d.asrOpening = false
		waiters := d.asrWaiters
		d.asrWaiters = nil
		if err == nil {
			d.asr = c
			c.onMessage = func(s string) {
				var m struct {
					Type string `json:"type"`
					Text string `json:"text"`
				}
				if json.Unmarshal([]byte(s), &m) != nil {
					return
				}
				if m.Type == "asr_final" && m.Text != "" {
					d.startTurn(m.Text, "voice")
				}
			}
			c.onClose = func() {
				d.mu.Lock()
				if d.asr == c {
					d.asr = nil
				}
				d.mu.Unlock()
			}
			c.sendText(string(marshalJSON(orderedJSON{{"type", "settings"}})))
		} else {
			logf("ASR gateway connect failed: %v", err)
		}
		d.mu.Unlock()
		for _, w := range waiters {
			if err == nil {
				w(c)
			} else {
				w(nil)
			}
		}
	}()
}

func (d *dialog) onBinary(buf []byte) {
	payload := append([]byte{}, buf...)
	d.ensureAsr(func(c *wsClient) {
		if c != nil {
			c.sendBinary(payload)
		}
	})
	d.mu.Lock()
	activate := !d.asrActive
	if activate {
		d.asrActive = true
	}
	d.mu.Unlock()
	if activate {
		d.ensureAsr(func(c *wsClient) {
			if c != nil {
				c.sendText(string(marshalJSON(orderedJSON{{"type", "ptt_start"}})))
			}
		})
	}
	// The page streams only while its mic is on and sends no mic-off
	// signal; a >1 s gap in frames flushes the gateway's current segment.
	d.mu.Lock()
	if d.pcmIdle != nil {
		d.pcmIdle.Stop()
	}
	d.pcmIdle = time.AfterFunc(time.Second, func() {
		d.mu.Lock()
		d.asrActive = false
		asr := d.asr
		d.mu.Unlock()
		if asr != nil && !asr.closed {
			asr.sendText(string(marshalJSON(orderedJSON{{"type", "ptt_end"}})))
		}
	})
	d.mu.Unlock()
}

func (d *dialog) onText(s string) {
	var m rawObj
	if json.Unmarshal([]byte(s), &m) != nil || m == nil {
		return
	}
	typ, _ := rawStrField(m, "type")
	switch typ {
	case "settings":
		if b, ok := rawBoolField(m, "voice_output"); ok {
			d.mu.Lock()
			d.voiceOut = b
			d.mu.Unlock()
		}
	case "hello":
		d.ensureAsr(func(*wsClient) {})
	case "text":
		t, _ := rawStrField(m, "text")
		d.startTurn(t, "text")
	case "context_inject":
		t, _ := rawStrField(m, "text")
		d.mu.Lock()
		d.ctxQueue = t
		d.mu.Unlock()
	case "stop":
		go d.stop()
	case "reset":
		go d.rotateSession()
	case "tts_playing", "tts_finished", "timeline_reset":
		// bookkeeping only in the original too
	default:
		// inject: removed from the page; unknown types ignored
	}
}

func (d *dialog) destroy() {
	d.mu.Lock()
	d.dead = true
	if d.pcmIdle != nil {
		d.pcmIdle.Stop()
	}
	cancel := d.sseCancel
	asr := d.asr
	d.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if asr != nil {
		asr.close()
	}
}
