package main

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

// HTTP API for the frontend (see API.md). Transport is plain HTTP + SSE:
// the event schema below is the contract; a WebSocket transport can carry
// the identical event objects later without changing any field.

type apiServer struct {
	orch    *Orchestrator
	turnSeq atomic.Int64
}

func newID() string {
	var b [8]byte
	rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

func (a *apiServer) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", a.health)
	mux.HandleFunc("POST /v1/sessions", a.createSession)
	mux.HandleFunc("POST /v1/chat", a.chat)
	mux.HandleFunc("GET /v1/events", a.events)
	mux.HandleFunc("POST /v1/interrupt", a.interrupt)
	mux.HandleFunc("POST /v1/say", a.say)
	mux.HandleFunc("GET /v1/memory", a.memoryList)
	mux.HandleFunc("POST /v1/look", a.look)
	return mux
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func (a *apiServer) health(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
	defer cancel()
	writeJSON(w, 200, map[string]any{
		"ok":       true,
		"brain_up": a.orch.brain.healthy(ctx),
		"tts":      a.orch.tts != nil,
		"tools":    a.orch.toolsOn,
		"memory_n": a.orch.mem.Count(),
		"vision":   a.orch.mgr.visionConfigured(),
	})
}

func (a *apiServer) createSession(w http.ResponseWriter, r *http.Request) {
	id := newID()
	a.orch.session(id)
	writeJSON(w, 200, map[string]any{"session_id": id})
}

type chatBody struct {
	SessionID string `json:"session_id"`
	Text      string `json:"text"`
}

// chat runs one turn synchronously and returns the full TurnResult.
// Sentence audio events are also published on the session's SSE stream as
// they complete, so a frontend can start playback before this returns.
func (a *apiServer) chat(w http.ResponseWriter, r *http.Request) {
	var body chatBody
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Text == "" {
		writeJSON(w, 400, map[string]any{"error": "need {session_id, text}"})
		return
	}
	if body.SessionID == "" {
		body.SessionID = "default"
	}
	if a.orch.swapping.Load() {
		writeJSON(w, 503, map[string]any{"error": "vision in progress: the brain is temporarily unloaded for a photo, try again shortly"})
		return
	}
	sess := a.orch.session(body.SessionID)
	res, err := a.orch.RunTurn(r.Context(), sess, body.Text)
	if err != nil {
		writeJSON(w, 502, map[string]any{"error": err.Error(), "partial": res})
		return
	}
	writeJSON(w, 200, res)
}

// events is the SSE stream for one session: turn_started, tone,
// sentence_audio, turn_done, interrupted, error.
func (a *apiServer) events(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("session")
	if id == "" {
		id = "default"
	}
	sess := a.orch.session(id)
	ch := sess.subscribe()
	defer sess.unsubscribe(ch)
	fl, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, 500, map[string]any{"error": "streaming unsupported"})
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(200)
	fl.Flush()
	for {
		select {
		case <-r.Context().Done():
			return
		case ev := <-ch:
			data, err := json.Marshal(ev)
			if err != nil {
				continue
			}
			w.Write([]byte("data: "))
			w.Write(data)
			w.Write([]byte("\n\n"))
			fl.Flush()
		}
	}
}

type interruptBody struct {
	SessionID string `json:"session_id"`
}

func (a *apiServer) interrupt(w http.ResponseWriter, r *http.Request) {
	var body interruptBody
	json.NewDecoder(r.Body).Decode(&body)
	if body.SessionID == "" {
		body.SessionID = "default"
	}
	sess := a.orch.session(body.SessionID)
	// Only cancel: the turn itself emits the single interrupted event as
	// it unwinds. Emitting here as well would double it, and emitting
	// when no turn is active would be a lie.
	if sess.Interrupt() && a.orch.tts != nil {
		a.orch.tts.DropQueued()
	}
	writeJSON(w, 200, map[string]any{"ok": true})
}

type sayBody struct {
	Text string `json:"text"`
	Tag  string `json:"tag,omitempty"` // TTS reference tag (V1 vocabulary); empty = neutral1
}

// say synthesizes one standalone line through the resident tts_host: the
// original Use page's wizard lines and name previews (its /api/tts_say),
// spoken outside any dialogue turn. It shares the single TTS queue with
// turn sentences (FIFO) but writes no history, extracts no memory and
// emits no events.
//
// Mutual exclusion with turns is a hard 409, never a queue: say's only
// callers speak while no turn is running by construction (the wizard is
// a first-run flow, previews are idle-time clicks), and a queued say
// would worst-case wait out a whole turn — minutes on the phone — with
// the page's request hanging open the entire time, for a line the page
// can simply skip. Same rule as /v1/look: a turn in progress wins, the
// latecomer gets an immediate, explicit refusal.
func (a *apiServer) say(w http.ResponseWriter, r *http.Request) {
	var body sayBody
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&body); err != nil || strings.TrimSpace(body.Text) == "" {
		writeJSON(w, 400, map[string]any{"error": "need {text}"})
		return
	}
	if len([]rune(body.Text)) > 500 {
		writeJSON(w, 400, map[string]any{"error": "text too long (max 500 chars)"})
		return
	}
	if a.orch.tts == nil {
		writeJSON(w, 503, map[string]any{"error": "tts is disabled in this scheduler"})
		return
	}
	if a.orch.anyTurnActive() {
		writeJSON(w, 409, map[string]any{"error": "a turn is in progress: say waits for no turn, retry once the reply finishes"})
		return
	}
	tag := body.Tag
	if tag == "" {
		tag = "neutral1"
	}
	res, err := a.orch.tts.Synthesize(r.Context(), ttsRequest{
		Text: body.Text, Tag: tag,
		Temperature: 0.85, TopK: 20, TopP: 0.9, Seed: a.orch.cfg.TTSSeed,
	})
	if err != nil {
		writeJSON(w, 502, map[string]any{"error": err.Error()})
		return
	}
	if res.Error != "" || res.File == "" {
		writeJSON(w, 502, map[string]any{"error": "tts host: " + res.Error})
		return
	}
	writeJSON(w, 200, map[string]any{
		"file": res.File, "duration_s": res.DurationS,
		"elapsed_ms": res.ElapsedMs, "tag": tag, "tag_used": res.TagUsed,
	})
}

type lookBody struct {
	SessionID string `json:"session_id"`
	Question  string `json:"question"`
	ImageB64  string `json:"image_b64"` // base64 of the photo bytes
	Mime      string `json:"mime"`      // default image/png
}

// look runs one photo request: the scheduler unloads brain+embed, loads
// the vision model, answers, and restores the resident set (vision.go).
// It is scheduler-side only in v1; the frontend has no camera channel.
func (a *apiServer) look(w http.ResponseWriter, r *http.Request) {
	var body lookBody
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 24<<20)).Decode(&body); err != nil {
		writeJSON(w, 400, map[string]any{"error": "bad JSON body"})
		return
	}
	if body.Question == "" || body.ImageB64 == "" {
		writeJSON(w, 400, map[string]any{"error": "need {question, image_b64}"})
		return
	}
	mime := body.Mime
	if mime == "" {
		mime = "image/png"
	}
	if !strings.HasPrefix(mime, "image/") {
		writeJSON(w, 400, map[string]any{"error": "mime must be an image type"})
		return
	}
	// Decode once to prove the payload is well-formed base64, then pass
	// the canonical re-encoding on to the vision server.
	raw, err := base64.StdEncoding.DecodeString(body.ImageB64)
	if err != nil {
		writeJSON(w, 400, map[string]any{"error": "image_b64 is not valid base64"})
		return
	}
	if len(raw) == 0 || len(raw) > 16<<20 {
		writeJSON(w, 400, map[string]any{"error": "image size out of range"})
		return
	}
	if body.SessionID == "" {
		body.SessionID = "default"
	}
	sess := a.orch.session(body.SessionID)
	res, status, err := a.orch.Look(r.Context(), sess, body.Question, mime,
		base64.StdEncoding.EncodeToString(raw))
	if err != nil {
		writeJSON(w, status, map[string]any{"error": err.Error(), "partial": res})
		return
	}
	writeJSON(w, 200, res)
}

func (a *apiServer) memoryList(w http.ResponseWriter, r *http.Request) {
	a.orch.mem.mu.Lock()
	pk := append([]Packet(nil), a.orch.mem.packets...)
	a.orch.mem.mu.Unlock()
	type item struct {
		ID   string `json:"id"`
		Text string `json:"text"`
	}
	items := make([]item, 0, len(pk))
	for _, p := range pk {
		items = append(items, item{p.ID, p.Text})
	}
	writeJSON(w, 200, map[string]any{"packets": items, "count": len(items)})
}
