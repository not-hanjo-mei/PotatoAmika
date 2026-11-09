package main

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Event is one frontend-facing stream event (see API.md). The same events
// are delivered over SSE and collected into the blocking /v1/chat response.
type Event struct {
	Type      string       `json:"type"` // turn_started|sentence_audio|tone|turn_done|interrupted|error
	SessionID string       `json:"session_id"`
	Text      string       `json:"text,omitempty"` // user text or sentence text
	Reply     string       `json:"reply,omitempty"`
	Tag       string       `json:"tag,omitempty"` // the one emotion tag: TTS reference + expression
	Sentence  *SentenceOut `json:"sentence,omitempty"`
	Error     string       `json:"error,omitempty"`
}

type SentenceOut struct {
	Index     int     `json:"index"`
	Text      string  `json:"text"`
	Tag       string  `json:"tag"`
	Wav       string  `json:"wav,omitempty"`
	DurationS float64 `json:"duration_s,omitempty"`
	SynthMs   float64 `json:"synth_ms,omitempty"`
	Error     string  `json:"error,omitempty"`
}

type TurnResult struct {
	SessionID  string             `json:"session_id"`
	Reply      string             `json:"reply"`
	Tag        string             `json:"tag"`
	Sentences  []SentenceOut      `json:"sentences"`
	Recall     []string           `json:"recalled_facts,omitempty"`
	FactsAdded int                `json:"facts_added"`
	TimingsMs  map[string]float64 `json:"timings_ms"`
}

// turnState tracks one in-flight turn so /v1/interrupt can cancel it.
// The turn itself owns all of its events (including the closing one);
// Interrupt only cancels and never emits.
type turnState struct {
	cancel context.CancelFunc
}

type Session struct {
	id      string
	mu      sync.Mutex
	history []chatMessage // full record, user/assistant alternating
	subs    map[chan Event]struct{}
	active  *turnState // non-nil while a turn is running
}

func (s *Session) emit(ev Event) {
	ev.SessionID = s.id
	s.mu.Lock()
	subs := make([]chan Event, 0, len(s.subs))
	for ch := range s.subs {
		subs = append(subs, ch)
	}
	s.mu.Unlock()
	for _, ch := range subs {
		select {
		case ch <- ev:
		default: // slow subscriber: drop, never block the pipeline
		}
	}
}

func (s *Session) subscribe() chan Event {
	ch := make(chan Event, 64)
	s.mu.Lock()
	if s.subs == nil {
		s.subs = map[chan Event]struct{}{}
	}
	s.subs[ch] = struct{}{}
	s.mu.Unlock()
	return ch
}

func (s *Session) unsubscribe(ch chan Event) {
	s.mu.Lock()
	delete(s.subs, ch)
	s.mu.Unlock()
}

// Interrupt cancels the session's in-flight turn, if any, and reports
// whether a turn was active. It emits nothing: the turn observes the
// cancellation at its next checkpoint, emits exactly one interrupted
// event as its closing event, and unwinds (see RunTurn).
func (s *Session) Interrupt() bool {
	s.mu.Lock()
	ts := s.active
	s.mu.Unlock()
	if ts == nil {
		return false
	}
	ts.cancel()
	return true
}

// Orchestrator owns the turn pipeline.
type Orchestrator struct {
	cfg      Config
	brain    *Brain
	mem      *MemoryStore
	extract  Extractor
	tts      *TTSClient // may be nil (text-only mode)
	tags     []string
	system   string // static system prompt, built once
	toolsOn  bool
	mcp      MCPClient
	sessMu   sync.Mutex
	sessions map[string]*Session

	// Vision swap state (vision.go): mgr owns the llama-server
	// processes; swapMu serializes photo requests; swapping is set for
	// the whole stop->vision->restore cycle so RunTurn can refuse new
	// turns while the brain is unloaded.
	mgr      *serverManager
	swapMu   sync.Mutex
	swapping atomic.Bool
}

// anyTurnActive reports whether any session has a turn in flight. The
// vision swap consults it before unloading the brain: a photo never
// cuts off a half-spoken reply.
func (o *Orchestrator) anyTurnActive() bool {
	o.sessMu.Lock()
	all := make([]*Session, 0, len(o.sessions))
	for _, s := range o.sessions {
		all = append(all, s)
	}
	o.sessMu.Unlock()
	for _, s := range all {
		s.mu.Lock()
		active := s.active != nil
		s.mu.Unlock()
		if active {
			return true
		}
	}
	return false
}

func (o *Orchestrator) session(id string) *Session {
	o.sessMu.Lock()
	defer o.sessMu.Unlock()
	if o.sessions == nil {
		o.sessions = map[string]*Session{}
	}
	s, ok := o.sessions[id]
	if !ok {
		s = &Session{id: id}
		o.sessions[id] = s
	}
	return s
}

// buildMessages assembles the prompt: static system prompt, recalled memory
// as a second system block, then the recent window trimmed to the context
// budget (oldest turns dropped first; memory and system are pinned).
func (o *Orchestrator) buildMessages(s *Session, recalled []Packet, userText string) []chatMessage {
	msgs := []chatMessage{{Role: "system", Content: o.system}}
	if len(recalled) > 0 {
		var sb strings.Builder
		sb.WriteString("Facts you remember about the user (use them naturally, do not list them back):\n")
		for _, p := range recalled {
			sb.WriteString("- " + p.Text + "\n")
		}
		msgs = append(msgs, chatMessage{Role: "system", Content: sb.String()})
	}
	s.mu.Lock()
	hist := append([]chatMessage(nil), s.history...)
	s.mu.Unlock()
	// Recent window: at most RecentTurns turns (2 msgs each).
	if max := o.cfg.RecentTurns * 2; len(hist) > max {
		hist = hist[len(hist)-max:]
	}
	budget := o.cfg.ContextTokens - o.cfg.GenReserveTokens
	used := estTokens(o.system) + estTokens(userText)
	for _, m := range msgs[1:] {
		used += estTokens(m.Content)
	}
	for len(hist) > 0 {
		total := used
		for _, m := range hist {
			total += estTokens(m.Content)
		}
		if total <= budget {
			break
		}
		if len(hist) >= 2 {
			hist = hist[2:]
		} else {
			hist = hist[1:]
		}
	}
	msgs = append(msgs, hist...)
	msgs = append(msgs, chatMessage{Role: "user", Content: userText})
	return msgs
}

// sentenceSplitter cuts streamed text into speakable sentences.
// It works on runes so multibyte punctuation is handled correctly.
type sentenceSplitter struct {
	buf []rune
}

func isSentenceEnd(rs []rune, i int) bool {
	c := rs[i]
	if c != '.' && c != '!' && c != '?' && c != '\n' {
		return false
	}
	if c == '.' {
		// decimal point or ellipsis: not a boundary
		if i+1 < len(rs) && rs[i+1] >= '0' && rs[i+1] <= '9' {
			return false
		}
		if i+1 < len(rs) && rs[i+1] == '.' {
			return false
		}
	}
	if i+1 >= len(rs) {
		return true
	}
	n := rs[i+1]
	return n == ' ' || n == '\n' || n == '"' || n == '\u201d'
}

func (sp *sentenceSplitter) Feed(s string) []string {
	sp.buf = append(sp.buf, []rune(s)...)
	var out []string
	start := 0
	for i := 0; i < len(sp.buf); i++ {
		if isSentenceEnd(sp.buf, i) {
			end := i + 1
			sent := strings.TrimSpace(string(sp.buf[start:end]))
			if sent != "" {
				out = append(out, sent)
			}
			start = end
		}
	}
	sp.buf = append([]rune(nil), sp.buf[start:]...)
	return out
}

func (sp *sentenceSplitter) Flush() []string {
	s := strings.TrimSpace(string(sp.buf))
	sp.buf = nil
	if s == "" {
		return nil
	}
	return []string{s}
}

// releasableText computes how much of the raw stream may be spoken already,
// given that Minico's leaked narration ends with a literal </think>: nothing
// is releasable until a think marker resolves, except in long marker-free
// output where the leak pattern would already have shown itself. At stream
// end the caller uses stripThink on the whole buffer instead.
func releasableText(raw string) string {
	if strings.Contains(raw, "</think>") || strings.Contains(raw, "<think>") {
		return stripThink(raw)
	}
	if len(raw) > 400 {
		return raw
	}
	return ""
}

// RunTurn executes one full turn: recall, generate (streaming), classify
// tone once, stream sentences into TTS as they complete, then run memory
// extraction synchronously. Events are emitted throughout.
func (o *Orchestrator) RunTurn(ctx context.Context, sess *Session, userText string) (*TurnResult, error) {
	if o.swapping.Load() {
		return &TurnResult{SessionID: sess.id, TimingsMs: map[string]float64{}},
			fmt.Errorf("vision in progress: the brain is temporarily unloaded for a photo, try again shortly")
	}
	ctx, cancel := context.WithCancel(ctx)
	ts := &turnState{cancel: cancel}
	sess.mu.Lock()
	sess.active = ts
	sess.mu.Unlock()
	defer func() {
		cancel()
		sess.mu.Lock()
		if sess.active == ts {
			sess.active = nil
		}
		sess.mu.Unlock()
	}()

	t0 := time.Now()
	timings := map[string]float64{}
	res := &TurnResult{SessionID: sess.id, TimingsMs: timings}
	// unwind is the single interrupted-exit path: it is the only place an
	// interrupted event is emitted, exactly once, as the turn's closing
	// event (never turn_done). An interrupted turn writes no history and
	// runs no memory extraction, so a half reply can never become a fact
	// source. Every pipeline phase checks ctx at its boundary and leaves
	// through here.
	unwind := func() (*TurnResult, error) {
		timings["total"] = msSince(t0)
		sess.emit(Event{Type: "interrupted", Text: userText})
		return res, ctx.Err()
	}
	sess.emit(Event{Type: "turn_started", Text: userText})

	// Tool stage: skipped entirely unless an MCP config exists (v1: never,
	// and the dormant client returns no tools regardless).
	if o.toolsOn {
		_, _ = o.mcp.Tools(ctx)
	}

	// Recall.
	var recalled []Packet
	if o.mem != nil {
		tR := time.Now()
		pk, err := o.mem.Recall(ctx, userText, o.cfg.MemoryRecallN, o.cfg.MemoryBudgetChars)
		if err == nil {
			recalled = pk
			for _, p := range pk {
				res.Recall = append(res.Recall, p.Text)
			}
		}
		timings["recall"] = msSince(tR)
	}

	msgs := o.buildMessages(sess, recalled, userText)

	// Generation + sentence pipeline.
	tG := time.Now()
	var rawAll strings.Builder
	var released strings.Builder // think-stripped text already fed to splitter
	splitter := &sentenceSplitter{}
	tag := ""
	var tagErr error
	var pending []chan ttsOutcome
	var sentences []SentenceOut
	idx := 0

	// Sentence cap (pipeline layer of the reply length control): once
	// ReplyMaxSentences sentences have been released, generation is
	// aborted via genCtx and the turn closes on exactly those sentences.
	// This is not an interrupt: the turn ctx stays live, the turn writes
	// history and extracts memory as usual, and res.Reply is rebuilt from
	// the released sentences so text, audio and history all agree.
	maxSent := o.cfg.ReplyMaxSentences // <= 0: pipeline cap disabled
	capped := false
	genCtx, genCancel := context.WithCancel(ctx)
	defer genCancel()

	handleSentences := func(list []string) {
		for _, sent := range list {
			if ctx.Err() != nil {
				return // interrupted mid-generation: enqueue nothing more
			}
			if maxSent > 0 && idx >= maxSent {
				// Cap hit: drop this and every later sentence (never a
				// half sentence — the splitter only yields complete
				// ones) and stop the model from generating more.
				capped = true
				genCancel()
				return
			}
			if tag == "" {
				tT := time.Now()
				tg, err := o.brain.ClassifyTone(ctx, userText, sent, o.tags)
				if err != nil {
					tagErr = err
					tg = "neutral1"
				}
				tag = tg
				timings["tone_classify"] = msSince(tT)
				sess.emit(Event{Type: "tone", Tag: tag})
			}
			sentences = append(sentences, SentenceOut{Index: idx, Text: sent, Tag: tag})
			if o.tts != nil {
				pending = append(pending, o.tts.SynthesizeAsync(ctx, ttsRequest{
					Text: sent, Tag: tag,
					Temperature: 0.85, TopK: 20, TopP: 0.9, Seed: o.cfg.TTSSeed,
				}))
			}
			idx++
		}
	}

	reply, genErr := o.brain.ReplyStream(genCtx, msgs, func(delta string) {
		rawAll.WriteString(delta)
		rel := releasableText(rawAll.String())
		if len(rel) > released.Len() {
			newPart := rel[released.Len():]
			released.WriteString(newPart)
			handleSentences(splitter.Feed(newPart))
		}
	})
	timings["generate"] = msSince(tG)
	if genErr != nil {
		if ctx.Err() != nil {
			return unwind()
		}
		if !capped {
			sess.emit(Event{Type: "error", Error: genErr.Error()})
			return res, genErr
		}
		// capped: the genCtx abort surfaces here as a stream error;
		// it is the intended end of generation, not a failure.
	}
	// Release whatever the streaming guard held back.
	if reply == "" {
		reply = stripThink(rawAll.String())
	}
	if len(reply) > released.Len() {
		handleSentences(splitter.Feed(reply[released.Len():]))
	}
	handleSentences(splitter.Flush())
	if capped {
		// The spoken reply is exactly the released sentences; anything
		// the model produced past the cap (or was about to) is dropped,
		// so the reply text, the audio and the stored history all carry
		// the same truncated-but-complete reply.
		parts := make([]string, len(sentences))
		for i, s := range sentences {
			parts[i] = s.Text
		}
		reply = strings.Join(parts, " ")
	}
	if tag == "" { // no sentence ever triggered classification (empty reply)
		tag = "neutral1"
		if reply != "" {
			if tg, err := o.brain.ClassifyTone(ctx, userText, reply, o.tags); err == nil {
				tag = tg
				sess.emit(Event{Type: "tone", Tag: tag})
			}
		}
	}
	_ = tagErr
	res.Reply = reply
	res.Tag = tag

	// Interrupt that landed while generation was finishing (or during
	// the trailing tone classification): stop before any audio drains.
	if ctx.Err() != nil {
		return unwind()
	}

	// Drain TTS results in sentence order, emitting audio events. The
	// wait aborts on cancellation: queued jobs die in the worker via
	// their own ctx, and the one synthesis already inside the host
	// finishes there but its result lands in a buffered channel nobody
	// reads, so it is never emitted.
	tW := time.Now()
	drained := 0
	for i, ch := range pending {
		var out ttsOutcome
		select {
		case out = <-ch:
		case <-ctx.Done():
		}
		if ctx.Err() != nil {
			timings["tts_drain"] = msSince(tW)
			res.Sentences = sentences[:drained]
			return unwind()
		}
		if out.err != nil {
			sentences[i].Error = out.err.Error()
		} else {
			sentences[i].Wav = out.res.File
			sentences[i].DurationS = out.res.DurationS
			sentences[i].SynthMs = out.res.ElapsedMs
			if out.res.TagUsed != "" {
				sentences[i].Tag = out.res.TagUsed
			}
		}
		drained++
		sess.emit(Event{Type: "sentence_audio", Sentence: &SentenceOut{
			Index: sentences[i].Index, Text: sentences[i].Text, Tag: sentences[i].Tag,
			Wav: sentences[i].Wav, DurationS: sentences[i].DurationS, SynthMs: sentences[i].SynthMs,
			Error: sentences[i].Error,
		}})
	}
	timings["tts_drain"] = msSince(tW)
	res.Sentences = sentences

	// Memory extraction: synchronous in v1 (idle worker is a seam, see
	// memory.go Extractor). Runs after audio so it never delays speech.
	// History is only committed after extraction, so an interrupt that
	// lands during extraction still leaves no trace of this turn.
	sess.mu.Lock()
	convo := append([]chatMessage(nil), sess.history...)
	sess.mu.Unlock()
	convo = append(convo,
		chatMessage{Role: "user", Content: userText},
		chatMessage{Role: "assistant", Content: reply})
	if o.extract != nil {
		tE := time.Now()
		window := convo
		if len(window) > 10 {
			window = window[len(window)-10:]
		}
		added, err := o.extract.Extract(ctx, window)
		if err == nil {
			res.FactsAdded = added
		} else if ctx.Err() == nil {
			log.Printf("turn: memory extraction failed: %v", err)
		}
		timings["extract"] = msSince(tE)
	}
	if ctx.Err() != nil {
		return unwind()
	}

	// History commit: only a turn that ran to completion gets recorded.
	sess.mu.Lock()
	sess.history = append(sess.history,
		chatMessage{Role: "user", Content: userText},
		chatMessage{Role: "assistant", Content: reply})
	sess.mu.Unlock()

	timings["total"] = msSince(t0)
	sess.emit(Event{Type: "turn_done", Reply: reply, Tag: tag})
	return res, nil
}

func msSince(t time.Time) float64 {
	return float64(time.Since(t).Microseconds()) / 1000.0
}
