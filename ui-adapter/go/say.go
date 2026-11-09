package main

import (
	"encoding/base64"
	"fmt"
	"os"
	"sync"
)

/* ------------------------------------------------------------------ *
 * tts_say: forwarded to the scheduler's POST /v1/say, which speaks
 * through the one resident tts_host. No private host here (it cost
 * ~375 MB on the preview; the phone has no room for a second one).
 * The scheduler refuses with 409 while a dialogue turn is running;
 * that surfaces to the page as a skipped line, never a hang.
 * Concurrent say calls are serialized through a local chain so the
 * page's back-to-back wizard lines keep their order.
 * ------------------------------------------------------------------ */

type sayJob struct {
	text, tag string
	res       chan sayJobResult
}

type sayJobResult struct {
	body rawObj
	err  error
}

var sayQueue = make(chan sayJob, 128)

func init() {
	go func() {
		for j := range sayQueue {
			j.res <- sayOnceSync(j.text, j.tag)
		}
	}()
}

type sayError struct {
	msg    string
	status int
}

func (e *sayError) Error() string { return e.msg }

func sayOnceSync(text, tag string) sayJobResult {
	r, err := schedJSON("POST", "/v1/say", orderedJSON{{"text", text}, {"tag", tag}})
	if err != nil {
		return sayJobResult{err: err}
	}
	if r.status == 200 && r.body != nil {
		if f, ok := rawStrField(r.body, "file"); ok && f != "" {
			return sayJobResult{body: r.body}
		}
	}
	msg := fmt.Sprintf("scheduler /v1/say returned %d", r.status)
	if r.body != nil {
		if e, ok := rawStrField(r.body, "error"); ok && e != "" {
			msg = e
		}
	}
	return sayJobResult{err: &sayError{msg: msg, status: r.status}}
}

func sayOnce(text, tag string) (rawObj, error) {
	res := make(chan sayJobResult, 1)
	sayQueue <- sayJob{text: text, tag: tag, res: res}
	out := <-res
	return out.body, out.err
}

/* Say results are cached by (tag, text): wizard lines are few, fixed and
 * re-requested by design ("hear it again", a step re-render, the wizard's
 * pointerdown+click double-fire for a parked line). A repeat then answers
 * in milliseconds instead of paying a full synthesis. In-flight calls for
 * the same line join one result, so a double-fire synthesizes once.
 * Errors (including the scheduler's 409 while a turn runs) are never
 * cached — a retry after the turn ends must be free to succeed. */

type sayPayload struct {
	Audio       string  `json:"audio"`
	AudioFormat string  `json:"audio_format"`
	Duration    float64 `json:"duration"`
	Spoken      string  `json:"spoken"`
	Emotion     string  `json:"emotion"`
	VoiceTag    string  `json:"voice_tag"`
	Language    string  `json:"language"`
}

const sayCacheMax = 16

var (
	sayMu       sync.Mutex
	sayCache    = map[string]*sayPayload{}
	sayOrder    []string
	sayInflight = map[string]*sayCall{}
)

type sayCall struct {
	done    chan struct{}
	payload *sayPayload
	err     error
}

func sayCached(text, tag, emotion string) (*sayPayload, error) {
	key := tag + "\n" + text
	sayMu.Lock()
	if hit, ok := sayCache[key]; ok {
		sayMu.Unlock()
		return hit, nil
	}
	if call, ok := sayInflight[key]; ok {
		sayMu.Unlock()
		<-call.done
		return call.payload, call.err
	}
	call := &sayCall{done: make(chan struct{})}
	sayInflight[key] = call
	sayMu.Unlock()

	go func() {
		var payload *sayPayload
		var err error
		defer func() {
			sayMu.Lock()
			delete(sayInflight, key)
			if err == nil && payload != nil {
				sayCache[key] = payload
				sayOrder = append(sayOrder, key)
				for len(sayCache) > sayCacheMax && len(sayOrder) > 0 {
					oldest := sayOrder[0]
					sayOrder = sayOrder[1:]
					delete(sayCache, oldest)
				}
			}
			sayMu.Unlock()
			call.payload = payload
			call.err = err
			close(call.done)
		}()
		r, e := sayOnce(text, tag)
		if e != nil {
			err = e
			return
		}
		file, _ := rawStrField(r, "file")
		audioRaw, e2 := os.ReadFile(file)
		if e2 != nil {
			err = e2
			return
		}
		tagUsed := tag
		if tu, ok := rawStrField(r, "tag_used"); ok && tu != "" {
			tagUsed = tu
		}
		dur, _ := rawNumField(r, "duration_s")
		emo := emotion
		if emo == "" {
			emo = v1toV2(tagUsed)
		}
		payload = &sayPayload{
			Audio:       base64.StdEncoding.EncodeToString(audioRaw),
			AudioFormat: "wav",
			Duration:    dur,
			Spoken:      text,
			Emotion:     emo,
			VoiceTag:    tagUsed,
			Language:    "en",
		}
	}()

	<-call.done
	return call.payload, call.err
}
