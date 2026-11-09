package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

// Vision flow (checklist item 6, BUDGET.md 眼睛一节): the vision model's
// ~685 MB peak does not fit next to the resident set (~290-330 MB free on
// the phone), so a photo is a swap, not a side load:
//
//	stop brain+embed -> load vision -> answer -> unload vision -> reload brain+embed
//
// History and the memory store live in the scheduler, not in the
// llama-server processes, so the swap loses no conversation state; the
// reload uses exactly the startup arguments (12k context, full persona
// is scheduler-side anyway).
//
// Safety rules:
//   - A turn in flight refuses the photo (a half-spoken reply is never
//     cut off by an unload).
//   - While a swap runs, new turns are refused up front (RunTurn guard).
//   - Every failure after the stop step still restores brain+embed
//     before the error is returned; the restore runs on a fresh context
//     so a cancelled or timed-out photo request cannot strand the
//     machine without a brain.

type LookResult struct {
	Answer    string             `json:"answer"`
	TimingsMs map[string]float64 `json:"timings_ms"`
}

// describe sends one image + question to the vision llama-server using
// the OpenAI-style multimodal content parts llama.cpp accepts.
func (m *serverManager) describe(ctx context.Context, question, mime, imgB64 string) (string, error) {
	type imageURL struct {
		URL string `json:"url"`
	}
	type part struct {
		Type     string    `json:"type"`
		Text     string    `json:"text,omitempty"`
		ImageURL *imageURL `json:"image_url,omitempty"`
	}
	type vmsg struct {
		Role    string `json:"role"`
		Content []part `json:"content"`
	}
	body := struct {
		Messages    []vmsg  `json:"messages"`
		MaxTokens   int     `json:"max_tokens"`
		Temperature float64 `json:"temperature"`
		Stream      bool    `json:"stream"`
	}{
		Messages: []vmsg{{Role: "user", Content: []part{
			{Type: "image_url", ImageURL: &imageURL{URL: "data:" + mime + ";base64," + imgB64}},
			{Type: "text", Text: question},
		}}},
		MaxTokens:   m.cfg.VisionMaxTokens,
		Temperature: 0.2,
		Stream:      false,
	}
	resp, err := m.vision.post(ctx, "/v1/chat/completions", body)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return "", fmt.Errorf("vision: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
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
		return "", fmt.Errorf("vision: no choices")
	}
	return stripThink(out.Choices[0].Message.Content), nil
}

// restoreAfterVision unloads vision (if up) and brings brain+embed back
// with the startup arguments. It runs on a fresh background context with
// its own budget: rollback must not be cancellable by the request that
// failed. Step timings land in res.
func (o *Orchestrator) restoreAfterVision(res *LookResult) error {
	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Minute)
	defer cancel()
	t := time.Now()
	o.mgr.stopVision()
	res.TimingsMs["vision_unload"] = msSince(t)
	t = time.Now()
	err := o.mgr.startBrainEmbed(ctx)
	res.TimingsMs["reload_servers"] = msSince(t)
	return err
}

// Look runs one photo request end to end. The returned status is the
// HTTP status the API layer should use: 200 on success, 409 when a turn
// or another photo is in progress, 503 when vision is not configured,
// 502 when the swap or the model failed (brain+embed already restored),
// 500 only when the restore itself failed.
func (o *Orchestrator) Look(ctx context.Context, sess *Session, question, mime, imgB64 string) (*LookResult, int, error) {
	if o.mgr == nil || !o.mgr.visionConfigured() {
		return nil, 503, fmt.Errorf("vision not configured (needs vision_model, vision_mmproj and spawned brain+embed)")
	}
	if !o.swapMu.TryLock() {
		return nil, 409, fmt.Errorf("another photo request is already in progress")
	}
	defer o.swapMu.Unlock()
	if o.anyTurnActive() {
		return nil, 409, fmt.Errorf("a turn is in progress; let it finish before taking a photo")
	}
	o.swapping.Store(true)
	defer o.swapping.Store(false)

	res := &LookResult{TimingsMs: map[string]float64{}}
	t0 := time.Now()

	t := time.Now()
	o.mgr.stopBrainEmbed()
	res.TimingsMs["stop_servers"] = msSince(t)

	t = time.Now()
	if err := o.mgr.startVision(ctx); err != nil {
		rerr := o.restoreAfterVision(res)
		if rerr != nil {
			return res, 500, fmt.Errorf("vision load failed: %v; restore of brain+embed also failed: %v", err, rerr)
		}
		return res, 502, fmt.Errorf("vision load failed: %v (brain+embed restored)", err)
	}
	res.TimingsMs["vision_load"] = msSince(t)

	askCtx, cancel := context.WithTimeout(ctx, time.Duration(o.cfg.VisionTimeoutS)*time.Second)
	t = time.Now()
	answer, aerr := o.mgr.describe(askCtx, question, mime, imgB64)
	cancel()
	res.TimingsMs["vision_answer"] = msSince(t)

	if rerr := o.restoreAfterVision(res); rerr != nil {
		return res, 500, fmt.Errorf("restore of brain+embed failed: %v (vision answer was: %q)", rerr, answer)
	}
	if aerr != nil {
		return res, 502, fmt.Errorf("vision answer failed: %v (brain+embed restored)", aerr)
	}

	res.Answer = answer
	res.TimingsMs["total"] = msSince(t0)

	// The photo round joins the session history, so follow-up turns can
	// refer to what was shown. Memory extraction is not run for it: the
	// answer is the vision model's description, not user testimony.
	if sess != nil {
		sess.mu.Lock()
		sess.history = append(sess.history,
			chatMessage{Role: "user", Content: "(showed a photo) " + question},
			chatMessage{Role: "assistant", Content: answer})
		sess.mu.Unlock()
	}
	return res, 200, nil
}
