package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Brain is a client for a llama.cpp llama-server running Minico-M1-1105.
// It also talks to a second llama-server instance for embeddings (Jina).
type Brain struct {
	base  string // chat/completions server
	embed string // embedding server (may be "")
	http  *http.Client
	cfg   Config
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func newBrain(cfg Config) *Brain {
	return &Brain{
		base:  strings.TrimRight(cfg.BrainURL, "/"),
		embed: strings.TrimRight(cfg.EmbedURL, "/"),
		http:  &http.Client{Timeout: 0}, // streaming; deadlines via ctx
		cfg:   cfg,
	}
}

func (b *Brain) healthy(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, "GET", b.base+"/health", nil)
	if err != nil {
		return false
	}
	resp, err := b.http.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

// stripThink removes Minico's leaked reasoning. With effort=off the model
// still emits first-person narration followed by a literal </think>, or a
// full <think>...</think> block. Rules (from quality-screen SUMMARY anomaly 1):
//  1. Drop complete <think>...</think> spans.
//  2. If a stray </think> remains, everything up to and including the last
//     one is narration; keep only what follows.
//  3. Trim leading blank lines (empty think-block residue).
func stripThink(s string) string {
	for {
		i := strings.Index(s, "<think>")
		if i < 0 {
			break
		}
		j := strings.Index(s[i:], "</think>")
		if j < 0 {
			// Unterminated opener: the rest is reasoning.
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

type chatRequest struct {
	Messages           []chatMessage   `json:"messages"`
	MaxTokens          int             `json:"max_tokens"`
	Temperature        float64         `json:"temperature"`
	TopK               int             `json:"top_k,omitempty"`
	RepeatPenalty      float64         `json:"repeat_penalty,omitempty"`
	Stream             bool            `json:"stream"`
	ChatTemplateKwargs map[string]any  `json:"chat_template_kwargs,omitempty"`
	ResponseFormat     json.RawMessage `json:"response_format,omitempty"`
}

func (b *Brain) post(ctx context.Context, path string, body any) (*http.Response, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", b.base+path, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return b.http.Do(req)
}

// ReplyStream generates the surface reply with effort=off and no grammar
// (checklist item 4: grammar on the reply degrades semantics; effort=off plus
// any grammar makes llama-server return 400, so constrained calls use "low").
// onText receives raw streamed deltas; the returned string is think-stripped.
func (b *Brain) ReplyStream(ctx context.Context, msgs []chatMessage, onText func(string)) (string, error) {
	body := chatRequest{
		Messages:           msgs,
		MaxTokens:          b.cfg.ReplyMaxTokens,
		Temperature:        b.cfg.ReplyTemperature,
		TopK:               b.cfg.ReplyTopK,
		RepeatPenalty:      b.cfg.ReplyRepeatPen,
		Stream:             true,
		ChatTemplateKwargs: map[string]any{"reasoning_effort": "off"},
	}
	resp, err := b.post(ctx, "/v1/chat/completions", body)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return "", fmt.Errorf("brain reply: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	var raw strings.Builder
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimSpace(line[len("data: "):])
		if payload == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			raw.WriteString(chunk.Choices[0].Delta.Content)
			if onText != nil {
				onText(chunk.Choices[0].Delta.Content)
			}
		}
	}
	if err := sc.Err(); err != nil {
		return stripThink(raw.String()), err
	}
	return stripThink(raw.String()), nil
}

// chatOnce is a non-streaming completion used for the small side calls
// (tone classification, memory extraction).
func (b *Brain) chatOnce(ctx context.Context, msgs []chatMessage, maxTokens int, temp float64, effort string, responseFormat json.RawMessage) (string, error) {
	body := chatRequest{
		Messages:           msgs,
		MaxTokens:          maxTokens,
		Temperature:        temp,
		TopK:               b.cfg.ReplyTopK,
		RepeatPenalty:      b.cfg.ReplyRepeatPen,
		Stream:             false,
		ChatTemplateKwargs: map[string]any{"reasoning_effort": effort},
		ResponseFormat:     responseFormat,
	}
	resp, err := b.post(ctx, "/v1/chat/completions", body)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return "", fmt.Errorf("brain chat: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
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
		return "", fmt.Errorf("brain chat: no choices")
	}
	return stripThink(out.Choices[0].Message.Content), nil
}

// ClassifyTone decides the single emotion tag for a turn. The tag is an enum
// member of the TTS reference tag set: the same value selects the TTS
// reference and drives the frontend (Live2D) expression (checklist item 17).
// This is an extractive/classification call, so a json_schema enum constraint
// is allowed here (checklist items 4 and 12); because a grammar is attached,
// effort is "low", not "off" (off + grammar = HTTP 400 on llama-server).
//
// The token budget must cover the effort=low reasoning, not just the JSON:
// llama-server puts the reasoning in reasoning_content and only emits the
// {"tag": ...} object into content once reasoning finishes. Measured usage
// is 45-93 completion tokens (reasoning plus ~7 of JSON). The original
// budget of 24 was exhausted by reasoning on every call (finish_reason
// "length", empty content), the parse failed, and the session silently fell
// back to neutral1 — every turn classified as neutral regardless of input.
// 160 leaves headroom over the worst observed case.
func (b *Brain) ClassifyTone(ctx context.Context, userText, reply string, tags []string) (string, error) {
	enumJSON, _ := json.Marshal(tags)
	schema := json.RawMessage(fmt.Sprintf(
		`{"type":"json_schema","json_schema":{"name":"tone","schema":{"type":"object","properties":{"tag":{"type":"string","enum":%s}},"required":["tag"],"additionalProperties":false},"strict":true}}`,
		enumJSON))
	prompt := "Pick the single emotion tag that best matches how the assistant's reply sounds. " +
		"Reply with only the JSON object.\n\nUser said: " + userText + "\nAssistant replied: " + reply
	out, err := b.chatOnce(ctx, []chatMessage{{Role: "user", Content: prompt}}, 160, 0, "low", schema)
	if err != nil {
		return "", err
	}
	var obj struct {
		Tag string `json:"tag"`
	}
	if err := json.Unmarshal([]byte(extractJSONObject(out)), &obj); err != nil {
		return "", fmt.Errorf("tone parse: %q: %w", out, err)
	}
	for _, t := range tags {
		if obj.Tag == t {
			return obj.Tag, nil
		}
	}
	return "", fmt.Errorf("tone %q not in enum", obj.Tag)
}

// ExtractFacts asks the brain (as background worker) for durable facts about
// the user. Minico answers with a categorized JSON *object*, not a list
// (quality-screen battery D), so parsing accepts both shapes.
func (b *Brain) ExtractFacts(ctx context.Context, convo []chatMessage) ([]string, error) {
	lang := "in English"
	if strings.EqualFold(b.cfg.MemoryLanguage, "zh") {
		lang = "in Chinese"
	}
	var sb strings.Builder
	sb.WriteString("Extract durable facts about the user from this conversation as a JSON object grouped by category, " +
		"each value a list of short statements " + lang + ". Include preferences, personal details, plans and events. " +
		"If there are no durable facts, output {}. Output only the JSON object.\n\n")
	for _, m := range convo {
		sb.WriteString(m.Role + ": " + m.Content + "\n")
	}
	out, err := b.chatOnce(ctx, []chatMessage{{Role: "user", Content: sb.String()}}, 300, 0.1, "low", nil)
	if err != nil {
		return nil, err
	}
	return parseFacts(out), nil
}

// parseFacts flattens either a categorized object {"cat": ["fact", ...]}
// or a plain list ["fact", ...] into a list of fact strings. Code fences
// and surrounding prose are tolerated.
func parseFacts(out string) []string {
	s := strings.TrimSpace(out)
	s = strings.TrimPrefix(s, "```json")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(s, "```")
	s = strings.TrimSpace(s)
	if i := strings.IndexAny(s, "{["); i > 0 {
		s = s[i:]
	}
	var facts []string
	var list []string
	if err := json.Unmarshal([]byte(s), &list); err == nil {
		for _, f := range list {
			if f = strings.TrimSpace(f); f != "" {
				facts = append(facts, f)
			}
		}
		return facts
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(s), &obj); err == nil {
		for _, raw := range obj {
			var arr []string
			if err := json.Unmarshal(raw, &arr); err == nil {
				for _, f := range arr {
					if f = strings.TrimSpace(f); f != "" {
						facts = append(facts, f)
					}
				}
				continue
			}
			var single string
			if err := json.Unmarshal(raw, &single); err == nil && strings.TrimSpace(single) != "" {
				facts = append(facts, strings.TrimSpace(single))
			}
		}
	}
	return facts
}

func extractJSONObject(s string) string {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "```json")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(s, "```")
	i := strings.Index(s, "{")
	j := strings.LastIndex(s, "}")
	if i >= 0 && j > i {
		return s[i : j+1]
	}
	return s
}

// Embed returns the embedding vector for one text from the Jina server.
func (b *Brain) Embed(ctx context.Context, text string) ([]float32, error) {
	if b.embed == "" {
		return nil, fmt.Errorf("embedding server not configured")
	}
	body := map[string]any{"input": text}
	data, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, "POST", b.embed+"/embedding", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := b.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("embed: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	// llama-server's /embedding wraps the pooled vector in an extra array:
	// [{"embedding": [[...768 floats...]], "index": 0}].
	var out []struct {
		Embedding [][]float32 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out) == 0 || len(out[0].Embedding) == 0 || len(out[0].Embedding[0]) == 0 {
		return nil, fmt.Errorf("embed: empty result")
	}
	return out[0].Embedding[0], nil
}

// Tokenize counts tokens exactly using the brain server's tokenizer.
func (b *Brain) Tokenize(ctx context.Context, text string) (int, error) {
	body := map[string]any{"content": text}
	data, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, "POST", b.base+"/tokenize", bytes.NewReader(data))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	c, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	req = req.WithContext(c)
	resp, err := b.http.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	var out struct {
		Tokens []int `json:"tokens"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return 0, err
	}
	return len(out.Tokens), nil
}
