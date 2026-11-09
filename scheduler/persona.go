package main

import (
	"encoding/json"
	"os"
	"strings"
	"time"
	"unicode"
)

// estTokens is the offline token estimator used for budget trimming before
// the server is up: CJK runes count ~1 token each, other text ~4 chars per
// token. Exact counts (via /tokenize) are logged at startup for the record.
func estTokens(s string) int {
	n := 0
	ascii := 0
	for _, r := range s {
		if unicode.Is(unicode.Han, r) || unicode.Is(unicode.Hiragana, r) ||
			unicode.Is(unicode.Katakana, r) || unicode.Is(unicode.Hangul, r) {
			n++
		} else {
			ascii++
		}
	}
	return n + (ascii+3)/4
}

// trimToTokenBudget cuts s at a paragraph boundary so its estimate stays
// within budget. Wording is never altered; this is a load-time length cap,
// used only when cfg.PersonaTrim is on (fallback for small windows — at the
// 12k default the full persona loads untrimmed, see SCHEDULER.md).
func trimToTokenBudget(s string, budget int) string {
	if estTokens(s) <= budget {
		return s
	}
	paras := strings.Split(s, "\n\n")
	var kept []string
	total := 0
	for _, p := range paras {
		t := estTokens(p)
		if len(kept) > 0 && total+t > budget {
			break
		}
		kept = append(kept, p)
		total += t
		if total > budget {
			break
		}
	}
	return strings.Join(kept, "\n\n")
}

// loadPersona reads the surface persona file verbatim and substitutes the
// two load-time placeholders, exactly like the original loader.
func loadPersona(cfg Config) (string, error) {
	b, err := os.ReadFile(cfg.PersonaFile)
	if err != nil {
		return "", err
	}
	s := strings.TrimPrefix(string(b), "\ufeff")
	s = strings.ReplaceAll(s, "{user_name}", cfg.UserName)
	city := cfg.City
	if city == "" {
		city = "the city"
	}
	s = strings.ReplaceAll(s, "{city}", city)
	return s, nil
}

// sceneSnippet returns the surface-level scene context from scenes.json:
// the single surface.context string (the surface section has no
// daily_context key — that lives under deep and feeds the examples file's
// placeholders, not the runtime). Only {user_name}/{city} placeholders
// occur in it. Returns "" when anything about the file shape is unexpected.
func sceneSnippet(cfg Config, now time.Time) string {
	if cfg.ScenesFile == "" {
		return ""
	}
	b, err := os.ReadFile(cfg.ScenesFile)
	if err != nil {
		return ""
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(b, &root); err != nil {
		return ""
	}
	var surface struct {
		Context string `json:"context"`
	}
	if raw, ok := root["surface"]; ok {
		if err := json.Unmarshal(raw, &surface); err != nil {
			return ""
		}
	}
	s := strings.TrimSpace(surface.Context)
	if s == "" {
		return ""
	}
	city := cfg.City
	if city == "" {
		city = "the city"
	}
	s = strings.ReplaceAll(s, "{user_name}", cfg.UserName)
	s = strings.ReplaceAll(s, "{city}", city)
	return s
}

// buildSystemPrompt assembles the static system prompt once at startup:
// the persona (full, unless cfg.PersonaTrim caps it to SystemBudgetTokens
// at a paragraph boundary) + a short runtime frame (language, scene, date).
// The persona wording itself is not edited; the frame is scheduler-owned text.
func buildSystemPrompt(cfg Config, persona string, now time.Time) string {
	var sb strings.Builder
	if cfg.PersonaTrim {
		persona = trimToTokenBudget(persona, cfg.SystemBudgetTokens)
	}
	sb.WriteString(persona)
	sb.WriteString("\n\n[Runtime]\n")
	if cfg.ReplyBrevity {
		n := "a few"
		if cfg.ReplyMaxSentences > 0 {
			n = "1 to " + itoa(cfg.ReplyMaxSentences)
		}
		sb.WriteString("Speak English only. Keep every reply to " + n + " short sentences, then stop as soon as you have answered. " +
			"Do not restate your background, your memories, or what was said earlier in the conversation unless the user asks about them directly.\n")
	} else {
		sb.WriteString("Speak English only. Keep replies short and conversational, a few sentences at most.\n")
	}
	sb.WriteString("Today is " + now.Format("Monday, 2006-01-02") + ".\n")
	if snip := sceneSnippet(cfg, now); snip != "" {
		sb.WriteString("Your day so far: " + snip + "\n")
	}
	return sb.String()
}
