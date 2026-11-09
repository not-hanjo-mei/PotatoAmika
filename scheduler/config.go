package main

import (
	"encoding/json"
	"os"
)

// Config is the single source of runtime knobs. Everything the phone build
// needs to change (paths, ports, engine) lives here or in the JSON file that
// overrides these defaults; nothing is compiled in per-device.
type Config struct {
	Listen string `json:"listen"` // HTTP listen address for the frontend API

	// Brain (Minico via llama-server).
	BrainURL       string `json:"brain_url"`        // attach mode: existing llama-server
	BrainSpawn     bool   `json:"brain_spawn"`      // spawn llama-server ourselves
	LlamaServerBin string `json:"llama_server_bin"` // path to llama-server binary
	BrainModel     string `json:"brain_model"`      // GGUF path
	BrainPort      int    `json:"brain_port"`
	BrainCtx       int    `json:"brain_ctx"`     // llama-server -c
	BrainThreads   int    `json:"brain_threads"` // llama-server -t

	// Embeddings (Jina v5 nano via a second llama-server, --embedding).
	EmbedURL     string `json:"embed_url"`
	EmbedSpawn   bool   `json:"embed_spawn"`
	EmbedModel   string `json:"embed_model"`
	EmbedPort    int    `json:"embed_port"`
	EmbedThreads int    `json:"embed_threads"`

	// Vision (LFM2.5-VL via a third llama-server, loaded on demand only:
	// its ~685 MB peak does not fit next to the resident set, so a photo
	// request stops brain+embed, runs vision, then restores them; see
	// vision.go and BUDGET.md). Empty model paths = vision disabled.
	VisionModel     string `json:"vision_model"`      // GGUF path
	VisionMmproj    string `json:"vision_mmproj"`     // mmproj GGUF path
	VisionPort      int    `json:"vision_port"`       // llama-server port while loaded
	VisionCtx       int    `json:"vision_ctx"`        // llama-server -c
	VisionThreads   int    `json:"vision_threads"`    // llama-server -t
	VisionMaxTokens int    `json:"vision_max_tokens"` // answer cap
	VisionTimeoutS  int    `json:"vision_timeout_s"`  // per-request ask timeout

	// Persona assets (read verbatim; wording is never rewritten, only
	// placeholder substitution {user_name}/{city}, exactly like the original).
	PersonaFile string `json:"persona_file"` // surface persona (表层.txt)
	ScenesFile  string `json:"scenes_file"`  // scenes.json, surface.context
	UserName    string `json:"user_name"`
	City        string `json:"city"`

	// Context management.
	ContextTokens      int  `json:"context_tokens"`       // total window target (12288)
	SystemBudgetTokens int  `json:"system_budget_tokens"` // cap for the static system prompt when trimming is on
	PersonaTrim        bool `json:"persona_trim"`         // trim persona to SystemBudgetTokens; default off (full load at 12k)
	RecentTurns        int  `json:"recent_turns"`         // hard cap on turns fed back (15)
	GenReserveTokens   int  `json:"gen_reserve_tokens"`   // room left for the reply

	// Memory.
	MemoryDir         string `json:"memory_dir"`
	MemoryBudgetChars int    `json:"memory_budget_chars"` // injected memory cap (3600)
	MemoryRecallN     int    `json:"memory_recall_n"`     // packets per recall (4)
	MemoryLanguage    string `json:"memory_language"`     // language facts are extracted/stored in; the original's build_f.json had "zh", PotatoAmika uses "en" (checklist item 13)

	// TTS host (tts-engine's tts_host, stdin JSON-lines protocol).
	TTSEnabled bool   `json:"tts_enabled"`
	TTSBin     string `json:"tts_bin"`
	TTSRoot    string `json:"tts_root"`   // --root (tts-engine tree)
	TTSEngine  string `json:"tts_engine"` // ort|free, passed through
	TTSThreads int    `json:"tts_threads"`
	TTSOutDir  string `json:"tts_outdir"`
	TTSSeed    int64  `json:"tts_seed"`

	// Tools (MCP). Empty path / missing file = tool stage never starts (v1).
	MCPConfigFile string `json:"mcp_config_file"`

	// Generation defaults (Minico GGUF README band, as used in quality-screen).
	ReplyTemperature float64 `json:"reply_temperature"`
	ReplyTopK        int     `json:"reply_top_k"`
	ReplyRepeatPen   float64 `json:"reply_repeat_penalty"`
	ReplyMaxTokens   int     `json:"reply_max_tokens"`

	// Reply length control (two layers, both default on): the runtime frame
	// carries an explicit brevity instruction (prompt layer), and the turn
	// pipeline hard-caps how many sentences reach TTS, aborting generation
	// once the cap is hit and closing the turn on the released sentences
	// only (no half sentence is ever spoken). ReplyMaxSentences <= 0
	// disables the pipeline cap.
	ReplyBrevity      bool `json:"reply_brevity"`
	ReplyMaxSentences int  `json:"reply_max_sentences"`
}

func defaultConfig() Config {
	return Config{
		Listen:             "127.0.0.1:18100",
		BrainSpawn:         true,
		BrainPort:          18091,
		BrainCtx:           12288,
		BrainThreads:       2,
		EmbedSpawn:         true,
		EmbedPort:          18092,
		EmbedThreads:       2,
		VisionPort:         18093,
		VisionCtx:          4096,
		VisionThreads:      2,
		VisionMaxTokens:    300,
		VisionTimeoutS:     300,
		UserName:           "friend",
		City:               "the city",
		ContextTokens:      12288,
		SystemBudgetTokens: 2300,
		PersonaTrim:        false,
		RecentTurns:        15,
		GenReserveTokens:   512,
		MemoryBudgetChars:  3600,
		MemoryRecallN:      4,
		MemoryLanguage:     "en",
		TTSEnabled:         true,
		TTSEngine:          "free",
		TTSThreads:         2,
		TTSSeed:            42,
		ReplyTemperature:   0.3,
		ReplyTopK:          50,
		ReplyRepeatPen:     1.05,
		ReplyMaxTokens:     220,
		ReplyBrevity:       true,
		ReplyMaxSentences:  3,
	}
}

func loadConfig(path string) (Config, error) {
	cfg := defaultConfig()
	if path == "" {
		return cfg, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return cfg, err
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return cfg, err
	}
	if cfg.BrainURL == "" && cfg.BrainSpawn {
		cfg.BrainURL = "http://127.0.0.1:" + itoa(cfg.BrainPort)
	}
	if cfg.EmbedURL == "" && cfg.EmbedSpawn {
		cfg.EmbedURL = "http://127.0.0.1:" + itoa(cfg.EmbedPort)
	}
	return cfg, nil
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var buf [12]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}
