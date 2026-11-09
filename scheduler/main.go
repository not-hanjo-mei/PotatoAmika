package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

// spawnServer starts a llama-server subprocess and waits for /health.
func spawnServer(bin string, args ...string) (*exec.Cmd, error) {
	cmd := exec.Command(bin, args...)
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd, nil
}

func waitHealthy(ctx context.Context, b *Brain, what string) error {
	deadline := time.Now().Add(180 * time.Second)
	for time.Now().Before(deadline) {
		if b.healthy(ctx) {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}
	return fmt.Errorf("%s server did not become healthy", what)
}

func main() {
	cfgPath := flag.String("config", "", "config JSON (defaults + this file)")
	flag.Parse()

	cfg, err := loadConfig(*cfgPath)
	if err != nil {
		log.Fatalf("config: %v", err)
	}
	ctx := context.Background()
	brain := newBrain(cfg)
	// All llama-server subprocesses live in the server manager: startup,
	// the vision swap, and shutdown all go through it (servers.go).
	mgr := newServerManager(cfg, brain)
	// On SIGINT/SIGTERM take the spawned servers down with us.
	sigCh := make(chan os.Signal, 2)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigCh
		mgr.stopAll()
		os.Exit(0)
	}()

	// Brain + embedding servers (spawn or attach, per config).
	if err := mgr.startBrainEmbed(ctx); err != nil {
		log.Fatalf("servers: %v", err)
	}

	// Persona + static system prompt (built once; fully static per request).
	persona, err := loadPersona(cfg)
	if err != nil {
		log.Fatalf("persona: %v", err)
	}
	system := buildSystemPrompt(cfg, persona, time.Now())
	if n, err := brain.Tokenize(ctx, system); err == nil {
		log.Printf("system prompt: %d tokens (exact), est %d", n, estTokens(system))
	} else {
		log.Printf("system prompt: est %d tokens", estTokens(system))
	}
	if n, err := brain.Tokenize(ctx, persona); err == nil {
		log.Printf("persona file: %d tokens (exact), %d chars", n, len([]rune(persona)))
	}

	// Memory store (from zero; JSON file + atomic writes).
	mem, err := openMemory(cfg.MemoryDir, brain.Embed)
	if err != nil {
		log.Fatalf("memory: %v", err)
	}
	log.Printf("memory: %d packets loaded", mem.Count())

	// TTS host.
	var tts *TTSClient
	if cfg.TTSEnabled {
		if err := os.MkdirAll(cfg.TTSOutDir, 0o755); err != nil {
			log.Fatalf("tts outdir: %v", err)
		}
		tts, err = startTTS(cfg, os.Stderr)
		if err != nil {
			log.Fatalf("tts: %v", err)
		}
		log.Printf("tts_host started (engine=%s)", cfg.TTSEngine)
	}

	tags := loadTags(cfg)
	orch := &Orchestrator{
		cfg:      cfg,
		brain:    brain,
		mem:      mem,
		tts:      tts,
		tags:     tags,
		system:   system,
		toolsOn:  toolStageEnabled(cfg),
		mcp:      newMCPClient(cfg),
		sessions: map[string]*Session{},
		mgr:      mgr,
	}
	if orch.toolsOn {
		log.Printf("MCP config present, but v1 keeps the tool stage dormant")
	}
	orch.extract = &syncExtractor{brain: brain, mem: mem}

	api := &apiServer{orch: orch}
	srv := &http.Server{Addr: cfg.Listen, Handler: api.routes()}
	log.Printf("scheduler listening on %s", cfg.Listen)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("serve: %v", err)
	}
}

// loadTags returns the valid TTS emotion tags: the keys of the tts-engine
// refpack index when available, else a built-in fallback of the same set.
func loadTags(cfg Config) []string {
	if cfg.TTSRoot != "" {
		b, err := os.ReadFile(filepath.Join(cfg.TTSRoot, "host", "refpacks", "index.json"))
		if err == nil {
			var idx map[string]json.RawMessage
			if err := json.Unmarshal(b, &idx); err == nil {
				var tags []string
				for k := range idx {
					if !strings.HasPrefix(k, "_") {
						tags = append(tags, k)
					}
				}
				if len(tags) > 0 {
					return tags
				}
			}
		}
	}
	return []string{"neutral1", "happy1", "happy2", "caring1", "sad1", "angry1",
		"playful1", "shy1", "surprise1", "worried1", "tired1", "calm1"}
}
