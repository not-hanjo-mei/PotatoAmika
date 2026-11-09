package main

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Packet is one durable fact about the user plus its embedding.
// Memory is a flat JSON file of packets (checklist item 11): atomic writes,
// brute-force cosine recall, built from zero.
type Packet struct {
	ID      string    `json:"id"`
	Text    string    `json:"text"`
	Vec     []float32 `json:"vec,omitempty"`
	Created time.Time `json:"created"`
}

type memoryFile struct {
	Packets []Packet `json:"packets"`
}

type MemoryStore struct {
	mu      sync.Mutex
	path    string
	packets []Packet
	embed   func(ctx context.Context, text string) ([]float32, error)
}

func openMemory(dir string, embedFn func(ctx context.Context, text string) ([]float32, error)) (*MemoryStore, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	m := &MemoryStore{path: filepath.Join(dir, "memory.json"), embed: embedFn}
	b, err := os.ReadFile(m.path)
	if err == nil {
		var f memoryFile
		if err := json.Unmarshal(b, &f); err == nil {
			m.packets = f.Packets
		}
	}
	return m, nil
}

func factID(text string) string {
	h := sha1.Sum([]byte(strings.ToLower(strings.Join(strings.Fields(text), " "))))
	return hex.EncodeToString(h[:8])
}

// AddFacts embeds and stores new facts, skipping exact duplicates
// (normalized-text hash). Returns how many were added.
func (m *MemoryStore) AddFacts(ctx context.Context, facts []string) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	seen := map[string]bool{}
	for _, p := range m.packets {
		seen[p.ID] = true
	}
	added := 0
	for _, f := range facts {
		id := factID(f)
		if seen[id] {
			continue
		}
		seen[id] = true
		p := Packet{ID: id, Text: f, Created: time.Now()}
		if m.embed != nil {
			vec, err := m.embed(ctx, f)
			if err == nil {
				p.Vec = vec
			} else {
				log.Printf("memory: embed failed for fact %q: %v (stored without vector)", f, err)
			}
		}
		m.packets = append(m.packets, p)
		added++
	}
	if added > 0 {
		if err := m.saveLocked(); err != nil {
			return added, err
		}
	}
	return added, nil
}

// saveLocked writes via temp file + rename so a crash cannot tear the file.
func (m *MemoryStore) saveLocked() error {
	data, err := json.MarshalIndent(memoryFile{Packets: m.packets}, "", " ")
	if err != nil {
		return err
	}
	tmp := m.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	f, err := os.Open(tmp)
	if err == nil {
		f.Sync()
		f.Close()
	}
	return os.Rename(tmp, m.path)
}

type scoredPacket struct {
	packet Packet
	score  float64
}

func cosine(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// Recall returns up to n packets most similar to the query, trimmed so the
// total injected text stays within budgetChars (checklist: 4 packets,
// 3600 chars total memory budget).
func (m *MemoryStore) Recall(ctx context.Context, query string, n, budgetChars int) ([]Packet, error) {
	m.mu.Lock()
	packets := append([]Packet(nil), m.packets...)
	m.mu.Unlock()
	if len(packets) == 0 || m.embed == nil {
		return nil, nil
	}
	qv, err := m.embed(ctx, query)
	if err != nil {
		return nil, err
	}
	scored := make([]scoredPacket, 0, len(packets))
	for _, p := range packets {
		if len(p.Vec) == 0 {
			continue
		}
		scored = append(scored, scoredPacket{p, cosine(qv, p.Vec)})
	}
	sort.Slice(scored, func(i, j int) bool { return scored[i].score > scored[j].score })
	var out []Packet
	used := 0
	for _, sp := range scored {
		if len(out) >= n {
			break
		}
		if used+len(sp.packet.Text) > budgetChars && len(out) > 0 {
			break
		}
		used += len(sp.packet.Text)
		out = append(out, sp.packet)
	}
	return out, nil
}

func (m *MemoryStore) Count() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.packets)
}

// Extractor is the background-worker seam for memory extraction. v1 uses the
// synchronous implementation; an idle-time worker can replace it later
// without touching the turn pipeline.
type Extractor interface {
	Extract(ctx context.Context, convo []chatMessage) (added int, err error)
}

type syncExtractor struct {
	brain *Brain
	mem   *MemoryStore
}

func (e *syncExtractor) Extract(ctx context.Context, convo []chatMessage) (int, error) {
	facts, err := e.brain.ExtractFacts(ctx, convo)
	if err != nil {
		return 0, err
	}
	if len(facts) == 0 {
		return 0, nil
	}
	return e.mem.AddFacts(ctx, facts)
}
