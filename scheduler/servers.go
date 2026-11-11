package main

import (
	"context"
	"fmt"
	"log"
	"os/exec"
	"sync"
	"syscall"
	"time"
)

// procHandle owns one spawned llama-server process. A reaper goroutine
// runs Wait exactly once; stop() asks politely (SIGTERM) first and only
// escalates to SIGKILL after the grace period.
type procHandle struct {
	cmd  *exec.Cmd
	done chan struct{}
}

func startProc(bin string, args ...string) (*procHandle, error) {
	cmd, err := spawnServer(bin, args...)
	if err != nil {
		return nil, err
	}
	p := &procHandle{cmd: cmd, done: make(chan struct{})}
	go func() {
		cmd.Wait()
		close(p.done)
	}()
	return p, nil
}

func (p *procHandle) pid() int {
	if p == nil || p.cmd == nil || p.cmd.Process == nil {
		return 0
	}
	return p.cmd.Process.Pid
}

// stop terminates the process and waits for it to actually exit (the
// memory is only free once it is gone, which is the whole point of the
// vision swap). Safe on a nil or already-dead handle.
func (p *procHandle) stop(grace time.Duration) {
	if p == nil || p.cmd == nil || p.cmd.Process == nil {
		return
	}
	select {
	case <-p.done:
		return
	default:
	}
	p.cmd.Process.Signal(syscall.SIGTERM)
	select {
	case <-p.done:
		return
	case <-time.After(grace):
	}
	p.cmd.Process.Kill()
	<-p.done
}

// serverManager owns the llama-server subprocesses the scheduler spawned
// (brain, embed, and the on-demand vision server). It is the only place
// that starts or stops them, so the vision swap can stop brain+embed and
// later restore them with exactly the arguments main() used.
type serverManager struct {
	cfg    Config
	brain  *Brain // chat client (base = BrainURL)
	embed  *Brain // health/embedding client (base = EmbedURL)
	vision *Brain // vision client (base = vision URL, only up during a photo)

	mu     sync.Mutex
	brainP *procHandle
	embedP *procHandle
	visP   *procHandle
}

func newServerManager(cfg Config, brain *Brain) *serverManager {
	m := &serverManager{cfg: cfg, brain: brain}
	if cfg.EmbedURL != "" {
		ecfg := cfg
		ecfg.BrainURL = cfg.EmbedURL
		m.embed = newBrain(ecfg)
	}
	vcfg := cfg
	vcfg.BrainURL = "http://127.0.0.1:" + itoa(cfg.VisionPort)
	m.vision = newBrain(vcfg)
	return m
}

// visionResident reports whether vision lives inside the brain server
// (Omni profile: the brain was spawned with its mmproj, so the same
// process that chats also answers photos and transcribes audio).
func (m *serverManager) visionResident() bool {
	return m != nil && m.cfg.BrainMmproj != ""
}

// visionConfigured reports whether the vision flow can run at all:
// either resident in the brain (brain_mmproj set), or the legacy swap,
// which needs model files configured and ownership of the brain and
// embed processes (attach mode cannot stop servers it did not spawn).
func (m *serverManager) visionConfigured() bool {
	if m == nil {
		return false
	}
	if m.cfg.BrainMmproj != "" {
		return true
	}
	return m.cfg.VisionModel != "" && m.cfg.VisionMmproj != "" &&
		m.cfg.BrainSpawn && m.cfg.EmbedSpawn
}

func (m *serverManager) brainArgs() []string {
	args := []string{"-m", m.cfg.BrainModel, "-c", itoa(m.cfg.BrainCtx),
		"--host", "127.0.0.1", "--port", itoa(m.cfg.BrainPort),
		"-t", itoa(m.cfg.BrainThreads), "--jinja"}
	if m.cfg.BrainMmproj != "" {
		args = append(args, "--mmproj", m.cfg.BrainMmproj)
	}
	if m.cfg.BrainThinking == "lfm3" {
		// Mirror the spike's gated setup: reasoning off at the server
		// level too (requests also send enable_thinking=false).
		args = append(args, "--reasoning", "off")
	}
	return args
}

func (m *serverManager) embedArgs() []string {
	return []string{"-m", m.cfg.EmbedModel, "-c", "1024",
		"--host", "127.0.0.1", "--port", itoa(m.cfg.EmbedPort),
		"-t", itoa(m.cfg.EmbedThreads), "--embedding", "--pooling", "mean"}
}

func (m *serverManager) visionArgs() []string {
	return []string{"-m", m.cfg.VisionModel, "--mmproj", m.cfg.VisionMmproj,
		"-c", itoa(m.cfg.VisionCtx),
		"--host", "127.0.0.1", "--port", itoa(m.cfg.VisionPort),
		"-t", itoa(m.cfg.VisionThreads)}
}

// waitUp polls /health until the server answers, the deadline passes,
// or (when we own the process) the process exits — a bad model path
// must fail in seconds, not after the full deadline, so the vision
// rollback can start right away.
func (m *serverManager) waitUp(ctx context.Context, b *Brain, what string, p *procHandle) error {
	deadline := time.Now().Add(180 * time.Second)
	for time.Now().Before(deadline) {
		if p != nil {
			select {
			case <-p.done:
				return fmt.Errorf("%s server exited during startup", what)
			default:
			}
		}
		if b.healthy(ctx) {
			if p == nil {
				return nil
			}
			// A healthy answer alone is not proof when we own the
			// process: a stale server squatting on the port would
			// answer for it while our child dies on the bind error
			// (seen in dev testing after a crashed scheduler left
			// orphans behind). The child must survive a short
			// settle window after the first answer.
			select {
			case <-p.done:
				return fmt.Errorf("%s server exited during startup", what)
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(1 * time.Second):
			}
			if b.healthy(ctx) {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}
	return fmt.Errorf("%s server did not become healthy", what)
}

// startBrainEmbed spawns the brain and embed servers (the arguments and
// health gate are the same ones startup uses) and waits until both are
// healthy. In attach mode (spawn flag off) it only health-checks the
// external instance. On failure everything it started is stopped again.
func (m *serverManager) startBrainEmbed(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.brainP == nil && m.cfg.BrainSpawn {
		p, err := startProc(m.cfg.LlamaServerBin, m.brainArgs()...)
		if err != nil {
			return fmt.Errorf("spawn brain: %w", err)
		}
		m.brainP = p
		log.Printf("brain llama-server pid %d on :%d", p.pid(), m.cfg.BrainPort)
	}
	if err := m.waitUp(ctx, m.brain, "brain", m.brainP); err != nil {
		m.stopBrainEmbedLocked()
		return err
	}
	if m.cfg.EmbedURL != "" && m.embedP == nil && m.cfg.EmbedSpawn {
		p, err := startProc(m.cfg.LlamaServerBin, m.embedArgs()...)
		if err != nil {
			m.stopBrainEmbedLocked()
			return fmt.Errorf("spawn embed: %w", err)
		}
		m.embedP = p
		log.Printf("embed llama-server pid %d on :%d", p.pid(), m.cfg.EmbedPort)
	}
	if m.embed != nil {
		if err := m.waitUp(ctx, m.embed, "embed", m.embedP); err != nil {
			m.stopBrainEmbedLocked()
			return err
		}
	}
	return nil
}

func (m *serverManager) stopBrainEmbedLocked() {
	if m.embedP != nil {
		m.embedP.stop(15 * time.Second)
		m.embedP = nil
	}
	if m.brainP != nil {
		m.brainP.stop(15 * time.Second)
		m.brainP = nil
	}
}

// stopBrainEmbed terminates both resident servers and waits for them to
// exit, so their memory is actually released before vision loads.
func (m *serverManager) stopBrainEmbed() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stopBrainEmbedLocked()
}

// startVision spawns the vision server and waits for /health.
func (m *serverManager) startVision(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.visP == nil {
		p, err := startProc(m.cfg.LlamaServerBin, m.visionArgs()...)
		if err != nil {
			return fmt.Errorf("spawn vision: %w", err)
		}
		m.visP = p
		log.Printf("vision llama-server pid %d on :%d", p.pid(), m.cfg.VisionPort)
	}
	if err := m.waitUp(ctx, m.vision, "vision", m.visP); err != nil {
		m.stopVisionLocked()
		return err
	}
	return nil
}

func (m *serverManager) stopVisionLocked() {
	if m.visP != nil {
		m.visP.stop(15 * time.Second)
		m.visP = nil
	}
}

func (m *serverManager) stopVision() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stopVisionLocked()
}

// stopAll is the shutdown path: every spawned server goes down.
func (m *serverManager) stopAll() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stopVisionLocked()
	m.stopBrainEmbedLocked()
}
