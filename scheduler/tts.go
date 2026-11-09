package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"sync"
)

// TTSClient drives tts-engine's tts_host over its stdin/stdout JSON-lines
// protocol (STAGE2.md): one request line in, one result line out, wav files
// land in the host's --outdir. Requests are serialized through a single
// worker goroutine, which is what makes the per-sentence pipeline safe:
// the scheduler can enqueue sentence N+1 while sentence N is synthesizing.
type TTSClient struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	lines  chan string
	jobs   chan ttsJob
	stderr io.Writer
}

type ttsRequest struct {
	Text        string  `json:"text"`
	Tag         string  `json:"tag"`
	Temperature float64 `json:"temperature,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
	Seed        int64   `json:"seed,omitempty"`
}

type TTSResult struct {
	File      string  `json:"file"`
	DurationS float64 `json:"duration_s"`
	ElapsedMs float64 `json:"elapsed_ms"`
	TagUsed   string  `json:"tag_used"`
	RSSMB     float64 `json:"rss_mb"`
	Error     string  `json:"error,omitempty"`
	RawJSON   string  `json:"-"`
}

type ttsJob struct {
	ctx    context.Context
	req    ttsRequest
	result chan ttsOutcome
}

type ttsOutcome struct {
	res TTSResult
	err error
}

func startTTS(cfg Config, stderr io.Writer) (*TTSClient, error) {
	args := []string{
		"--root", cfg.TTSRoot,
		"--engine", cfg.TTSEngine,
		"--threads", itoa(cfg.TTSThreads),
		"--outdir", cfg.TTSOutDir,
	}
	cmd := exec.Command(cfg.TTSBin, args...)
	cmd.Stderr = stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	c := &TTSClient{
		cmd:    cmd,
		stdin:  stdin,
		lines:  make(chan string, 4),
		jobs:   make(chan ttsJob, 64),
		stderr: stderr,
	}
	go func() {
		sc := bufio.NewScanner(stdout)
		sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		for sc.Scan() {
			c.lines <- sc.Text()
		}
		close(c.lines)
	}()
	go c.worker()
	return c, nil
}

func (c *TTSClient) worker() {
	for job := range c.jobs {
		if job.ctx.Err() != nil {
			job.result <- ttsOutcome{err: job.ctx.Err()}
			continue
		}
		data, _ := json.Marshal(job.req)
		if _, err := fmt.Fprintf(c.stdin, "%s\n", data); err != nil {
			job.result <- ttsOutcome{err: fmt.Errorf("tts write: %w", err)}
			continue
		}
		// Await the result line, skipping (and logging) anything that is
		// not a JSON object, so a stray host line can never desync the
		// request/response pairing.
		var line string
		var ok bool
		for {
			line, ok = <-c.lines
			if !ok {
				break
			}
			if len(line) > 0 && line[0] == '{' {
				break
			}
			fmt.Fprintf(c.stderr, "[scheduler] skipped non-result tts line: %.200s\n", line)
		}
		if !ok {
			job.result <- ttsOutcome{err: fmt.Errorf("tts host exited")}
			continue
		}
		var res TTSResult
		if err := json.Unmarshal([]byte(line), &res); err != nil {
			job.result <- ttsOutcome{err: fmt.Errorf("tts result parse: %w", err)}
			continue
		}
		res.RawJSON = line
		job.result <- ttsOutcome{res: res}
	}
}

// Synthesize enqueues one sentence and waits for its result. Safe to call
// from the pipeline: enqueueing is decoupled from completion, so callers
// that only need fire-and-forget can use SynthesizeAsync.
func (c *TTSClient) Synthesize(ctx context.Context, req ttsRequest) (TTSResult, error) {
	ch := c.SynthesizeAsync(ctx, req)
	select {
	case out := <-ch:
		return out.res, out.err
	case <-ctx.Done():
		return TTSResult{}, ctx.Err()
	}
}

func (c *TTSClient) SynthesizeAsync(ctx context.Context, req ttsRequest) chan ttsOutcome {
	ch := make(chan ttsOutcome, 1)
	c.jobs <- ttsJob{ctx: ctx, req: req, result: ch}
	return ch
}

// DropQueued discards not-yet-started jobs (used on interrupt). A synthesis
// already inside the host cannot be aborted mid-flight; it finishes and its
// result is dropped by the cancelled context.
func (c *TTSClient) DropQueued() {
	for {
		select {
		case job := <-c.jobs:
			job.result <- ttsOutcome{err: context.Canceled}
		default:
			return
		}
	}
}

func (c *TTSClient) Close() {
	close(c.jobs)
	c.stdin.Close()
	if c.cmd.Process != nil {
		c.cmd.Process.Kill()
		c.cmd.Wait()
	}
}

var _ = sync.Once{}
