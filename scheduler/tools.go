package main

import (
	"context"
	"encoding/json"
	"os"
)

// Tool layer (checklist item 12): only an MCP client seam is kept. v1 ships
// with the tool stage disabled — the brain's native tool calls measured
// 8/16, below the 90% bar — and with no MCP config file the stage is never
// entered at all. Nothing in the turn pipeline reads native tool_calls.
//
// A future enablement is two-stage by design (few-shot intent classification
// with a single-token constraint, then json_schema grammar argument fill);
// it plugs in behind this interface.

type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Schema      json.RawMessage `json:"input_schema"`
}

type MCPClient interface {
	Tools(ctx context.Context) ([]Tool, error)
	Call(ctx context.Context, name string, args json.RawMessage) (json.RawMessage, error)
	Close() error
}

// dormantMCP is the v1 implementation: no servers configured, no tools.
type dormantMCP struct{}

func (dormantMCP) Tools(context.Context) ([]Tool, error) { return nil, nil }
func (dormantMCP) Call(context.Context, string, json.RawMessage) (json.RawMessage, error) {
	return nil, errToolsDisabled
}
func (dormantMCP) Close() error { return nil }

var errToolsDisabled = &toolsError{"tool stage disabled in v1"}

type toolsError struct{ s string }

func (e *toolsError) Error() string { return e.s }

// toolStageEnabled reports whether a tool stage may run at all. It requires
// an existing, non-empty MCP config file; otherwise the pipeline skips the
// stage without even constructing a client session.
func toolStageEnabled(cfg Config) bool {
	if cfg.MCPConfigFile == "" {
		return false
	}
	fi, err := os.Stat(cfg.MCPConfigFile)
	return err == nil && fi.Size() > 0
}

func newMCPClient(cfg Config) MCPClient {
	// Even with a config file present, v1 keeps the stage dormant: the
	// two-stage enablement path is not implemented yet (see SCHEDULER.md).
	return dormantMCP{}
}
