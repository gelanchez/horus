# HORUS Developer MCP

A Model Context Protocol (MCP) server that boosts Claude Code productivity when developing HORUS.

## Features

### Codebase Navigation
- `horus_get_architecture` - Get high-level overview of HORUS architecture
- `horus_find_module` - Find where structs, traits, functions are defined
- `horus_get_crate_info` - Get detailed info about a specific crate
- `horus_list_nodes` - List all built-in node implementations
- `horus_get_message_types` - List all message types for IPC
- `horus_find_tests` - Find tests related to a module
- `horus_get_dependencies` - Get dependency graph

### Runtime Debugging
- `horus_list_topics` - List active shared memory topics
- `horus_inspect_topic` - Get details about a specific topic
- `horus_list_sessions` - List active HORUS sessions
- `horus_get_node_health` - Check node heartbeats
- `horus_tail_logs` - Get recent log entries
- `horus_check_shm` - Check shared memory state
- `horus_kill_session` - Kill a running session
- `horus_clean_shm` - Clean up stale shared memory

### Benchmark Tracking
- `horus_run_benchmark` - Run benchmarks and store results
- `horus_get_benchmark_history` - Get historical results
- `horus_compare_benchmark` - Compare against baseline, detect regressions
- `horus_set_baseline` - Set current results as baseline
- `horus_list_benchmarks` - List available benchmarks

### Validation
- `horus_check_docs` - Check docs for broken links, code samples
- `horus_check_api_coverage` - Check documentation coverage
- `horus_validate_cargo` - Run cargo check and clippy
- `horus_check_tests` - Run test suite
- `horus_check_formatting` - Check rustfmt compliance
- `horus_find_todos` - Find TODO/FIXME comments
- `horus_check_dead_code` - Find dead code warnings

## Installation

The MCP is automatically available when working in the HORUS directory with Claude Code.

### Manual Setup

```bash
cd horus_mcp
npm install
npm run build
```

### Configuration

The MCP is configured via `.claude/mcp.json` in the HORUS root:

```json
{
  "mcpServers": {
    "horus-mcp": {
      "command": "node",
      "args": ["/path/to/horus/horus_mcp/dist/index.js"],
      "env": {
        "HORUS_ROOT": "/path/to/horus"
      }
    }
  }
}
```

## Usage

When Claude Code starts in the HORUS directory, the MCP tools become available automatically.

Example prompts:
- "Show me the HORUS architecture"
- "Find where the Scheduler is defined"
- "Check if there are any performance regressions"
- "List active topics in shared memory"
- "Run the link_performance benchmark"

## Data Storage

Benchmark metrics are stored in `~/.horus-mcp/metrics.db` (SQLite).
