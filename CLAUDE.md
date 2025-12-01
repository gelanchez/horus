# HORUS Development Guide

This is the HORUS robotics framework - a high-performance, real-time robotics system written in Rust.

## MCP Tools Available

The `horus-mcp` server provides specialized tools for HORUS development. Use these to work more efficiently:

### Start Here
- `horus_get_architecture` - Understand HORUS structure before making changes
- `horus_find_module <name>` - Quickly locate where things are defined

### When Debugging
- `horus_list_topics` - See active IPC topics
- `horus_get_node_health` - Check if nodes are running
- `horus_tail_logs` - View recent log output

### Before Committing
- `horus_validate_cargo` - Run cargo check + clippy
- `horus_check_tests` - Run test suite
- `horus_compare_benchmark` - Check for performance regressions

## Project Structure

```
HORUS/
├── horus/           # Main unified crate (re-exports)
├── horus_core/      # Core runtime (Node, Hub, Link, Scheduler)
├── horus_macros/    # Procedural macros (node!, message!)
├── horus_manager/   # CLI and dashboard
├── horus_library/   # Built-in nodes, algorithms, messages
├── horus_mcp/       # This MCP server (dev tooling)
├── benchmarks/      # Performance benchmarks
└── docs-site/       # Documentation website
```

## Key Concepts

- **Node**: Computational unit with `tick()` method
- **Hub<T>**: Multi-producer multi-consumer pub/sub (~481ns latency)
- **Link<T>**: Single-producer single-consumer (~248ns latency)
- **Scheduler**: Orchestrates node execution with priorities

## Shared Memory Locations

- Linux: `/dev/shm/horus/`
- macOS: `/tmp/horus/`
- Topics: `{base}/topics/`
- Heartbeats: `{base}/heartbeats/`
- Sessions: `{base}/sessions/{id}/`

## Common Tasks

### Adding a new built-in node
1. Create in `horus_library/nodes/`
2. Add to `horus_library/lib.rs`
3. Add documentation in `docs-site/content/docs/built-in-nodes/`

### Running benchmarks
```bash
cargo bench -p horus_benchmarks --bench link_performance
```

### Running tests
```bash
cargo test --workspace
```

### Building the CLI
```bash
cargo build -p horus_manager --release
```

## Performance Targets

- Hub round-trip: <500ns
- Link round-trip: <300ns
- Scheduler tick overhead: <1µs
