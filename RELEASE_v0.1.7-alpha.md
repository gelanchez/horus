# HORUS v0.1.7-alpha

Alpha release with significant architectural changes.

## Communication Layer Redesign

The Hub/Link architecture has been unified into a single **Topic** API. This change aligns HORUS with ROS2 terminology, reducing cognitive overhead for developers transitioning from ROS2 ecosystems.

```rust
// Old: Separate Hub and Link with different configurations
let hub = Hub::new("sensor_data")?;
let link = Link::new("sensor_data")?;

// New: Unified Topic API
let topic: Topic<SensorData> = Topic::new("sensor_data")?;
topic.send(data, &mut None)?;
let received = topic.recv(&mut None);
```

The framework now automatically selects the optimal IPC backend based on message type and access patterns:

| Backend | Latency | Use Case |
|---------|---------|----------|
| PodShm | ~50ns | POD types, cross-process |
| SpscShm | ~85ns | Single producer/consumer |
| MpmcShm | ~167ns | Multiple producers/consumers |

## Scheduler Improvements

The scheduler now supports Linux real-time scheduling policies:

- `SCHED_FIFO` for deterministic real-time execution
- `SCHED_RR` for round-robin real-time scheduling
- Configurable priority levels (1-99)

Requires `CAP_SYS_NICE` capability or root privileges on Linux.

## Codebase Restructuring

This release includes a major refactor to support future scaling:

- **sim2d and sim3d**: Now developed as standalone packages. They will continue to be maintained separately.
- **Built-in nodes**: Moved to separate development. These will be open-sourced when feature-complete.
- **Core crates**: Restructured for modularity (`horus_core`, `horus_manager`, `horus_ai`, `horus_perception`)

## API Changes

Several APIs have changed for improved developer experience. See migration notes in the documentation.

## Roadmap

- Current: v0.1.x (Alpha)
- Expected Beta: v0.5.x

---

Full changelog and documentation at: https://github.com/softmata/horus
