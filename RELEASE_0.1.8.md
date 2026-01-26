# HORUS 0.1.8 Release Notes

**Release Date:** January 2026

## Performance & Architecture Improvements

This release focuses on **maximizing runtime performance**, **scalability**, and **debugging capability** by separating concerns and removing overhead from the core scheduler.

### Breaking Changes

#### 1. Scheduler API: `priority` → `order`

The scheduler builder API has been renamed for clarity:

```rust
// Before (0.1.7)
scheduler.add(node).priority(0).done();

// After (0.1.8)
scheduler.add(node).order(0).done();
```

**Why:** "Order" better reflects execution sequencing, while "priority" implied preemption (which HORUS doesn't support). Lower numbers execute first.

#### 2. Logging Removed from Core Runtime

**Context management and logging have been decoupled from the core scheduler:**

- **Removed:** `enable_logging` parameter from node registration
- **Removed:** Built-in `NodeContext` from scheduler
- **Result:** Zero logging overhead in hot path (~133ns improvement per tick)

**Migration:**

```rust
// Before (0.1.7)
scheduler.add(node).priority(0).enable_logging(true).done();

// After (0.1.8)
scheduler.add(node).order(0).done();

// Use send_logged() explicitly when needed
topic.send_logged(msg).ok();  // ~300ns (with logging)
topic.send(msg).ok();          // ~167ns (no logging)
```

#### 3. Context Separation

Node context (for debugging/monitoring) is now **opt-in** and managed separately:

```rust
// Debugging/monitoring is now external to core runtime
// Use `horus monitor` or custom telemetry integrations
```

## Benefits

### Performance
- **Faster hot path:** Removed context overhead from every tick
- **Lower memory:** No per-node context allocation by default
- **Better cache:** Tighter scheduler loop

### Scalability
- **Cleaner separation:** Runtime vs. observability
- **Flexible debugging:** Add telemetry when needed, not always-on
- **Production-ready:** Deploy without monitoring tax

### Developer Experience
- **Explicit logging:** `send_logged()` makes overhead visible
- **Simpler defaults:** New projects start lean
- **Better names:** `.order()` is clearer than `.priority()`

## Migration Guide

1. **Rename `priority()` to `order()`** in all `scheduler.add()` calls
2. **Remove `enable_logging()`** parameters (no longer needed)
3. **Use `send_logged()`** only where you need monitoring
4. **Use `send()`** for high-frequency hot paths

## Template Updates

The `horus new` templates now reflect these changes:
- Non-macro template includes `get_publishers()` for topic discovery
- Macro paths fixed for proper module resolution
- Default to `send_logged()` in examples (easy to optimize later)

## Full Changelog

- **BREAKING:** Rename `priority()` → `order()` in scheduler API
- **BREAKING:** Remove `enable_logging` from scheduler builder
- **BREAKING:** Remove automatic `NodeContext` allocation
- **FIX:** Non-macro template now implements `get_publishers()`
- **FIX:** Macro module paths use absolute `::horus::horus_core::`
- **PERF:** Disable safety monitor by default (prototyping-friendly)
- **PERF:** Zero logging overhead when using `send()`

---

**Upgrade today:**
```bash
cargo install horus_manager --force
pip install --upgrade horus-robotics
```

For questions or issues, visit: https://github.com/softmata/horus/issues
