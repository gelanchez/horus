// Benchmark binary - allow clippy warnings
#![allow(unused_imports)]
#![allow(clippy::all)]
#![allow(dead_code)]

//! # Backend Latency Benchmark
//!
//! Tests all 10 Topic backend configurations to verify zero-overhead IPC targets:
//!
//! | Backend            | Target p99  | Description                        |
//! |--------------------|-------------|------------------------------------|
//! | Direct(unchecked)  | < 10ns      | Same-thread, zero-overhead (~3-5ns)|
//! | Direct(safe)       | < 100ns     | Same-thread with safety checks     |
//! | SpscIntra          | < 100ns     | In-process SPSC ring buffer        |
//! | MpscIntra          | < 150ns     | In-process MPSC                    |
//! | SpmcIntra          | < 150ns     | In-process SPMC                    |
//! | MpmcIntra          | < 200ns     | In-process MPMC                    |
//! | SpscShm            | < 300ns     | Cross-process SPSC shared mem      |
//! | MpscShm            | < 400ns     | Cross-process MPSC shared mem      |
//! | SpmcShm            | < 400ns     | Cross-process SPMC shared mem      |
//! | MpmcShm            | < 500ns     | Cross-process MPMC shared mem      |
//!
//! ## Usage
//!
//! ```bash
//! cargo build --release --bin backend_latency_benchmark
//! ./target/release/backend_latency_benchmark
//! ```

use colored::Colorize;
use horus_core::communication::Topic;
use horus_core::core::LogSummary;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_rdtsc;

const ITERATIONS: usize = 100_000;
const WARMUP: usize = 10_000;

// Simple test message - small and serializable
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestMessage {
    sequence: u32,
    value: f64,
}

impl LogSummary for TestMessage {
    fn log_summary(&self) -> String {
        format!("TestMsg(seq={})", self.sequence)
    }
}

/// Read CPU timestamp counter
#[inline(always)]
fn rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _rdtsc()
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Fallback for non-x86_64: use Instant
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        let start = START.get_or_init(Instant::now);
        start.elapsed().as_nanos() as u64
    }
}

/// Calibrate rdtsc overhead
fn calibrate_rdtsc() -> u64 {
    let mut min_cost = u64::MAX;

    for _ in 0..100 {
        let _ = rdtsc();
    }

    for _ in 0..1000 {
        let start = rdtsc();
        let end = rdtsc();
        let cost = end.wrapping_sub(start);
        if cost > 0 && cost < min_cost {
            min_cost = cost;
        }
    }

    min_cost
}

/// Detect CPU frequency using RDTSC calibration
fn detect_cpu_frequency() -> f64 {
    println!("  Calibrating CPU frequency...");

    // Measure using RDTSC and Instant
    let start_tsc = rdtsc();
    let start_time = Instant::now();

    std::thread::sleep(Duration::from_millis(100));

    let end_tsc = rdtsc();
    let elapsed_ns = start_time.elapsed().as_nanos() as u64;

    let cycles = end_tsc.wrapping_sub(start_tsc);
    let freq_ghz = (cycles as f64) / (elapsed_ns as f64);

    if freq_ghz <= 0.0 || freq_ghz.is_nan() || freq_ghz.is_infinite() || freq_ghz < 0.5 || freq_ghz > 10.0 {
        // Fallback: try /proc/cpuinfo
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if line.starts_with("cpu MHz") {
                    if let Some(freq_str) = line.split(':').nth(1) {
                        if let Ok(mhz) = freq_str.trim().parse::<f64>() {
                            let ghz = mhz / 1000.0;
                            if ghz > 0.5 && ghz < 10.0 {
                                return ghz;
                            }
                        }
                    }
                }
            }
        }
        // Last resort default
        return 3.0;
    }

    freq_ghz
}

#[derive(Debug, Clone)]
struct LatencyStats {
    name: String,
    p50_ns: u64,
    p99_ns: u64,
    p999_ns: u64,
    min_ns: u64,
    max_ns: u64,
    mean_ns: f64,
}

impl LatencyStats {
    fn compute(name: &str, cycles: Vec<u64>, cpu_ghz: f64, rdtsc_overhead: u64) -> Self {
        // Remove rdtsc overhead and convert to nanoseconds
        let mut samples: Vec<u64> = cycles
            .iter()
            .map(|&c| {
                let adjusted = c.saturating_sub(rdtsc_overhead);
                (adjusted as f64 / cpu_ghz) as u64
            })
            .collect();

        samples.sort_unstable();

        let len = samples.len();
        let p50_idx = len * 50 / 100;
        let p99_idx = len * 99 / 100;
        let p999_idx = len * 999 / 1000;

        let sum: u64 = samples.iter().sum();
        let mean_ns = sum as f64 / len as f64;

        Self {
            name: name.to_string(),
            p50_ns: samples[p50_idx],
            p99_ns: samples[p99_idx],
            p999_ns: samples[p999_idx.min(len - 1)],
            min_ns: samples[0],
            max_ns: samples[len - 1],
            mean_ns,
        }
    }

    fn print(&self, target_ns: u64) {
        let status = if self.p99_ns <= target_ns {
            "✓ PASS".green()
        } else {
            "✗ FAIL".red()
        };

        println!(
            "  {:15} p50={:>5}ns  p99={:>5}ns  p999={:>6}ns  min={:>4}ns  max={:>6}ns  target={:>4}ns  {}",
            self.name,
            self.p50_ns,
            self.p99_ns,
            self.p999_ns,
            self.min_ns,
            self.max_ns,
            target_ns,
            status
        );
    }
}

/// Benchmark DirectChannel (same-thread) - Safe API
fn bench_direct_channel(cpu_ghz: f64, rdtsc_overhead: u64) -> LatencyStats {
    let topic_name = format!("bench_direct_{}", std::process::id());
    let topic: Topic<TestMessage> = Topic::direct(&topic_name);

    println!("  DirectChannel   using {}", topic.backend_type());

    // Warmup
    for i in 0..WARMUP {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let _ = topic.send(msg);
        let _ = topic.recv();
    }

    // Measure
    let mut latencies = Vec::with_capacity(ITERATIONS);
    for i in 0..ITERATIONS {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let start = rdtsc();
        let _ = topic.send(msg);
        let end = rdtsc();
        latencies.push(end.wrapping_sub(start));
        let _ = topic.recv();
    }

    LatencyStats::compute("Direct(safe)", latencies, cpu_ghz, rdtsc_overhead)
}

/// Benchmark DirectChannel (same-thread) - Unchecked API (~3-5ns target)
fn bench_direct_channel_unchecked(cpu_ghz: f64, rdtsc_overhead: u64) -> LatencyStats {
    let topic_name = format!("bench_direct_unc_{}", std::process::id());
    let topic: Topic<TestMessage> = Topic::direct(&topic_name);

    println!("  Direct(unchecked) using {}", topic.backend_type());

    // Warmup - SAFETY: Same thread, single-threaded benchmark, DirectChannel backend
    for i in 0..WARMUP {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        unsafe { topic.send_unchecked(msg) };
        let _ = unsafe { topic.recv_unchecked() };
    }

    // Measure - SAFETY: Same thread, single-threaded benchmark, DirectChannel backend
    let mut latencies = Vec::with_capacity(ITERATIONS);
    for i in 0..ITERATIONS {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let start = rdtsc();
        unsafe { topic.send_unchecked(msg) };
        let end = rdtsc();
        latencies.push(end.wrapping_sub(start));
        let _ = unsafe { topic.recv_unchecked() };
    }

    LatencyStats::compute("Direct(unchecked)", latencies, cpu_ghz, rdtsc_overhead)
}

/// Benchmark SpscIntra (in-process SPSC)
fn bench_spsc_intra(cpu_ghz: f64, rdtsc_overhead: u64) -> LatencyStats {
    let topic_name = format!("bench_spsc_intra_{}", std::process::id());
    let (producer, consumer) = Topic::<TestMessage>::spsc_intra(&topic_name);

    println!("  SpscIntra       using {}", producer.backend_type());

    // Warmup
    for i in 0..WARMUP {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let _ = producer.send(msg);
        let _ = consumer.recv();
    }

    // Measure
    let mut latencies = Vec::with_capacity(ITERATIONS);
    for i in 0..ITERATIONS {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let start = rdtsc();
        let _ = producer.send(msg);
        let end = rdtsc();
        latencies.push(end.wrapping_sub(start));
        let _ = consumer.recv();
    }

    LatencyStats::compute("SpscIntra", latencies, cpu_ghz, rdtsc_overhead)
}

/// Benchmark MpscIntra (in-process MPSC)
fn bench_mpsc_intra(cpu_ghz: f64, rdtsc_overhead: u64) -> LatencyStats {
    let topic_name = format!("bench_mpsc_intra_{}", std::process::id());
    let (producer, consumer) = Topic::<TestMessage>::mpsc_intra(&topic_name, 1024);

    println!("  MpscIntra       using {}", producer.backend_type());

    // Warmup
    for i in 0..WARMUP {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let _ = producer.send(msg);
        let _ = consumer.recv();
    }

    // Measure
    let mut latencies = Vec::with_capacity(ITERATIONS);
    for i in 0..ITERATIONS {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let start = rdtsc();
        let _ = producer.send(msg);
        let end = rdtsc();
        latencies.push(end.wrapping_sub(start));
        let _ = consumer.recv();
    }

    LatencyStats::compute("MpscIntra", latencies, cpu_ghz, rdtsc_overhead)
}

/// Benchmark SpmcIntra (in-process SPMC)
fn bench_spmc_intra(cpu_ghz: f64, rdtsc_overhead: u64) -> LatencyStats {
    let topic_name = format!("bench_spmc_intra_{}", std::process::id());
    let (producer, consumer) = Topic::<TestMessage>::spmc_intra(&topic_name);

    println!("  SpmcIntra       using {}", producer.backend_type());

    // Warmup
    for i in 0..WARMUP {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let _ = producer.send(msg);
        let _ = consumer.recv();
    }

    // Measure
    let mut latencies = Vec::with_capacity(ITERATIONS);
    for i in 0..ITERATIONS {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let start = rdtsc();
        let _ = producer.send(msg);
        let end = rdtsc();
        latencies.push(end.wrapping_sub(start));
        let _ = consumer.recv();
    }

    LatencyStats::compute("SpmcIntra", latencies, cpu_ghz, rdtsc_overhead)
}

/// Benchmark MpmcIntra (in-process MPMC)
fn bench_mpmc_intra(cpu_ghz: f64, rdtsc_overhead: u64) -> LatencyStats {
    let topic_name = format!("bench_mpmc_intra_{}", std::process::id());
    let (producer, consumer) = Topic::<TestMessage>::mpmc_intra(&topic_name, 1024);

    println!("  MpmcIntra       using {}", producer.backend_type());

    // Warmup
    for i in 0..WARMUP {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let _ = producer.send(msg);
        let _ = consumer.recv();
    }

    // Measure
    let mut latencies = Vec::with_capacity(ITERATIONS);
    for i in 0..ITERATIONS {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let start = rdtsc();
        let _ = producer.send(msg);
        let end = rdtsc();
        latencies.push(end.wrapping_sub(start));
        let _ = consumer.recv();
    }

    LatencyStats::compute("MpmcIntra", latencies, cpu_ghz, rdtsc_overhead)
}

/// Benchmark SpscShm (cross-process SPSC)
fn bench_spsc_shm(cpu_ghz: f64, rdtsc_overhead: u64) -> Option<LatencyStats> {
    let topic_name = format!("bench_spsc_shm_{}", std::process::id());

    let producer = match Topic::<TestMessage>::spsc_shm(&topic_name, true) {
        Ok(t) => t,
        Err(e) => {
            println!("  SpscShm         SKIP: {}", e);
            return None;
        }
    };
    let consumer = match Topic::<TestMessage>::spsc_shm(&topic_name, false) {
        Ok(t) => t,
        Err(e) => {
            println!("  SpscShm         SKIP consumer: {}", e);
            return None;
        }
    };

    println!("  SpscShm         using {}", producer.backend_type());

    // Warmup
    for i in 0..WARMUP {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let _ = producer.send(msg);
        let _ = consumer.recv();
    }

    // Measure
    let mut latencies = Vec::with_capacity(ITERATIONS);
    for i in 0..ITERATIONS {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let start = rdtsc();
        let _ = producer.send(msg);
        let end = rdtsc();
        latencies.push(end.wrapping_sub(start));
        let _ = consumer.recv();
    }

    Some(LatencyStats::compute("SpscShm", latencies, cpu_ghz, rdtsc_overhead))
}

/// Benchmark MpscShm (cross-process MPSC)
fn bench_mpsc_shm(cpu_ghz: f64, rdtsc_overhead: u64) -> Option<LatencyStats> {
    let topic_name = format!("bench_mpsc_shm_{}", std::process::id());

    let producer = match Topic::<TestMessage>::mpsc_shm(&topic_name, 1024, true) {
        Ok(t) => t,
        Err(e) => {
            println!("  MpscShm         SKIP: {}", e);
            return None;
        }
    };
    let consumer = match Topic::<TestMessage>::mpsc_shm(&topic_name, 1024, false) {
        Ok(t) => t,
        Err(e) => {
            println!("  MpscShm         SKIP consumer: {}", e);
            return None;
        }
    };

    println!("  MpscShm         using {}", producer.backend_type());

    // Warmup
    for i in 0..WARMUP {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let _ = producer.send(msg);
        let _ = consumer.recv();
    }

    // Measure
    let mut latencies = Vec::with_capacity(ITERATIONS);
    for i in 0..ITERATIONS {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let start = rdtsc();
        let _ = producer.send(msg);
        let end = rdtsc();
        latencies.push(end.wrapping_sub(start));
        let _ = consumer.recv();
    }

    Some(LatencyStats::compute("MpscShm", latencies, cpu_ghz, rdtsc_overhead))
}

/// Benchmark SpmcShm (cross-process SPMC)
fn bench_spmc_shm(cpu_ghz: f64, rdtsc_overhead: u64) -> Option<LatencyStats> {
    let topic_name = format!("bench_spmc_shm_{}", std::process::id());

    let producer = match Topic::<TestMessage>::spmc_shm(&topic_name, true) {
        Ok(t) => t,
        Err(e) => {
            println!("  SpmcShm         SKIP: {}", e);
            return None;
        }
    };
    let consumer = match Topic::<TestMessage>::spmc_shm(&topic_name, false) {
        Ok(t) => t,
        Err(e) => {
            println!("  SpmcShm         SKIP consumer: {}", e);
            return None;
        }
    };

    println!("  SpmcShm         using {}", producer.backend_type());

    // Warmup
    for i in 0..WARMUP {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let _ = producer.send(msg);
        let _ = consumer.recv();
    }

    // Measure
    let mut latencies = Vec::with_capacity(ITERATIONS);
    for i in 0..ITERATIONS {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let start = rdtsc();
        let _ = producer.send(msg);
        let end = rdtsc();
        latencies.push(end.wrapping_sub(start));
        let _ = consumer.recv();
    }

    Some(LatencyStats::compute("SpmcShm", latencies, cpu_ghz, rdtsc_overhead))
}

/// Benchmark MpmcShm (cross-process MPMC)
fn bench_mpmc_shm(cpu_ghz: f64, rdtsc_overhead: u64) -> Option<LatencyStats> {
    let topic_name = format!("bench_mpmc_shm_{}", std::process::id());

    // Use from_endpoint to get actual MpmcShm backend
    let topic: Topic<TestMessage> = match Topic::from_endpoint(&topic_name) {
        Ok(t) => t,
        Err(e) => {
            println!("  MpmcShm         SKIP: {}", e);
            return None;
        }
    };

    println!("  MpmcShm         using {}", topic.backend_type());

    // Warmup
    for i in 0..WARMUP {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let _ = topic.send(msg);
        let _ = topic.recv();
    }

    // Measure
    let mut latencies = Vec::with_capacity(ITERATIONS);
    for i in 0..ITERATIONS {
        let msg = TestMessage { sequence: i as u32, value: 1.0 };
        let start = rdtsc();
        let _ = topic.send(msg);
        let end = rdtsc();
        latencies.push(end.wrapping_sub(start));
        let _ = topic.recv();
    }

    Some(LatencyStats::compute("MpmcShm", latencies, cpu_ghz, rdtsc_overhead))
}

fn main() {
    println!("\n{}", "═".repeat(80));
    println!("{}", "  HORUS Backend Latency Benchmark - Zero-Overhead Verification".bold());
    println!("{}", "═".repeat(80));

    // Calibration
    println!("\n{}", "Calibration".bold().underline());
    let rdtsc_overhead = calibrate_rdtsc();
    println!("  RDTSC overhead: {} cycles", rdtsc_overhead);

    let cpu_ghz = detect_cpu_frequency();
    println!("  CPU frequency:  {:.3} GHz", cpu_ghz);
    println!("  Iterations:     {} (warmup: {})", ITERATIONS, WARMUP);

    // Backend targets (p99 nanoseconds) - Performance-first aggressive targets
    let targets = [
        ("Direct(unchecked)", 10u64),  // ~3-5ns documented target, 10ns with margin
        ("Direct(safe)", 100),          // Safe API with thread check + metrics (~50-75ns typical)
        ("SpscIntra", 100),
        ("MpscIntra", 150),
        ("SpmcIntra", 150),
        ("MpmcIntra", 200),
        ("SpscShm", 300),
        ("MpscShm", 400),
        ("SpmcShm", 400),
        ("MpmcShm", 500),
    ];

    // Run benchmarks
    println!("\n{}", "In-Process Backends".bold().underline());
    let mut results: Vec<(LatencyStats, u64)> = Vec::new();

    results.push((bench_direct_channel_unchecked(cpu_ghz, rdtsc_overhead), targets[0].1));
    results.push((bench_direct_channel(cpu_ghz, rdtsc_overhead), targets[1].1));
    results.push((bench_spsc_intra(cpu_ghz, rdtsc_overhead), targets[2].1));
    results.push((bench_mpsc_intra(cpu_ghz, rdtsc_overhead), targets[3].1));
    results.push((bench_spmc_intra(cpu_ghz, rdtsc_overhead), targets[4].1));
    results.push((bench_mpmc_intra(cpu_ghz, rdtsc_overhead), targets[5].1));

    println!("\n{}", "Shared Memory Backends".bold().underline());
    if let Some(stats) = bench_spsc_shm(cpu_ghz, rdtsc_overhead) {
        results.push((stats, targets[6].1));  // SpscShm: 300ns
    }
    if let Some(stats) = bench_mpsc_shm(cpu_ghz, rdtsc_overhead) {
        results.push((stats, targets[7].1));  // MpscShm: 400ns
    }
    if let Some(stats) = bench_spmc_shm(cpu_ghz, rdtsc_overhead) {
        results.push((stats, targets[8].1));  // SpmcShm: 400ns
    }
    if let Some(stats) = bench_mpmc_shm(cpu_ghz, rdtsc_overhead) {
        results.push((stats, targets[9].1));  // MpmcShm: 500ns
    }

    // Results summary
    println!("\n{}", "═".repeat(80));
    println!("{}", "  Results Summary".bold());
    println!("{}", "═".repeat(80));

    let mut all_pass = true;
    for (stats, target) in &results {
        stats.print(*target);
        if stats.p99_ns > *target {
            all_pass = false;
        }
    }

    println!("{}", "═".repeat(80));

    if all_pass {
        println!("\n{}", "✓ ALL BACKENDS MEET LATENCY TARGETS".green().bold());
    } else {
        println!("\n{}", "✗ SOME BACKENDS FAILED TO MEET TARGETS".red().bold());
        println!("\nNote: Targets are aggressive. Run on bare metal with:");
        println!("  - CPU governor set to 'performance'");
        println!("  - Turbo boost disabled");
        println!("  - Minimal background processes");
        std::process::exit(1);
    }

    // Detailed comparison table
    println!("\n{}", "Latency Comparison (p99 nanoseconds)".bold().underline());
    println!();
    println!("  Backend         Measured    Target    Delta");
    println!("  ─────────────────────────────────────────────");
    for (stats, target) in &results {
        let delta = stats.p99_ns as i64 - *target as i64;
        let delta_str = if delta <= 0 {
            format!("{:+}ns", delta).green()
        } else {
            format!("+{}ns", delta).red()
        };
        println!(
            "  {:15} {:>6}ns    {:>4}ns    {}",
            stats.name, stats.p99_ns, target, delta_str
        );
    }
    println!();
}
