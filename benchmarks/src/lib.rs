//! HORUS Benchmark Suite Library
//!
//! Benchmarking utilities for the HORUS robotics framework.

#![allow(clippy::all)]
#![allow(deprecated)]
#![allow(unused_imports)]
#![allow(unused_assignments)]
#![allow(unreachable_patterns)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Standard message sizes used in robotics applications
pub const MESSAGE_SIZES: &[(&str, usize)] = &[
    ("control_command", 64),
    ("sensor_reading", 128),
    ("lidar_scan", 4096),
    ("pointcloud", 65536),
    ("camera_frame", 1_000_000),
    ("map_update", 10_000_000),
];

/// Common frequencies in robotics systems
pub const FREQUENCIES: &[(&str, u32)] = &[
    ("control_loop", 1000),
    ("planning", 100),
    ("perception", 30),
    ("lidar", 10),
    ("localization", 50),
];

/// Benchmark result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub framework: String,
    pub message_size: usize,
    pub iterations: usize,
    pub total_duration: Duration,
    pub latencies: Vec<Duration>,
    pub throughput: f64,
    pub cpu_usage: f32,
    pub memory_usage: usize,
}

impl BenchmarkResult {
    pub fn statistics(&self) -> Statistics {
        let mut latencies_ns: Vec<f64> =
            self.latencies.iter().map(|d| d.as_nanos() as f64).collect();
        latencies_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = latencies_ns.len();
        let mean = latencies_ns.iter().sum::<f64>() / len as f64;

        let median_value = if len % 2 == 0 {
            (latencies_ns[len / 2 - 1] + latencies_ns[len / 2]) / 2.0
        } else {
            latencies_ns[len / 2]
        };

        Statistics {
            mean: Duration::from_nanos(mean as u64),
            median: Duration::from_nanos(median_value as u64),
            p50: Duration::from_nanos(median_value as u64),
            p95: Duration::from_nanos(calculate_percentile(&latencies_ns, 95.0) as u64),
            p99: Duration::from_nanos(calculate_percentile(&latencies_ns, 99.0) as u64),
            min: Duration::from_nanos(latencies_ns[0] as u64),
            max: Duration::from_nanos(latencies_ns[len - 1] as u64),
            std_dev: calculate_std_dev(&latencies_ns, mean),
        }
    }
}

/// Statistical metrics for benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Statistics {
    pub mean: Duration,
    pub median: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub min: Duration,
    pub max: Duration,
    pub std_dev: f64,
}

fn calculate_std_dev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let variance =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

fn calculate_percentile(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let n = values.len();
    if n == 1 {
        return values[0];
    }

    let h = (n - 1) as f64 * (percentile / 100.0);
    let h_floor = h.floor() as usize;
    let h_ceil = h.ceil() as usize;

    if h_floor >= n - 1 {
        return values[n - 1];
    }

    let lower = values[h_floor];
    let upper = values[h_ceil];
    let weight = h - h_floor as f64;

    lower + weight * (upper - lower)
}

/// Benchmark message for testing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkMessage {
    pub id: u64,
    pub timestamp: u64,
    #[serde(with = "serde_bytes")]
    pub payload: Vec<u8>,
}

impl BenchmarkMessage {
    pub fn new(id: u64, size: usize) -> Self {
        Self {
            id,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            payload: vec![0u8; size],
        }
    }
}

/// CPU governor management for consistent benchmarks
pub fn set_performance_governor() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("sudo")
            .args(["cpupower", "frequency-set", "-g", "performance"])
            .output()?;
    }
    Ok(())
}

/// Warmup iterations to stabilize cache and branch prediction
pub fn warmup<F>(iterations: usize, mut f: F)
where
    F: FnMut(),
{
    for _ in 0..iterations {
        f();
        std::hint::black_box(());
    }
}
