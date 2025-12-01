/// Hardware Sensors Demo
///
/// This example demonstrates how to use hardware sensors with HORUS.
///
/// Sensors shown:
/// - NMEA GPS via Serial
/// - RPLidar A2 via Serial
/// - Force/Torque sensor via NetFT (with `netft` feature)
///
/// Build with hardware support:
/// ```bash
/// cargo run --example hardware_sensors_demo -p horus_library --features all-sensors
/// ```
///
/// Build for simulation (no hardware):
/// ```bash
/// cargo run --example hardware_sensors_demo -p horus_library --features all-sensors
/// ```
use horus_core::Scheduler;
use horus_library::nodes::gps::{GpsBackend, GpsNode};
use horus_library::nodes::lidar::{LidarBackend, LidarNode};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== HORUS Hardware Sensors Demo ===\n");

    // Check command line arguments for mode
    let args: Vec<String> = env::args().collect();
    let use_hardware = args.iter().any(|arg| arg == "--hardware");

    if use_hardware {
        println!("Running with REAL HARDWARE");
        println!("Make sure sensors are connected!\n");
    } else {
        println!("Running in SIMULATION MODE");
        println!("Use --hardware flag to connect to real sensors\n");
    }

    let mut scheduler = Scheduler::new();

    // === GPS Setup ===
    println!("Setting up GPS...");
    let gps = if use_hardware {
        println!("  -> NMEA GPS on /dev/ttyUSB0 @ 9600 baud");
        let mut node = GpsNode::new_with_backend("gps.fix", GpsBackend::NmeaSerial)?;
        node.set_serial_config("/dev/ttyUSB0", 9600);
        node.set_update_rate(1.0); // 1 Hz
        node.set_min_satellites(4);
        node
    } else {
        println!("  -> Simulation mode (San Francisco: 37.7749N, 122.4194W)");
        let mut node = GpsNode::new_with_topic("gps.fix")?;
        node.set_simulation_position(37.7749, -122.4194, 10.0);
        node
    };
    scheduler.add(Box::new(gps), 0, None); // High priority

    // === LiDAR Setup ===
    println!("Setting up LiDAR...");
    let lidar = if use_hardware {
        println!("  -> RPLidar A2 on /dev/ttyUSB1");
        println!("  WARNING: Motor will start spinning!");
        let mut node = LidarNode::new_with_backend("scan", LidarBackend::RplidarA2)?;
        node.set_serial_port("/dev/ttyUSB1");
        node.set_scan_frequency(10.0); // 10 Hz
        node
    } else {
        println!("  -> Simulation mode (synthetic obstacles)");
        LidarNode::new_with_topic("scan")?
    };
    scheduler.add(Box::new(lidar), 1, None); // Lower priority

    println!("\nSensors configured!");
    println!("\nPublishing to topics:");
    println!("  - gps.fix    : GPS position and accuracy");
    println!("  - scan       : LiDAR point cloud");
    println!("\nNote: IMU support requires 'mpu6050-imu' or 'bno055-imu' features");
    println!("      (currently disabled due to embedded-hal version conflicts)");
    println!("\nStarting scheduler...\n");
    println!("Press Ctrl+C to stop.\n");

    // Run the system (scheduler handles SIGTERM/SIGINT)
    scheduler.run()?;

    Ok(())
}
