//! Hardware Drivers for HORUS
//!
//! This module provides the foundation for building hardware drivers.
//!
//! # Current Status
//!
//! Built-in driver implementations are not currently included in the base
//! horus_library crate. Users implement their own drivers using the traits
//! provided by horus_core.
//!
//! # Driver Traits
//!
//! Driver traits are defined in `horus_core::driver`:
//!
//! - **`Driver`** - Base driver trait with lifecycle methods
//! - **`Sensor<Output=T>`** - Sensor driver trait for reading data
//! - **`Actuator<Command=T>`** - Actuator driver trait for sending commands
//!
//! # Building Custom Drivers
//!
//! ```rust,ignore
//! use horus_core::driver::{Driver, Sensor, Actuator};
//! use horus_library::messages::ImuData;
//!
//! pub struct MyImuDriver {
//!     // Driver state
//! }
//!
//! impl Driver for MyImuDriver {
//!     fn init(&mut self) -> Result<(), Box<dyn std::error::Error>> {
//!         // Initialize hardware
//!         Ok(())
//!     }
//!
//!     fn shutdown(&mut self) {
//!         // Cleanup
//!     }
//! }
//!
//! impl Sensor for MyImuDriver {
//!     type Output = ImuData;
//!
//!     fn read(&mut self) -> Option<Self::Output> {
//!         // Read from hardware
//!         None
//!     }
//! }
//! ```
//!
//! # Message Types for Drivers
//!
//! Standard message types from `horus_library::messages` for driver I/O:
//!
//! | Category | Types |
//! |----------|-------|
//! | Sensors | `ImuData`, `LaserScan`, `PointCloud`, `BatteryState` |
//! | Vision | `Image`, `CameraInfo`, `CompressedImage` |
//! | Control | `MotorCommand`, `JointState`, `ServoCommand` |
//! | I/O | `DigitalIO`, `AnalogIO`, `CanFrame`, `ModbusData`, `I2CData` |
