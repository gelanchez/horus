//! HORUS Library Nodes
//!
//! This module provides access to HORUS node infrastructure.
//!
//! # Current Status
//!
//! Built-in node implementations are not currently included in the base horus_library
//! crate. Users implement their own nodes using the HORUS core infrastructure.
//!
//! # Available Infrastructure
//!
//! ## From horus_core
//! - `Node` trait - Base trait all nodes must implement
//! - `NodeInfo` - Node metadata and identification
//! - `Topic<T>` - Type-safe pub/sub communication
//! - `Scheduler` - Real-time node execution
//!
//! ## From horus_library::messages
//! Standard robotics message types for node communication:
//! - Geometry: `Pose2D`, `Pose3D`, `Twist`, `Transform`
//! - Sensors: `ImuData`, `LaserScan`, `PointCloud`
//! - Navigation: `Odometry`, `Path`, `OccupancyGrid`
//! - Vision: `Image`, `CameraInfo`
//! - Control: `MotorCommand`, `JointState`
//! - I/O: `DigitalIO`, `AnalogIO`, `CanFrame`
//!
//! # Building Custom Nodes
//!
//! ```rust,ignore
//! use horus_core::{Node, NodeInfo, Topic};
//! use horus_library::messages::{Twist, Odometry};
//!
//! pub struct MyRobotNode {
//!     cmd_vel: Topic<Twist>,
//!     odom: Topic<Odometry>,
//! }
//!
//! impl Node for MyRobotNode {
//!     fn info(&self) -> NodeInfo {
//!         NodeInfo::new("my_robot_node")
//!     }
//!
//!     fn tick(&mut self) {
//!         // Node logic here
//!     }
//! }
//! ```

// Re-export core HORUS types for convenience
pub use horus_core::{Node, NodeInfo, Topic};
