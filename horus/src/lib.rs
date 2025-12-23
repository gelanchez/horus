//! # HORUS - Hybrid Optimized Robotics Unified System
//!
//! HORUS provides a comprehensive framework for building robotics applications in Rust,
//! with a focus on performance, safety, and developer experience.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use horus::prelude::*;
//! use horus::library::messages::cmd_vel::CmdVel;
//!
//! pub struct MyNode {
//!     publisher: Hub<CmdVel>,
//! }
//!
//! impl Node for MyNode {
//!     fn name(&self) -> &'static str { "MyNode" }
//!
//!     fn tick(&mut self, _ctx: Option<&mut NodeInfo>) {
//!         // Node logic here
//!     }
//! }
//! ```
//!
//! ## Features
//!
//! - **Zero-copy IPC** with multiple backend support
//! - **Type-safe message passing**
//! - **Built-in monitoring and debugging**
//! - **Standard library of components**
//! - **Comprehensive tooling**

// Re-export core components (avoiding conflicts)
pub use horus_core::{self, *};

// Re-export macros
#[cfg(feature = "macros")]
pub use horus_macros::*;

// Re-export standard library with alias
pub use horus_library as library;

// Re-export serde at crate root for macro-generated code
pub use serde;

/// The HORUS prelude - everything you need to get started
///
/// This includes all core types, advanced features, and commonly used components.
/// Just add `use horus::prelude::*;` to get started.
pub mod prelude {
    // ============================================
    // Core Node Types
    // ============================================
    pub use horus_core::core::node::NodeConfig;
    pub use horus_core::core::{LogSummary, Node, NodeInfo, NodeInfoExt, NodeState};

    // ============================================
    // Communication (IPC)
    // ============================================
    pub use horus_core::communication::{Hub, Link};

    // ============================================
    // Scheduling
    // ============================================
    pub use horus_core::scheduling::{ExecutionMode, RobotPreset, Scheduler, SchedulerConfig};

    // ============================================
    // Safety & Fault Tolerance
    // ============================================
    pub use horus_core::scheduling::{
        // BlackBox flight recorder
        BlackBox,
        BlackBoxEvent,
        // Checkpointing
        Checkpoint,
        CheckpointManager,
        // Circuit breaker
        CircuitBreaker,
        CircuitState,
        // Redundancy/TMR voting
        RedundancyManager,
        // Safety monitoring
        SafetyMonitor,
        SafetyState,
        SafetyStats,
        VoteResult,
        VotingStrategy,
        WCETEnforcer,
        Watchdog,
    };

    // ============================================
    // Advanced Executors
    // ============================================
    pub use horus_core::scheduling::{
        AsyncIOExecutor, AsyncResult, BackgroundExecutor, ParallelExecutor,
    };

    // ============================================
    // Profiling & Intelligence
    // ============================================
    pub use horus_core::scheduling::{
        ExecutionTier, NodeProfile, NodeTier, OfflineProfiler, ProfileData, ProfileError,
    };

    // ============================================
    // Record/Replay
    // ============================================
    pub use horus_core::scheduling::{
        NodeRecorder, NodeRecording, NodeReplayer, NodeTickSnapshot, RecordingConfig,
        RecordingManager, SchedulerRecording,
    };

    // ============================================
    // Telemetry
    // ============================================
    pub use horus_core::scheduling::{TelemetryEndpoint, TelemetryManager};

    // ============================================
    // Runtime (OS-level features)
    // ============================================
    pub use horus_core::scheduling::{
        apply_rt_optimizations, get_core_count, get_max_rt_priority, get_numa_node_count,
        lock_all_memory, set_realtime_priority, set_thread_affinity,
    };

    // ============================================
    // JIT Compilation
    // ============================================
    pub use horus_core::scheduling::JITCompiler;

    // ============================================
    // Memory & Tensors
    // ============================================
    pub use horus_core::memory::{TensorHandle, TensorPool, TensorPoolConfig};

    // CUDA support (requires "cuda" feature)
    #[cfg(feature = "cuda")]
    pub use horus_core::memory::{cuda_available, cuda_device_count};

    // ============================================
    // HFrame Transform System
    // ============================================
    pub use horus_library::hframe::{timestamp_now, HFrame, HFrameConfig, Transform};

    // ============================================
    // Message Types (ALL from horus_library)
    // ============================================
    pub use horus_library::messages::tensor::{HorusTensor, TensorDevice, TensorDtype};
    // Re-export all message types from horus_library
    pub use horus_library::messages::*;

    // ============================================
    // Algorithms
    // ============================================
    pub use horus_library::algorithms::{
        astar::AStar, differential_drive::DifferentialDrive, ekf::EKF, kalman_filter::KalmanFilter,
        occupancy_grid::OccupancyGrid, pid::PID, pure_pursuit::PurePursuit, rrt::RRT,
    };

    // ============================================
    // Error Types (clean aliases - no Horus prefix)
    // ============================================
    pub use horus_core::error::{Error, Result};
    // Backward compatibility - users can still use HorusError/HorusResult
    pub use horus_core::error::{HorusError, HorusResult};

    // ============================================
    // Common Std Types
    // ============================================
    pub use std::sync::{Arc, Mutex};
    pub use std::time::{Duration, Instant};

    // ============================================
    // Macros
    // ============================================
    #[cfg(feature = "macros")]
    pub use horus_macros::*;

    // ============================================
    // Common Traits
    // ============================================
    pub use serde::{Deserialize, Serialize};

    // Re-export anyhow for error handling
    pub use anyhow::{anyhow, bail, ensure, Context, Result as AnyResult};

    // ============================================
    // Built-in Nodes (standard-nodes feature)
    // ============================================
    pub use horus_library::nodes::{
        DifferentialDriveNode, EmergencyStopNode, JoystickInputNode, KeyboardInputNode,
        LocalizationNode, PathPlannerNode, PidControllerNode,
    };

    // ============================================
    // Hardware-specific Nodes (require feature flags)
    // ============================================
    #[cfg(feature = "serial-hardware")]
    pub use horus_library::nodes::SerialNode;

    #[cfg(feature = "gpio-hardware")]
    pub use horus_library::nodes::{DigitalIONode, EncoderNode, ServoControllerNode};

    #[cfg(any(feature = "bno055-imu", feature = "mpu6050-imu"))]
    pub use horus_library::nodes::ImuNode;

    #[cfg(feature = "rplidar")]
    pub use horus_library::nodes::LidarNode;

    #[cfg(feature = "modbus-hardware")]
    pub use horus_library::nodes::ModbusNode;
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get HORUS version
pub fn version() -> &'static str {
    VERSION
}
