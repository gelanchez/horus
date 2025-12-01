//! Integration test suite for sim3d
//!
//! This module contains comprehensive integration tests covering:
//! - Multi-system interactions
//! - Performance benchmarks
//! - Stress tests
//! - Determinism validation
//! - Joint validation
//! - Sensor accuracy

#![allow(dead_code)]
#![allow(unused_imports)]

pub mod benchmarks;
pub mod determinism;
pub mod joint_validation;
pub mod manipulation;
pub mod multi_robot;
pub mod navigation;
pub mod sensor_accuracy;
pub mod sensors;
pub mod stress;
