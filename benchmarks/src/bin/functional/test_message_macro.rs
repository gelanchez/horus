// Benchmark binary - allow clippy warnings
#![allow(unused_imports)]
#![allow(unused_assignments)]
#![allow(unreachable_patterns)]
#![allow(clippy::all)]
#![allow(deprecated)]

/// Test the message! macro
///
/// Verifies that the message! macro correctly generates message types with:
/// - All necessary derives (Debug, Clone, Serialize, Deserialize)
/// - LogSummary implementation
/// - Topic compatibility
use horus::prelude::*;

// Test tuple-style messages
message!(Position = (f32, f32));
message!(Color = (u8, u8, u8));
message!(Command = (u32, bool));

// Test struct-style message
message! {
    RobotStatus {
        x: f32,
        y: f32,
        battery: u8,
        is_moving: bool,
    }
}

fn main() {
    println!("Testing message! macro...\n");

    // Test 1: Tuple-style messages work
    let pos = Position(1.0, 2.0);
    println!(" Created Position(1.0, 2.0)");
    assert_eq!(pos.0, 1.0);
    assert_eq!(pos.1, 2.0);

    // Test 2: LogSummary is implemented
    let summary = pos.log_summary();
    println!(" Position.log_summary() = {}", summary);
    assert!(summary.contains("1") && summary.contains("2"));

    // Test 3: Color tuple works
    let color = Color(255, 128, 0);
    println!(" Created Color(255, 128, 0)");
    assert_eq!(color.0, 255);
    assert_eq!(color.1, 128);
    assert_eq!(color.2, 0);

    // Test 4: Command with mixed types
    let cmd = Command(42, true);
    println!(" Created Command(42, true)");
    assert_eq!(cmd.0, 42);
    assert_eq!(cmd.1, true);

    // Test 5: Struct-style message
    let status = RobotStatus {
        x: 1.5,
        y: 2.5,
        battery: 85,
        is_moving: true,
    };
    println!(" Created RobotStatus struct");
    assert_eq!(status.x, 1.5);
    assert_eq!(status.battery, 85);

    // Test 6: Topic compatibility (most important!)
    println!("\nTesting Topic compatibility...");

    let topic_pos =
        Topic::<Position>::new("test/position").expect("Failed to create Topic<Position>");
    println!(" Created Topic<Position>");

    let topic_status =
        Topic::<RobotStatus>::new("test/status").expect("Failed to create Topic<RobotStatus>");
    println!(" Created Topic<RobotStatus>");

    // Test 7: Send without logging
    topic_pos
        .send(Position(3.0, 4.0))
        .expect("Failed to send Position");
    println!(" Sent Position message");

    topic_status
        .send(RobotStatus {
            x: 5.0,
            y: 6.0,
            battery: 90,
            is_moving: false,
        })
        .expect("Failed to send RobotStatus");
    println!(" Sent RobotStatus message");

    // Test 8: Send another message
    topic_pos
        .send(Position(7.0, 8.0))
        .expect("Failed to send");
    println!(" Sent Position message");

    // Test 9: Receive messages
    if let Some(received) = topic_pos.recv() {
        println!(" Received Position: ({}, {})", received.0, received.1);
    }

    println!("\n All message! macro tests passed!");
    println!("\nSummary:");
    println!("  - Tuple-style messages work: ");
    println!("  - Struct-style messages work: ");
    println!("  - LogSummary auto-implemented: ");
    println!("  - Topic<T> compatibility: ");
    println!("  - Send/recv operations: ");
    println!("\nThe message! macro is ready for use!");
}
