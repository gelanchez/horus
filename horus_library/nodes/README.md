# HORUS Node Infrastructure

This directory provides the foundation for building HORUS nodes.

## Current Status

**Note**: Built-in node implementations are not currently included in the base horus_library crate. Users implement their own nodes using the HORUS core infrastructure.

## What's Available

### Core Infrastructure (from horus_core)

- **`Node` trait** - Base trait all nodes implement
- **`NodeInfo`** - Node metadata and identification
- **`Topic<T>`** - Type-safe pub/sub communication
- **`Scheduler`** - Real-time node execution

### Message Types (from horus_library::messages)

Standard robotics message types are provided for building nodes:

| Category | Types |
|----------|-------|
| **Geometry** | `Pose2D`, `Pose3D`, `Transform`, `Twist`, `Vector3`, `Quaternion` |
| **Sensors** | `ImuData`, `LaserScan`, `PointCloud`, `BatteryState`, `Temperature` |
| **Navigation** | `Odometry`, `Path`, `OccupancyGrid`, `Goal` |
| **Vision** | `Image`, `CameraInfo`, `CompressedImage` |
| **Control** | `MotorCommand`, `JointState`, `ServoCommand` |
| **I/O** | `DigitalIO`, `AnalogIO`, `CanFrame`, `ModbusData`, `I2CData` |

## Building Custom Nodes

Implement the `Node` trait to create your own nodes:

```rust
use horus_core::{Node, NodeInfo, Topic};
use horus_library::messages::{Twist, Odometry};

pub struct MyRobotNode {
    cmd_vel: Topic<Twist>,
    odom: Topic<Odometry>,
}

impl Node for MyRobotNode {
    fn info(&self) -> NodeInfo {
        NodeInfo::new("my_robot_node")
    }

    fn tick(&mut self) {
        // Read velocity commands
        if let Some(vel) = self.cmd_vel.read() {
            // Apply to motors...
        }

        // Publish odometry
        let odom = Odometry { /* ... */ };
        self.odom.write(&odom);
    }
}
```

## Hardware Integration

For hardware integration, use the driver traits from `horus_core::driver`:

```rust
use horus_core::driver::{Driver, Sensor, Actuator};

// Implement for your hardware
impl Sensor for MyImuDriver {
    type Output = ImuData;
    fn read(&mut self) -> Option<Self::Output> { /* ... */ }
}
```

## Testing Without Hardware

Use the `sim2d` or `sim3d` tools for testing nodes without physical hardware:

```bash
# Launch 2D simulation
horus sim2d --robot my_robot.yaml

# Launch 3D simulation
horus sim3d --robot my_robot.urdf
```

## See Also

- [HORUS Core Documentation](../../horus_core/README.md)
- [Message Types](../messages/)
- [HFrame Coordinate Transforms](../hframe/)
