# HORUS Robotics - Python Bindings

[![PyPI](https://img.shields.io/pypi/v/horus-robotics)](https://pypi.org/project/horus-robotics/)
[![Python](https://img.shields.io/pypi/pyversions/horus-robotics)](https://pypi.org/project/horus-robotics/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://github.com/softmata/horus/blob/main/LICENSE)

**Ultra-low latency robotics framework with Python bindings** - 100x faster than ROS2.

HORUS is a production-grade robotics framework built in Rust, delivering **87ns IPC latency** and **12M+ messages/second** throughput. This package provides native Python bindings via PyO3.

## Installation

```bash
pip install horus-robotics
```

## Quick Start

```python
import horus

# Create a publisher
pub = horus.Publisher("sensor.data")

# Create a subscriber
sub = horus.Subscriber("sensor.data")

# Publish data
pub.publish({"temperature": 25.5, "humidity": 60})

# Receive data
data = sub.receive()
print(data)
```

## Features

- **Ultra-Low Latency**: 87ns IPC latency (100x faster than ROS2)
- **High Throughput**: 12M+ messages/second
- **Zero-Copy**: Shared memory transport for maximum performance
- **Type-Safe**: Strong typing with Pydantic-like validation
- **Cross-Language**: Seamless interop with Rust and C++ nodes
- **Real-Time Ready**: Priority-based scheduling with deadline guarantees

## Core Components

### Publisher/Subscriber (Hub)

Multi-producer multi-consumer pub/sub with ~481ns round-trip latency:

```python
import horus

# Publisher
pub = horus.Publisher("robot.commands")
pub.publish({"velocity": [1.0, 0.0, 0.0]})

# Subscriber
sub = horus.Subscriber("robot.commands")
msg = sub.receive(timeout_ms=100)
```

### Point-to-Point (Link)

Single-producer single-consumer for lowest latency (~248ns):

```python
sender = horus.LinkSender("fast.channel")
receiver = horus.LinkReceiver("fast.channel")

sender.send(data)
data = receiver.recv()
```

### Nodes

Create custom processing nodes:

```python
import horus

class SensorNode(horus.Node):
    def __init__(self):
        super().__init__("sensor_node")
        self.pub = horus.Publisher("sensor.data")

    def tick(self):
        reading = self.read_sensor()
        self.pub.publish(reading)

# Run the node
node = SensorNode()
horus.run(node, rate_hz=100)
```

## Message Types

HORUS provides standard robotics message types:

```python
from horus.messages import (
    Twist,           # Velocity commands
    Pose,            # Position + orientation
    Image,           # Camera images
    PointCloud,      # 3D point clouds
    JointState,      # Robot joint positions
    LaserScan,       # LIDAR data
    Imu,             # IMU readings
    NavSatFix,       # GPS coordinates
    WrenchStamped,   # Force/torque sensor
)
```

## Performance Comparison

| Framework | IPC Latency | Throughput |
|-----------|-------------|------------|
| **HORUS** | **87ns** | **12M+ msg/s** |
| ROS2 (FastDDS) | 50-100us | ~100K msg/s |
| ZeroMQ | 10-30us | ~1M msg/s |

## Documentation

- **Full Documentation**: [docs.horus-registry.dev](https://docs.horus-registry.dev)
- **Getting Started**: [Installation Guide](https://docs.horus-registry.dev/installation)
- **API Reference**: [Python API](https://docs.horus-registry.dev/python-api)
- **Examples**: [GitHub Examples](https://github.com/softmata/horus/tree/main/examples)

## Requirements

- Python 3.9+
- Linux (x86_64)
- glibc 2.39+

## Related Packages

- `horus` - Full HORUS framework (Rust CLI + runtime)
- `horus-library` - Standard message types and nodes

## License

Apache-2.0 - see [LICENSE](https://github.com/softmata/horus/blob/main/LICENSE)

## Links

- **GitHub**: [github.com/softmata/horus](https://github.com/softmata/horus)
- **Discord**: [Join Community](https://discord.gg/hEZC3ev2Nf)
- **Documentation**: [docs.horus-registry.dev](https://docs.horus-registry.dev)
