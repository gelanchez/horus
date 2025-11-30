# F1Tenth Adaptive Racing System

An autonomous racing controller that learns and improves lap times through exploration and adaptive learning.

## Overview

This project implements a three-phase racing approach:

1. **Exploration Phase** (Lap 1): The robot uses gap-following to navigate the track safely while recording its path
2. **Learning Phase**: After the first lap, the controller builds an optimized racing line from the exploration data
3. **Racing Phase** (Laps 2+): Pure pursuit path following on the optimized racing line, with continuous improvement

## Features

- **Gap-Following Navigation**: Reactive obstacle avoidance for safe exploration
- **Racing Line Optimization**: Smooths exploration path and calculates optimal speeds based on curvature
- **Pure Pursuit Control**: Precise path following with adaptive lookahead distance
- **Collision Avoidance**: Emergency override when obstacles are detected
- **Lap Timing & Statistics**: Tracks improvement across laps
- **Curriculum Learning**: Progress through increasingly difficult tracks

## Race Tracks

| Track | Difficulty | Description |
|-------|------------|-------------|
| Oval Sprint | 1 | Simple oval - learn basics |
| Chicane Challenge | 2 | Tight chicanes - learn braking |
| Monaco Miniature | 3 | Street circuit with hairpins |
| Silverstone Sprint | 4 | High-speed corners |
| Chaos Circuit | 5 | Random obstacles |

## Quick Start

### Terminal 1: Start the simulator
```bash
cd /path/to/HORUS

# Easy track
cargo run -p sim2d -- --robot tests/sim2d/f1tenth_race/f1tenth_robot.yaml

# Or load a specific track (when world loading is implemented)
# cargo run -p sim2d -- --world tests/sim2d/f1tenth_race/maps/track_01_oval.yaml
```

### Terminal 2: Run the racing controller
```bash
cd tests/sim2d/f1tenth_race
horus run
```

## How It Works

### Phase 1: Exploration
- Uses LIDAR to find the widest gap (safest direction)
- Records position samples every 0.2m
- Runs at conservative speed (60% of max)
- Builds understanding of track layout

### Phase 2: Learning
- Applies moving average smoothing to recorded path
- Calculates curvature at each point using Menger formula
- Determines optimal speed using: `v = sqrt(a_lateral / curvature)`
- Creates racing line with speed profile

### Phase 3: Racing
- Pure pursuit follows the racing line
- Adaptive lookahead: faster = look further ahead
- Speed from pre-computed optimal values
- Emergency collision avoidance overrides if needed

## Performance Expectations

- **Lap 1 (Exploration)**: ~50% slower than optimal
- **Lap 2**: ~20-30% improvement over Lap 1
- **Lap 3+**: Continuous refinement, ~5-10% per lap
- **Final**: Within 10-15% of theoretical optimal

## Configuration

Edit `f1tenth_robot.yaml` to tune:
- `max_speed`: Maximum velocity (default: 8.0 m/s)
- `max_steering_angle`: Steering limit (default: 0.4 rad)
- Sensor parameters (LIDAR range, resolution)

Edit `main.rs` to tune:
- `gap_threshold`: Minimum gap width to consider (default: 2.0m)
- `safety_distance`: Emergency braking distance (default: 0.5m)
- `lookahead_distance`: Pure pursuit lookahead (default: 1.0m)

## Extending

### Add New Tracks
1. Create a YAML file in `maps/` following the existing format
2. Define walls, obstacles, checkpoints, and spawn position
3. Set appropriate `optimal_lap_time` for scoring

### Improve the Controller
- Implement Model Predictive Control (MPC) for better trajectory optimization
- Add reinforcement learning for racing line refinement
- Implement opponent detection for head-to-head racing

## Files

```
f1tenth_race/
├── main.rs              # Racing controller implementation
├── horus.yaml           # HORUS project configuration
├── f1tenth_robot.yaml   # Robot physical configuration
├── README.md            # This file
└── maps/
    ├── track_01_oval.yaml
    ├── track_02_chicane.yaml
    ├── track_03_monaco.yaml
    ├── track_04_silverstone.yaml
    └── track_05_random.yaml
```

## License

Part of the HORUS robotics framework.
