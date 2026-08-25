# Motion Planning Notes

## Control Mode

The `IiwaDriver` control mode is set in `scenario_data` in `scan_object.py`:

- `position_only` — send desired joint positions, KUKA's internal PD handles torques
- `torque_only` — send torques directly, you handle everything including gravity
- `position_and_torque` — send both; PD tracks position + feedforward torque added on top

Currently using `position_only`.

## Where Desired Position Comes From

Two sources depending on state:

- **Idle:** `ConstantVectorSource(default_position)` is wired to `station.GetInputPort("iiwa.position")` via the diagram
- **Moving:** `move_along_trajectory()` calls `FixValue(station_context, q_desired)` which overrides the diagram wire. `q_desired = traj.value(traj_time)` samples the TOPPRA trajectory at the current sim time.

`FixValue` wins over diagram connections when called.

## Command Rate

The Python loop advances `0.01s` at a time, so `FixValue` is called at ~100Hz. The TOPPRA trajectory is smooth and continuous but gets sampled at 100Hz — effectively a staircase approximation. This is fine because:
- Steps are tiny at slow scanning speeds
- KUKA's internal controller runs at 1kHz and interpolates between commands
- 100Hz is within the normal FRI command rate

## How Commands Reach the Hardware

```
FixValue(q_desired)
    → station.GetInputPort("iiwa.position")    (external station)
    → IiwaDriver (sim: SimIiwaDriver PD; hardware: LCM → FRI)
    → KUKA internal 1kHz PD controller
    → motors
```

Drake's `IiwaDriver` just publishes desired positions over LCM. The KUKA hardware closes the torque loop itself.

## Internal vs External Station

`IiwaHardwareStationDiagram` contains two stations:

- **External station** (`MakeHardwareStation`): runs `SimIiwaDriver` in sim or real `IiwaDriver` on hardware. This is what receives `FixValue` commands and drives the robot.
- **Internal station** (`InternalStationDiagram`): a non-simulated kinematic model used for IK, GCS, and trajectory optimization. Does not receive commands directly.

The internal station is wired to receive `position_measured` from the external station — so it mirrors where the robot actually is (after dynamics), not where it was commanded to go.

```
FixValue(q_desired)
    → external station (dynamics)
    → position_measured
    → internal station (kinematic mirror, used for planning)
```
