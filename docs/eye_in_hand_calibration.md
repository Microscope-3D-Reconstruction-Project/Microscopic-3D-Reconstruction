# Eye-in-Hand Calibration — Best Practices

Eye-in-hand = camera is **mounted on the robot end-effector** and moves with it.  
The goal is to find **T_camera_to_flange**: the fixed transform from the camera frame to the robot's tool flange frame.

---

## Prerequisites

- Camera **intrinsics must already be calibrated** (see `calibration_image_collection.md`)
- Robot **forward kinematics must be accurate** — bad joint encoders or bad URDF = bad result
- The calibration target must be **fixed in the world** (does not move during collection)

---

## Target Placement

- Place the target so it is **visible from many robot configurations**, not just one region
- Avoid placing the target directly below or directly in front — you want angular variety
- Target should be at a **representative working distance** (similar to actual use)

---

## Pose Variety — The Most Important Part

You need diverse end-effector poses while keeping the target visible.

| What to Vary | Why |
|---|---|
| **Approach angle** (different joint configs, same TCP) | Separates camera-to-flange from flange-to-world |
| **Distance to target** (near / mid / far) | Constrains translation component |
| **In-plane rotation of end-effector** | Constrains rotational component of T_camera_to_flange |
| **Target position in the image** (corners, edges, center) | Distortion is position-dependent — center-only datasets bias the extrinsic solve |

**Minimum:** ~15–20 pose pairs  
**Ideal:** 30–50 poses with strong angular diversity

---

## Data to Record at Each Pose

For every image you take, record simultaneously:

1. The **robot joint angles** (or the FK-computed flange pose in world frame)
2. The **detected target pose** in the camera frame (from PnP / corner detection)

These must be **time-synchronized** — capture both at the same instant, with robot fully stopped.

---

## Robot Must Be Fully Stopped

- Command the robot to a pose, **wait for it to settle**, then capture
- Even 1–2 mm of residual motion causes large calibration errors
- If using FRI / torque control, verify the commanded and measured joint angles are converged

---

## Common Failure Modes

| Problem | Symptom | Fix |
|---|---|---|
| All poses from same direction | Large translation error in T_camera_to_flange | Add poses with different approach angles |
| Target barely moves in image | Poorly constrained rotation | Move robot so target covers full image |
| Robot not settled at capture | High residual error, inconsistent results | Add dwell time before capture |
| Bad intrinsics | Systematic reprojection pattern | Redo intrinsic calibration first |
| Target moved mid-session | Sudden jump in reprojection error | Restart; fix target more firmly |

---

## Solving the Calibration

Use an **AX = XB** solver (e.g., Tsai, Park-Martin, or Daniilidis).

- Provide **all pose pairs at once** — don't solve incrementally
- Use a solver that handles **multiple rotations**, not just one pair
- Report both **rotation error** (deg) and **translation error** (mm) on held-out pairs

---

## Validation

- **Reprojection test**: use the calibrated T_camera_to_flange to project the target into held-out images — error should be < 1–2 px
- **Physical sanity check**: transform a known world point into the camera frame and verify it looks correct visually
- **Repeatability test**: command the robot to the same pose multiple times; the projected target should land in the same image location each time
- If translation error is high but rotation is fine: you need more distance variation in your poses
- If rotation error is high: you need more in-plane end-effector rotation in your poses

---

## Recalibration Triggers

Redo the calibration if:

- The camera or mount is physically disturbed or tightened/loosened
- You switch to a different end-effector or tool flange
- The robot is re-zeroed or re-mastered
- Reprojection error on a known fixture suddenly increases
