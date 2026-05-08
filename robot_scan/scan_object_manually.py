"""
robot_scan/scan_object_manually.py

Moves the robot to 5 predefined hemisphere viewpoints. At each viewpoint:
  1. Robot moves to the scan point.
  2. User jogs (camera frame) until the object is centred on the crosshair.
  3. User clicks "Move to Next" — the robot scans along the optical axis from
     the jogged position, capturing frames.
  4. Robot automatically returns to the original scan-point pose before moving
     to the next viewpoint.

The crosshair is drawn at (cx, cy) from calibrated intrinsics.

Usage:
    python robot_scan/scan_object_manually.py --use_hardware
"""

import argparse
import queue
import threading

from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np

from demo_config import (
    CAMERA_SOURCE,
    DEFAULT_POSITION,
    ELBOW_ANGLE,
    HEMISPHERE_CENTER,
    HEMISPHERE_RADIUS,
    R_AXIS,
    T_CAM_TO_TIP,
    V_AXIS,
    get_config,
)
from drake.lcmt_iiwa_status import lcmt_iiwa_status
from manipulation.station import LoadScenario
from pydrake.all import (
    AddFrameTriadIllustration,
    ApplySimulatorConfig,
    ConstantVectorSource,
    DiagramBuilder,
    MeshcatVisualizer,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Simulator,
)
from pydrake.lcm import DrakeLcm
from termcolor import colored

from iiwa_setup.iiwa import IiwaHardwareStationDiagram
from iiwa_setup.util.visualizations import draw_triad
from utils.jog import try_jog_tip
from utils.kuka_geo_kin import KinematicsSolver
from utils.planning import (
    compute_hemisphere_traj_async,
    compute_optical_axis_traj_async,
    generate_hemisphere_waypoints,
    move_along_trajectory,
    plot_trajectory_in_meshcat,
)
from utils.plotting import plot_hemisphere_waypoints
from utils.RRT import plot_rrt_raw_path_in_meshcat
from utils.RRTStar import plan_rrt_star_async
from utils.safety import filter_ik_solutions
from utils.sew_stereo import (
    compute_psi_from_matrices,
    compute_sew_and_ref_matrices,
    get_sew_joint_positions,
)


class State(Enum):
    WAITING_TO_GO_TO_START = auto()
    COMPUTING_MOVE_TO_START = auto()
    MOVING_TO_START = auto()
    WAITING_FOR_NEXT_SCAN = auto()
    COMPUTING_IKS = auto()
    MOVING_ALONG_HEMISPHERE = auto()
    PLANNING_RRT_FALLBACK = auto()
    COMPUTING_RRT_FALLBACK = auto()
    MOVING_ALONG_RRT = auto()
    AWAITING_JOG_CONFIRM = auto()
    COMPUTING_OPTICAL_TRAJ = auto()
    WAITING_BEFORE_OPTICAL = auto()
    MOVING_DOWN_OPTICAL_AXIS = auto()
    RETURNING_AFTER_OPTICAL = auto()
    COMPUTING_RESET = auto()
    RESETTING = auto()
    DONE = auto()


def main(
    use_hardware: bool,
    no_cam: bool = False,
    start_idx: int = 0,
    live_view: bool = True,
    camera_source: int = CAMERA_SOURCE,
    skip_opt: bool = False,
    num_scan_points: int = 50,
    coverage: float = 0.5,
) -> None:
    cfg = get_config(use_hardware)
    speed_factor = cfg["speed_factor"]
    max_joint_velocities = cfg["max_joint_velocities"]
    vel_limits = cfg["vel_limits"]
    acc_limits = cfg["acc_limits"]

    elbow_angle = ELBOW_ANGLE
    default_position = DEFAULT_POSITION
    r = R_AXIS
    v = V_AXIS
    T_cam_to_tip = T_CAM_TO_TIP

    # Optical scan parameters
    distance_along_optical_axis = 0.035
    num_pictures = 30

    print(colored("=" * 60, "cyan"))
    print(colored("  scan_object_manually — configuration", "cyan"))
    print(colored("=" * 60, "cyan"))
    print(colored("  Args:", "white"))
    print(f"    use_hardware  : {use_hardware}")
    print(f"    no_cam        : {no_cam}")
    print(f"    start_idx     : {start_idx}")
    print(f"    live_view     : {live_view}")
    print(f"    camera_source : {camera_source}")
    print(colored("  Hemisphere:", "white"))
    print(f"    center  : {HEMISPHERE_CENTER}")
    print(f"    radius  : {HEMISPHERE_RADIUS}")
    print(f"    coverage: {coverage}")
    print(colored("  Optical scan:", "white"))
    print(f"    distance : {distance_along_optical_axis} m")
    print(f"    pictures : {num_pictures}")
    print(colored("  Config:", "white"))
    print(f"    speed_factor         : {speed_factor}")
    print(f"    max_joint_vel (deg/s): {np.rad2deg(max_joint_velocities).round(2)}")
    print(f"    vel_limits  (rad/s)  : {vel_limits.round(3)}")
    print(f"    acc_limits  (rad/s²) : {acc_limits.round(3)}")
    print(colored("=" * 60, "cyan"))

    scenario_data = """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa14_microscope.dmd.yaml
    plant_config:
        time_step: 0.005
        contact_model: "hydroelastic_with_fallback"
        discrete_contact_approximation: "sap"
    model_drivers:
        iiwa: !IiwaDriver
            lcm_bus: "default"
            control_mode: position_only
    lcm_buses:
        default:
            lcm_url: ""
    """

    # ==================================================================
    # Parameters
    # ==================================================================
    hemisphere_pos = np.asarray(HEMISPHERE_CENTER, dtype=float)
    hemisphere_radius = HEMISPHERE_RADIUS
    hemisphere_axis = np.array([-1.0, 0.0, 0.0])

    # ==================================================================
    # Outputs setup
    # ==================================================================
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    scans_base = (
        Path(__file__).parent.parent / "microscope-data" / "manual_scans" / date_str
    )
    scans_base.mkdir(parents=True, exist_ok=True)
    print(colored(f"✓ Session outputs → {scans_base}", "cyan"))

    # ==================================================================
    # Waypoint generation
    # ==================================================================
    hemisphere_waypoints = generate_hemisphere_waypoints(
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
        coverage=coverage,
        num_scan_points=num_scan_points,
    )
    labels = [str(i) for i in range(len(hemisphere_waypoints))]
    print(colored(f"\n{len(hemisphere_waypoints)} scan waypoints:", "cyan"))
    for i, wp in enumerate(hemisphere_waypoints):
        print(f"  [{i}]: {np.round(wp.translation(), 4)}")

    plot_hemisphere_waypoints(
        hemisphere_waypoints,
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
        output_path=outputs_dir / "hemisphere_waypoints.png",
        visualize=True,
    )

    # ==================================================================
    # Read initial hardware position via LCM before building diagram
    # ==================================================================
    if use_hardware:
        print(colored("Waiting for IIWA_STATUS from hardware...", "cyan"))
        _lc = DrakeLcm()
        _q_hardware = [None]

        def _iiwa_handler(data: bytes) -> None:
            msg = lcmt_iiwa_status.decode(data)
            _q_hardware[0] = np.array(msg.joint_position_measured)

        _lc.Subscribe("IIWA_STATUS", _iiwa_handler)
        while _q_hardware[0] is None:
            _lc.HandleSubscriptions(100)
        initial_q = _q_hardware[0]
        print(
            colored(
                f"✓ Hardware position: {np.rad2deg(initial_q).round(1)} deg", "cyan"
            )
        )
    else:
        initial_q = default_position

    # ==================================================================
    # Diagram setup
    # ==================================================================
    builder = DiagramBuilder()
    scenario = LoadScenario(data=scenario_data)

    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            hemisphere_pos=hemisphere_pos,
            hemisphere_radius=hemisphere_radius,
            use_hardware=use_hardware,
        ),
    )

    kinematics_solver = KinematicsSolver(station, r, v)

    dummy = builder.AddSystem(ConstantVectorSource(initial_q))
    builder.Connect(dummy.get_output_port(), station.GetInputPort("iiwa.position"))

    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    camera_frame = station.get_internal_plant().GetFrameByName("camera_link")
    AddFrameTriadIllustration(
        scene_graph=station.internal_station.get_scene_graph(),
        plant=station.get_internal_plant(),
        frame=camera_frame,
        length=0.1,
        radius=0.002,
        name="camera_link",
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    meshcat = station.internal_meshcat

    # ==================================================================
    # Buttons + sliders
    # ==================================================================
    meshcat.AddButton("Stop Simulation")
    meshcat.AddButton("Move to Next")
    meshcat.AddButton("Reset")

    JOG_BUTTONS = {
        "Jog Right (camera frame)": np.array([1.0, 0.0, 0.0]),
        "Jog Left (camera frame)": np.array([-1.0, 0.0, 0.0]),
        "Jog Down (camera frame)": np.array([0.0, 1.0, 0.0]),
        "Jog Up (camera frame)": np.array([0.0, -1.0, 0.0]),
        "Jog Forward (camera frame)": np.array([0.0, 0.0, 1.0]),
        "Jog Backward (camera frame)": np.array([0.0, 0.0, -1.0]),
    }
    for name in JOG_BUTTONS:
        meshcat.AddButton(name)

    # Keyboard shortcuts active when the OpenCV window is focused:
    #   w/s  → up/down   (camera-frame y)
    #   a/d  → left/right (camera-frame x)
    #   q/e  → backward/forward (camera-frame z)
    #   Enter → confirm (same as "Move to Next")
    KEY_JOG_MAP = {
        ord("d"): np.array([1.0, 0.0, 0.0]),
        ord("a"): np.array([-1.0, 0.0, 0.0]),
        ord("s"): np.array([0.0, 1.0, 0.0]),
        ord("w"): np.array([0.0, -1.0, 0.0]),
        ord("e"): np.array([0.0, 0.0, 1.0]),
        ord("q"): np.array([0.0, 0.0, -1.0]),
    }
    KEY_CONFIRM = 13  # Enter
    print(
        colored(
            "  Keyboard (OpenCV window): w/s=up/down  a/d=left/right  q/e=back/fwd  Enter=confirm",
            "cyan",
        )
    )

    joint_lower_limits = station.get_internal_plant().GetPositionLowerLimits()
    joint_upper_limits = station.get_internal_plant().GetPositionUpperLimits()
    for i in range(7):
        meshcat.AddSlider(
            f"Joint {i+1} (deg)",
            np.rad2deg(joint_lower_limits[i]),
            np.rad2deg(joint_upper_limits[i]),
            0.1,
            0,
        )
    meshcat.AddSlider("Current PSI (deg)", -180, 180, 0.1, 0)

    for i, wp in enumerate(hemisphere_waypoints):
        draw_triad(
            meshcat,
            f"hemisphere_waypoint_{i}",
            wp @ T_cam_to_tip,
            length=0.02,
            radius=0.001,
            opacity=0.5,
        )

    # ==================================================================
    # Pre-compute IK for all 5 waypoints
    # ==================================================================
    n = len(hemisphere_waypoints)
    q_array = np.full((n, 7), np.nan)
    failed_indices = []
    q_prev = default_position.copy()

    print(colored(f"\nPre-computing IK for {n} waypoints...", "cyan"))
    for i, wp in enumerate(hemisphere_waypoints):
        target_rot = wp.rotation().matrix()
        target_pos = wp.translation()

        Q = kinematics_solver.IK_for_microscope(target_rot, target_pos, psi=elbow_angle)
        Q = filter_ik_solutions(
            station, Q, target_rot, target_pos, joint_lower_limits, joint_upper_limits
        )

        if Q.shape[0] == 0:
            print(colored(f"  [{i}] FAIL: no valid IK solutions", "yellow"))
            failed_indices.append(i)
            continue

        q_des = kinematics_solver.find_closest_solution(Q, q_prev)
        q_array[i] = q_des
        q_prev = q_des
        print(f"  [{i}] {labels[i]}: {np.rad2deg(q_des).round(2)} deg")

    n_valid = int(np.sum(~np.isnan(q_array).any(axis=1)))
    print(
        colored(
            f"\nPre-computation done: {n_valid}/{n} valid, {len(failed_indices)} failed.",
            "cyan",
        )
    )
    if failed_indices:
        print(colored(f"  Failed indices: {failed_indices}", "yellow"))

    np.savetxt(outputs_dir / "hemisphere_q_solutions.csv", q_array, delimiter=",")

    # ==================================================================
    # Camera setup
    # ==================================================================
    camera = None
    _latest_frame = None
    _latest_frame_lock = None
    _capture_stop = None
    _frame_queue = None
    _capture_thread = None
    _writer_thread = None

    if not no_cam:
        camera = cv2.VideoCapture(camera_source)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not camera.isOpened():
            print(
                colored(
                    f"⚠ Could not open camera device {camera_source} – live view unavailable",
                    "yellow",
                )
            )

        _latest_frame_lock = threading.Lock()
        _capture_stop = threading.Event()
        _frame_queue = queue.Queue(maxsize=0)

        def _capture_loop():
            nonlocal _latest_frame
            while not _capture_stop.is_set():
                ret, frame = camera.read()
                if ret:
                    with _latest_frame_lock:
                        _latest_frame = frame

        def _writer_loop():
            while True:
                item = _frame_queue.get()
                if item is None:
                    _frame_queue.task_done()
                    break
                frame_data, path_str = item
                cv2.imwrite(path_str, frame_data)
                _frame_queue.task_done()

        _capture_thread = threading.Thread(target=_capture_loop, daemon=True)
        _writer_thread = threading.Thread(target=_writer_loop, daemon=True)
        _capture_thread.start()
        _writer_thread.start()
        print(colored("✓ Camera threads started", "cyan"))
    else:
        print(colored("✓ Camera disabled via --no_cam", "yellow"))

    if live_view and no_cam:
        print(colored("⚠ --live_view has no effect when --no_cam is set", "yellow"))

    # ==================================================================
    # State machine setup
    # ==================================================================
    state = State.WAITING_TO_GO_TO_START
    prev_state = state
    scan_idx = start_idx
    curr_idx = 0
    trajectory_start_time = 0.0

    move_to_start_result = {
        "ready": False,
        "success": False,
        "trajectory": None,
        "path": None,
    }
    hemisphere_ik_result = {
        "ready": False,
        "valid_joints": True,
        "valid_velocities": True,
        "valid_collisions": True,
        "trajectory": None,
    }
    optical_axis_ik_result = {
        "ready": False,
        "valid_joints": True,
        "valid_velocities": True,
        "valid_collisions": True,
        "trajectory": None,
    }
    rrt_result = {"ready": False, "success": False, "trajectory": None, "path": None}
    reset_result = {"ready": False, "success": False, "trajectory": None, "path": None}

    hemisphere_trajectory = None
    optical_axis_trajectory = None

    # Per-scan photo state
    scan_frame_dir = None
    optical_pause_start_time = 0.0
    optical_halfway_time = 0.0
    capture_traj_times: np.ndarray = np.array([])
    next_capture_idx = 0
    scan_frame_idx = 0
    is_pausing_for_capture = False
    pause_start_sim_time = 0.0
    hold_traj_time = 0.0

    num_move_to_next_clicks = 0
    num_reset_clicks = 0
    jog_click_counts = {name: 0 for name in JOG_BUTTONS}
    _keyboard_confirm = False

    MOVING_STATES = {
        State.COMPUTING_MOVE_TO_START,
        State.MOVING_TO_START,
        State.COMPUTING_IKS,
        State.PLANNING_RRT_FALLBACK,
        State.COMPUTING_RRT_FALLBACK,
        State.MOVING_ALONG_HEMISPHERE,
        State.MOVING_ALONG_RRT,
        State.COMPUTING_OPTICAL_TRAJ,
        State.WAITING_BEFORE_OPTICAL,
        State.MOVING_DOWN_OPTICAL_AXIS,
        State.RETURNING_AFTER_OPTICAL,
        State.COMPUTING_RESET,
        State.RESETTING,
    }
    _jog_target_q: np.ndarray | None = None
    _q_at_scan: np.ndarray | None = None
    _q_jogged: np.ndarray | None = None

    print(colored("\nReady. Press 'Move to Next' in Meshcat to begin.", "cyan"))
    print(
        colored(
            "  At each viewpoint: jog until the object is on the crosshair,\n"
            "  then press 'Move to Next' to scan along the optical axis.",
            "cyan",
        )
    )

    # ==================================================================
    # Main simulation loop
    # ==================================================================
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        if state != prev_state:
            print(colored(f"  [{state.name}]", "grey"))
            prev_state = state

        station_context = station.GetMyContextFromRoot(simulator.get_context())
        internal_plant = station.get_internal_plant()
        internal_plant_context = station.get_internal_plant_context()
        q_now = station.GetOutputPort("iiwa.position_measured").Eval(station_context)

        for i in range(7):
            meshcat.SetSliderValue(f"Joint {i+1} (deg)", np.rad2deg(q_now[i]))

        # PSI display
        p_J2, p_J4, p_J6 = get_sew_joint_positions(
            internal_plant, internal_plant_context
        )
        R_WP_np, R_WR_np = compute_sew_and_ref_matrices(p_J2, p_J4, p_J6, r, v)
        psi_rad = compute_psi_from_matrices(R_WP_np, R_WR_np)
        if R_WR_np is not None:
            meshcat.SetSliderValue("Current PSI (deg)", np.rad2deg(psi_rad))

        # Clear jog target once robot has arrived
        if _jog_target_q is not None and np.all(np.abs(q_now - _jog_target_q) < 0.005):
            _jog_target_q = None

        # Live camera view + keyboard input
        key = 0xFF
        if live_view and not no_cam and _latest_frame_lock is not None:
            with _latest_frame_lock:
                lf = _latest_frame.copy() if _latest_frame is not None else None
            if lf is not None:
                cv2.imshow(
                    "Live View", cv2.resize(lf, (lf.shape[1] // 2, lf.shape[0] // 2))
                )
            key = cv2.waitKey(1) & 0xFF
            if key == KEY_CONFIRM:
                _keyboard_confirm = True
            elif key in KEY_JOG_MAP and state not in MOVING_STATES:
                direction = KEY_JOG_MAP[key]
                q_des = try_jog_tip(
                    station,
                    q_now,
                    direction,
                    0.001,
                    kinematics_solver,
                    elbow_angle,
                    joint_lower_limits,
                    joint_upper_limits,
                )
                if q_des is not None:
                    station.GetInputPort("iiwa.position").FixValue(
                        station_context, q_des
                    )
                    _jog_target_q = q_des

        # ------------------------------------------------------------------
        # Jog buttons — active only when not moving
        if state not in MOVING_STATES:
            for btn_name, direction in JOG_BUTTONS.items():
                if meshcat.GetButtonClicks(btn_name) > jog_click_counts[btn_name]:
                    jog_click_counts[btn_name] += 1
                    q_des = try_jog_tip(
                        station,
                        q_now,
                        direction,
                        0.001,
                        kinematics_solver,
                        elbow_angle,
                        joint_lower_limits,
                        joint_upper_limits,
                    )
                    if q_des is not None:
                        station.GetInputPort("iiwa.position").FixValue(
                            station_context, q_des
                        )
                        _jog_target_q = q_des
                        print(colored(f"  Jogged {btn_name[4:]}", "cyan"))

        # Reset button — available from any non-moving state
        if state not in MOVING_STATES:
            if meshcat.GetButtonClicks("Reset") > num_reset_clicks:
                num_reset_clicks += 1
                print(colored("Planning RRT* reset to default_position...", "cyan"))
                reset_result["ready"] = False
                reset_result["success"] = False
                threading.Thread(
                    target=plan_rrt_star_async,
                    args=(
                        station,
                        q_now,
                        default_position,
                        vel_limits,
                        acc_limits,
                        reset_result,
                    ),
                    daemon=True,
                ).start()
                state = State.COMPUTING_RESET

        # ------------------------------------------------------------------
        if state == State.WAITING_TO_GO_TO_START:
            _btn_fired = (
                meshcat.GetButtonClicks("Move to Next") > num_move_to_next_clicks
            )
            if not _btn_fired and not _keyboard_confirm:
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue
            if _btn_fired:
                num_move_to_next_clicks += 1
            _keyboard_confirm = False

            first_valid = next(
                (i for i in range(n) if not np.isnan(q_array[i]).any()), None
            )
            if first_valid is None:
                print(colored("❌ No valid waypoints found. Quitting.", "red"))
                break

            q_des = q_array[first_valid]
            print(colored(f"Planning RRT* move to waypoint {first_valid}...", "cyan"))
            move_to_start_result["ready"] = False
            move_to_start_result["success"] = False
            threading.Thread(
                target=plan_rrt_star_async,
                args=(
                    station,
                    q_now,
                    q_des,
                    vel_limits,
                    acc_limits,
                    move_to_start_result,
                ),
                daemon=True,
            ).start()
            state = State.COMPUTING_MOVE_TO_START

        # ------------------------------------------------------------------
        elif state == State.COMPUTING_MOVE_TO_START:
            if not move_to_start_result["ready"]:
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue
            if not move_to_start_result["success"]:
                print(
                    colored(
                        "❌ RRT* to first waypoint failed. Retrying on next click.",
                        "red",
                    )
                )
                state = State.WAITING_TO_GO_TO_START
            else:
                trajectory_start_time = simulator.get_context().get_time()
                state = State.MOVING_TO_START

        # ------------------------------------------------------------------
        elif state == State.MOVING_TO_START:
            traj_complete = move_along_trajectory(
                move_to_start_result["trajectory"],
                trajectory_start_time,
                simulator,
                station,
            )
            if traj_complete:
                print(
                    colored(
                        "✓ At first waypoint. Jog to centre the object, then press 'Move to Next'.",
                        "green",
                    )
                )
                state = State.WAITING_FOR_NEXT_SCAN

        # ------------------------------------------------------------------
        elif state == State.WAITING_FOR_NEXT_SCAN:
            meshcat.Delete("hemisphere_traj")
            meshcat.Delete("rrt_raw_path")
            meshcat.Delete("rrt_traj")

            while scan_idx < n and np.isnan(q_array[scan_idx]).any():
                print(
                    colored(
                        f"  Skipping waypoint {scan_idx} (IK failed at pre-computation).",
                        "yellow",
                    )
                )
                scan_idx += 1

            if scan_idx >= n:
                print(colored("✓ All waypoints visited.", "green"))
                state = State.DONE
                continue

            label = labels[scan_idx] if scan_idx < len(labels) else str(scan_idx)
            print(colored(f"\n── Waypoint {scan_idx}/{n - 1} ({label}) ──", "cyan"))

            pose_target = hemisphere_waypoints[scan_idx] @ T_cam_to_tip
            draw_triad(
                meshcat, "next_scan_target", pose_target, length=0.1, radius=0.002
            )

            eef_pose = internal_plant.GetFrameByName(
                "microscope_tip_link"
            ).CalcPoseInWorld(internal_plant_context)

            hemisphere_ik_result["ready"] = False
            threading.Thread(
                target=compute_hemisphere_traj_async,
                args=(
                    station,
                    hemisphere_pos,
                    hemisphere_radius,
                    hemisphere_axis,
                    eef_pose,
                    pose_target,
                    kinematics_solver,
                    q_now,
                    elbow_angle,
                    hemisphere_ik_result,
                    True,
                    scan_idx,
                    joint_lower_limits,
                    joint_upper_limits,
                    speed_factor,
                    max_joint_velocities,
                ),
                daemon=True,
            ).start()
            state = State.COMPUTING_IKS

        # ------------------------------------------------------------------
        elif state == State.COMPUTING_IKS:
            if not hemisphere_ik_result["ready"]:
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue

            hemisphere_trajectory = hemisphere_ik_result["trajectory"]
            hemisphere_valid = (
                hemisphere_ik_result["valid_joints"]
                and hemisphere_ik_result["valid_velocities"]
            )

            if hemisphere_valid:
                plot_trajectory_in_meshcat(
                    station,
                    hemisphere_trajectory,
                    rgba=Rgba(0, 1, 0, 1),
                    name="hemisphere_traj",
                )
                trajectory_start_time = simulator.get_context().get_time()
                state = State.MOVING_ALONG_HEMISPHERE
            else:
                if not hemisphere_ik_result["valid_joints"]:
                    print(colored("  Hemisphere path: invalid joint values.", "yellow"))
                if not hemisphere_ik_result["valid_velocities"]:
                    print(
                        colored(
                            "  Hemisphere path: invalid joint velocities.", "yellow"
                        )
                    )
                print(colored("  Falling back to RRT*-Connect...", "yellow"))
                state = State.PLANNING_RRT_FALLBACK

        # ------------------------------------------------------------------
        elif state == State.PLANNING_RRT_FALLBACK:
            meshcat.Delete("rrt_raw_path")
            meshcat.Delete("rrt_traj")
            rrt_result["ready"] = False
            rrt_result["success"] = False
            q_target = q_array[scan_idx]
            print(
                colored(
                    f"  Launching RRT*-Connect: current → waypoint {scan_idx}...",
                    "cyan",
                )
            )
            threading.Thread(
                target=plan_rrt_star_async,
                args=(station, q_now, q_target, vel_limits, acc_limits, rrt_result),
                daemon=True,
            ).start()
            state = State.COMPUTING_RRT_FALLBACK

        # ------------------------------------------------------------------
        elif state == State.COMPUTING_RRT_FALLBACK:
            if not rrt_result["ready"]:
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue

            if not rrt_result["success"]:
                print(
                    colored(
                        f"❌ RRT*-Connect also failed for waypoint {scan_idx}. Quitting.",
                        "red",
                    )
                )
                break

            plot_rrt_raw_path_in_meshcat(
                station,
                rrt_result["path"],
                name="rrt_raw_path",
                rgba=Rgba(1.0, 0.4, 0.0, 1.0),
            )
            plot_trajectory_in_meshcat(
                station,
                rrt_result["trajectory"],
                rgba=Rgba(0, 1, 1, 1),
                name="rrt_traj",
            )
            print(colored("  ✓ RRT*-Connect found path. Executing...", "green"))
            trajectory_start_time = simulator.get_context().get_time()
            state = State.MOVING_ALONG_RRT

        # ------------------------------------------------------------------
        elif state == State.MOVING_ALONG_RRT:
            traj_complete = move_along_trajectory(
                rrt_result["trajectory"], trajectory_start_time, simulator, station
            )
            if traj_complete:
                _q_at_scan = q_now.copy()
                curr_idx = scan_idx
                print(
                    colored(
                        f"✓ At waypoint {curr_idx}. Jog to centre the object, then press 'Move to Next'.",
                        "green",
                    )
                )
                state = State.AWAITING_JOG_CONFIRM

        # ------------------------------------------------------------------
        elif state == State.MOVING_ALONG_HEMISPHERE:
            traj_complete = move_along_trajectory(
                hemisphere_trajectory, trajectory_start_time, simulator, station
            )
            if traj_complete:
                _q_at_scan = q_now.copy()
                curr_idx = scan_idx
                print(
                    colored(
                        f"✓ At waypoint {curr_idx}. Jog to centre the object, then press 'Move to Next'.",
                        "green",
                    )
                )
                state = State.AWAITING_JOG_CONFIRM

        # ------------------------------------------------------------------
        elif state == State.AWAITING_JOG_CONFIRM:
            # User jogs until object is centred, then presses Enter or "Move to Next"
            _btn_confirm = (
                meshcat.GetButtonClicks("Move to Next") > num_move_to_next_clicks
            )
            if _btn_confirm or _keyboard_confirm:
                if _btn_confirm:
                    num_move_to_next_clicks += 1
                _keyboard_confirm = False

                T_world_cam = internal_plant.GetFrameByName(
                    "camera_link"
                ).CalcPoseInWorld(internal_plant_context)

                # Record the jogged joint config.
                # For the optical trajectory pose, use the hemisphere waypoint
                # rotation but the actual (jogged) camera translation.
                # hemisphere_waypoints use sphere_frame convention: z = p - center
                # (outward), so -z moves toward the object. camera_link has z
                # pointing inward (optical axis), which would scan in the wrong
                # direction if used directly.
                _q_jogged = q_now.copy()
                jogged_cam_pose = RigidTransform(
                    hemisphere_waypoints[curr_idx].rotation(),
                    T_world_cam.translation(),
                )

                # Set up frame output directory for this scan
                scan_frame_dir = scans_base / f"scan{curr_idx:02d}"
                scan_frame_dir.mkdir(parents=True, exist_ok=True)
                print(colored(f"  Frame dir: {scan_frame_dir}", "cyan"))

                if skip_opt:
                    # Return to the original scan pose before moving to the next
                    # waypoint — RETURNING_AFTER_OPTICAL handles the FixValue servo
                    # and scan_idx increment for both branches.
                    state = State.RETURNING_AFTER_OPTICAL
                else:
                    # Launch async optical axis trajectory computation
                    optical_axis_ik_result["ready"] = False
                    threading.Thread(
                        target=compute_optical_axis_traj_async,
                        args=(
                            station,
                            jogged_cam_pose,
                            kinematics_solver,
                            _q_jogged,
                            elbow_angle,
                            optical_axis_ik_result,
                            True,
                            curr_idx,
                            joint_lower_limits,
                            joint_upper_limits,
                            distance_along_optical_axis,
                            speed_factor,
                            max_joint_velocities,
                        ),
                        daemon=True,
                    ).start()
                    print(colored("  Computing optical axis trajectory...", "cyan"))
                    state = State.COMPUTING_OPTICAL_TRAJ

        # ------------------------------------------------------------------
        elif state == State.COMPUTING_OPTICAL_TRAJ:
            if not optical_axis_ik_result["ready"]:
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue

            if not optical_axis_ik_result["valid_joints"]:
                print(
                    colored(
                        "  ❌ Optical traj has joint-limit violations — skipping scan.",
                        "red",
                    )
                )
                state = State.RETURNING_AFTER_OPTICAL
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue
            if not optical_axis_ik_result["valid_velocities"]:
                print(
                    colored(
                        "  ❌ Optical traj exceeds velocity limits — skipping scan.",
                        "red",
                    )
                )
                state = State.RETURNING_AFTER_OPTICAL
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue

            optical_axis_trajectory = optical_axis_ik_result["trajectory"]
            optical_halfway_time = optical_axis_trajectory.end_time() / 2.0

            # Reset per-scan photo state
            capture_traj_times = np.linspace(0.0, optical_halfway_time, num_pictures)
            next_capture_idx = 0
            scan_frame_idx = 0
            is_pausing_for_capture = False
            pause_start_sim_time = 0.0
            hold_traj_time = 0.0

            print(
                colored(
                    f"  ✓ Optical axis trajectory ready "
                    f"(duration {optical_axis_trajectory.end_time():.2f}s). "
                    f"Starting in 1s...",
                    "green",
                )
            )
            optical_pause_start_time = simulator.get_context().get_time()
            state = State.WAITING_BEFORE_OPTICAL

        # ------------------------------------------------------------------
        elif state == State.WAITING_BEFORE_OPTICAL:
            if simulator.get_context().get_time() - optical_pause_start_time >= 1.0:
                trajectory_start_time = simulator.get_context().get_time()
                state = State.MOVING_DOWN_OPTICAL_AXIS

        # ------------------------------------------------------------------
        elif state == State.MOVING_DOWN_OPTICAL_AXIS:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            station_context_mut = station.GetMyMutableContextFromRoot(
                simulator.get_mutable_context()
            )

            if is_pausing_for_capture:
                # Hold robot still during photo pause
                q_hold = optical_axis_trajectory.value(hold_traj_time)
                station.GetInputPort("iiwa.position").FixValue(
                    station_context_mut, q_hold
                )

                elapsed_pause = current_time - pause_start_sim_time
                if elapsed_pause >= 0.5:
                    # Save camera pose
                    if scan_frame_dir is not None:
                        cam_pose = internal_plant.GetFrameByName(
                            "camera_link"
                        ).CalcPoseInWorld(internal_plant_context)
                        np.save(
                            str(scan_frame_dir / f"pose_{scan_frame_idx:05d}.npy"),
                            cam_pose.GetAsMatrix4(),
                        )

                    if not no_cam and _latest_frame_lock is not None:
                        with _latest_frame_lock:
                            frame = (
                                _latest_frame.copy()
                                if _latest_frame is not None
                                else None
                            )
                        if frame is not None and scan_frame_dir is not None:
                            frame_path = str(
                                scan_frame_dir / f"frame_{scan_frame_idx:05d}.png"
                            )
                            _frame_queue.put((frame, frame_path))
                            scan_frame_idx += 1
                            print(
                                colored(
                                    f"  📷 {scan_frame_idx}/{num_pictures} "
                                    f"at t={hold_traj_time:.3f}s",
                                    "cyan",
                                )
                            )
                    trajectory_start_time += elapsed_pause
                    next_capture_idx += 1
                    is_pausing_for_capture = False

            elif traj_time <= optical_axis_trajectory.end_time():
                q_desired = optical_axis_trajectory.value(traj_time)
                station.GetInputPort("iiwa.position").FixValue(
                    station_context_mut, q_desired
                )

                # Trigger photo stop if reached next scheduled time
                if (
                    next_capture_idx < len(capture_traj_times)
                    and traj_time >= capture_traj_times[next_capture_idx]
                ):
                    hold_traj_time = capture_traj_times[next_capture_idx]
                    pause_start_sim_time = current_time
                    is_pausing_for_capture = True
                    print(
                        colored(
                            f"  ⏸ Stop {next_capture_idx + 1}/{num_pictures} "
                            f"at t={hold_traj_time:.3f}s",
                            "yellow",
                        )
                    )

            elif traj_time > optical_axis_trajectory.end_time() + 1.0:
                print(
                    colored(
                        f"  ✓ Optical scan {curr_idx} complete "
                        f"({scan_frame_idx} frames). Returning to scan pose...",
                        "green",
                    )
                )
                state = State.RETURNING_AFTER_OPTICAL

        # ------------------------------------------------------------------
        elif state == State.RETURNING_AFTER_OPTICAL:
            # Command robot back to the original (pre-jog) scan pose
            station.GetInputPort("iiwa.position").FixValue(station_context, _q_at_scan)
            if np.all(np.abs(q_now - _q_at_scan) < 0.01):
                print(colored("  ✓ Back at scan pose.", "green"))
                scan_idx += 1
                if scan_idx >= n:
                    print(
                        colored(
                            "✓ All waypoints scanned. Returning to default position...",
                            "green",
                        )
                    )
                    reset_result["ready"] = False
                    reset_result["success"] = False
                    threading.Thread(
                        target=plan_rrt_star_async,
                        args=(
                            station,
                            q_now,
                            default_position,
                            vel_limits,
                            acc_limits,
                            reset_result,
                        ),
                        daemon=True,
                    ).start()
                    state = State.COMPUTING_RESET
                else:
                    state = State.WAITING_FOR_NEXT_SCAN

        # ------------------------------------------------------------------
        elif state == State.COMPUTING_RESET:
            if not reset_result["ready"]:
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue
            if not reset_result["success"]:
                print(colored("❌ RRT* reset planning failed.", "red"))
                state = State.WAITING_TO_GO_TO_START
            else:
                plot_rrt_raw_path_in_meshcat(
                    station,
                    reset_result["path"],
                    name="rrt_raw_path",
                    rgba=Rgba(1.0, 0.4, 0.0, 1.0),
                )
                plot_trajectory_in_meshcat(
                    station,
                    reset_result["trajectory"],
                    rgba=Rgba(0, 1, 1, 1),
                    name="rrt_traj",
                )
                trajectory_start_time = simulator.get_context().get_time()
                print(colored("  Executing reset trajectory...", "green"))
                state = State.RESETTING

        # ------------------------------------------------------------------
        elif state == State.RESETTING:
            traj_complete = move_along_trajectory(
                reset_result["trajectory"], trajectory_start_time, simulator, station
            )
            if traj_complete:
                print(colored("✓ Reset to default_position.", "green"))
                _q_at_scan = None
                _q_jogged = None
                state = State.DONE if scan_idx >= n else State.WAITING_TO_GO_TO_START

        # ------------------------------------------------------------------
        elif state == State.DONE:
            print(colored("\nScan complete. Press 'Stop Simulation' to exit.", "cyan"))
            break

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)

    # ==================================================================
    # Cleanup
    # ==================================================================
    for btn in [
        "Stop Simulation",
        "Move to Next",
        "Reset",
        *JOG_BUTTONS,
    ]:
        meshcat.DeleteButton(btn)
    for i in range(7):
        meshcat.DeleteSlider(f"Joint {i+1} (deg)")
    meshcat.DeleteSlider("Current PSI (deg)")

    if live_view or not no_cam:
        cv2.destroyAllWindows()

    if not no_cam and camera is not None:
        _capture_stop.set()
        if _frame_queue is not None:
            _frame_queue.put(None)
        _capture_thread.join(timeout=5)
        if _writer_thread is not None:
            _writer_thread.join(timeout=30)
        camera.release()
        print(colored("✓ Camera shut down cleanly.", "cyan"))

    print(colored("Session ended.", "cyan"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manually scan object: jog to centre at each viewpoint, then auto-scan along optical axis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python robot_scan/scan_object_manually.py --use_hardware
        """,
    )
    parser.add_argument(
        "--use_hardware", action="store_true", help="Connect to real iiwa hardware."
    )
    parser.add_argument("--no_cam", action="store_true", help="Disable camera.")
    parser.add_argument(
        "--no_live_view",
        action="store_true",
        help="Disable live camera feed window (live view is on by default).",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Waypoint index to start from (default: 0).",
    )
    parser.add_argument(
        "--camera_source",
        type=int,
        default=CAMERA_SOURCE,
        help="Camera device number.",
    )
    parser.add_argument(
        "--skip_opt",
        action="store_true",
        help="Skip optical axis scan (jog-and-confirm only, no frame capture).",
    )
    parser.add_argument(
        "--num_scan_points",
        type=int,
        default=50,
        help="Number of hemisphere scan points (default: 50).",
    )

    args = parser.parse_args()
    main(
        use_hardware=args.use_hardware,
        no_cam=args.no_cam,
        start_idx=args.start_idx,
        live_view=not args.no_live_view,
        camera_source=args.camera_source,
        skip_opt=args.skip_opt,
        num_scan_points=args.num_scan_points,
    )
