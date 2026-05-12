"""
demos/scan_object.py

Scans an object from hemisphere viewpoints. Two differences from
scan_object_and_save_frames.py:

1. IK for every waypoint is pre-computed upfront and stored as numpy arrays
   (valid configs + failed indices), like find_valid_waypoints.py.

2. When the hemisphere path is unsafe, RRT*-Connect is used as fallback
   instead of kinematic trajectory optimization. If RRT* also fails, the
   program quits. Preview buttons let you inspect the raw / smoothed RRT*
   path before committing to execution.

Usage:
    python demos/scan_object.py
    python demos/scan_object.py --use_hardware
    python demos/scan_object.py --no_cam
    python demos/scan_object.py --skip_opt
"""

import argparse
import queue
import threading
import time

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
    NUM_PICTURES,
    NUM_SCAN_POINTS,
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
    Box,
    ConstantVectorSource,
    DiagramBuilder,
    MeshcatVisualizer,
    Rgba,
    Simulator,
)
from pydrake.lcm import DrakeLcm
from pydrake.systems.primitives import VectorLogSink
from termcolor import colored

from iiwa_setup.iiwa import IiwaHardwareStationDiagram
from iiwa_setup.util.visualizations import draw_triad
from utils.kuka_geo_kin import KinematicsSolver
from utils.planning import (
    compute_hemisphere_traj_async,
    compute_optical_axis_traj_async,
    correct_waypoints_for_principal_point,
    generate_hemisphere_waypoints,
    move_along_trajectory,
    plot_trajectory_in_meshcat,
)
from utils.plotting import plot_hemisphere_ik_results, plot_hemisphere_waypoints
from utils.ray_reconstruction import load_intrinsics
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
    AWAITING_HEMISPHERE_CONFIRM = auto()
    AWAITING_RRT_CONFIRM = auto()
    MOVING_ALONG_RRT = auto()
    WAITING_BEFORE_OPTICAL = auto()
    MOVING_DOWN_OPTICAL_AXIS = auto()
    DONE = auto()


def _animate_configs(
    configs, station, station_context, simulator, meshcat
):  # TODO: This is super dangerous in hardware, would actually move the robot!!!
    """Animate robot through configs forward then in reverse (for previewing)."""
    for q in list(configs) + list(reversed(configs)):
        station.GetInputPort("iiwa.position").FixValue(station_context, q)
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)
        for i, qi in enumerate(q):
            meshcat.SetSliderValue(f"Joint {i+1} (deg)", round(np.rad2deg(qi), 1))


def main(
    use_hardware: bool,
    no_cam: bool = False,
    live_view: bool = True,
    skip_opt: bool = False,
    start_idx: int = 0,
    wait: bool = False,
    num_scan_points: int = NUM_SCAN_POINTS,
    only_rrt: bool = False,
) -> None:
    cfg = get_config(use_hardware)
    speed_factor = cfg["speed_factor"]
    max_joint_velocities = cfg["max_joint_velocities"]
    vel_limits = cfg["vel_limits"]
    acc_limits = cfg["acc_limits"]

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

    coverage = 1.0
    distance_along_optical_axis = 0.04
    num_pictures = NUM_PICTURES

    elbow_angle = ELBOW_ANGLE
    default_position = DEFAULT_POSITION
    r = R_AXIS
    v = V_AXIS
    T_cam_to_tip = T_CAM_TO_TIP

    # Load intrinsics for crosshair overlay
    camera_K = load_intrinsics()
    cx_full = int(camera_K[0, 2])
    cy_full = int(camera_K[1, 2])
    cx_disp = cx_full // 2
    cy_disp = cy_full // 2

    print(colored("=" * 60, "cyan"))
    print(colored("  scan_object — configuration", "cyan"))
    print(colored("=" * 60, "cyan"))
    print(colored("  Args:", "white"))
    print(f"    use_hardware     : {use_hardware}")
    print(f"    no_cam           : {no_cam}")
    print(f"    live_view        : {live_view}")
    print(f"    skip_opt         : {skip_opt}")
    print(f"    start_idx        : {start_idx}")
    print(f"    wait             : {wait}")
    print(f"    only_rrt         : {only_rrt}")
    print(f"    num_scan_points  : {num_scan_points}")
    print(colored("  Hemisphere:", "white"))
    print(f"    center  : {hemisphere_pos}")
    print(f"    radius  : {hemisphere_radius}")
    print(f"    axis    : {hemisphere_axis}")
    print(colored("  Intrinsics (crosshair):", "white"))
    print(f"    cx = {cx_full}  cy = {cy_full}  →  display ({cx_disp}, {cy_disp})")
    print(colored("  Config:", "white"))
    print(f"    speed_factor         : {speed_factor}")
    print(f"    max_joint_vel (deg/s): {np.rad2deg(max_joint_velocities).round(2)}")
    print(f"    vel_limits  (rad/s)  : {vel_limits.round(3)}")
    print(f"    acc_limits  (rad/s²) : {acc_limits.round(3)}")
    print(colored("=" * 60, "cyan"))

    # ==================================================================
    # Outputs setup
    # ==================================================================
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    scans_base = Path(__file__).parent.parent / "microscope-data" / "scans" / date_str
    scans_base.mkdir(parents=True, exist_ok=True)

    # Write scan parameters to the session folder
    params_text = f"""\
date: {date_str}
hemisphere_pos: {hemisphere_pos.tolist()}
hemisphere_radius: {hemisphere_radius}
num_scan_points: {num_scan_points}
coverage: {coverage}
distance_along_optical_axis: {distance_along_optical_axis}
num_pictures: {num_pictures}
elbow_angle_deg: {np.rad2deg(elbow_angle):.2f}
use_hardware: {use_hardware}
skip_opt: {skip_opt}
wait: {wait}
start_idx: {start_idx}
"""
    (scans_base / "scan_params.txt").write_text(params_text)
    print(colored(f"✓ Scan params saved to {scans_base / 'scan_params.txt'}", "cyan"))

    # ==================================================================
    # Waypoint generation
    # ==================================================================
    hemisphere_waypoints = generate_hemisphere_waypoints(
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
        num_scan_points=num_scan_points,
        coverage=coverage,
    )
    hemisphere_waypoints = correct_waypoints_for_principal_point(
        hemisphere_waypoints, camera_K, image_width=1920, image_height=1080
    )
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

    state_logger = builder.AddSystem(VectorLogSink(7))
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        state_logger.get_input_port(),
    )

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
        opacity=0.1,
        name="camera_link",
    )

    optical_center_frame = station.get_internal_plant().GetFrameByName("optical_center")
    AddFrameTriadIllustration(
        scene_graph=station.internal_station.get_scene_graph(),
        plant=station.get_internal_plant(),
        frame=optical_center_frame,
        length=0.1,
        radius=0.001,
        name="optical_center",
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
    meshcat.AddButton("Move to Scan")
    meshcat.AddButton("Preview RRT* Raw")
    meshcat.AddButton("Preview RRT* Smooth")
    meshcat.AddButton("Execute Path")

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

    # ==================================================================
    # Pre-compute IK for all waypoints (like find_valid_waypoints.py)
    # ==================================================================
    n = len(hemisphere_waypoints)
    q_array = np.full((n, 7), np.nan)  # valid rows filled in; failed stay NaN
    failed_indices = []
    q_prev = default_position.copy()

    print(colored(f"\nPre-computing IK for {n} waypoints...", "cyan"))
    for i, wp in enumerate(hemisphere_waypoints):
        target_rot = wp.rotation().matrix()
        target_pos = wp.translation()

        Q = kinematics_solver.IK_for_microscope(target_rot, target_pos, psi=elbow_angle)
        Q = filter_ik_solutions(
            station,
            Q,
            target_rot,
            target_pos,
            joint_lower_limits,
            joint_upper_limits,
            tip_frame_name="optical_center",
        )

        if Q.shape[0] == 0:
            print(colored(f"  [{i}] FAIL: no valid IK solutions", "yellow"))
            failed_indices.append(i)
            continue

        q_des = kinematics_solver.find_closest_solution(Q, q_prev)
        q_array[i] = q_des
        q_prev = q_des
        print(f"  [{i}] OK: {np.rad2deg(q_des).round(2)} deg")

    n_valid = int(np.sum(~np.isnan(q_array).any(axis=1)))
    print(
        colored(
            f"\nPre-computation done: {n_valid}/{n} valid, {len(failed_indices)} failed.",
            "cyan",
        )
    )
    if failed_indices:
        print(colored(f"  Failed indices: {failed_indices}", "yellow"))

    # Persist to disk
    np.savetxt(outputs_dir / "hemisphere_q_solutions.csv", q_array, delimiter=",")
    np.save(outputs_dir / "hemisphere_q_solutions.npy", q_array)
    np.save(outputs_dir / "hemisphere_q_failed_indices.npy", np.array(failed_indices))

    # Save waypoint positions and rotations
    wp_positions = np.array([wp.translation() for wp in hemisphere_waypoints])
    wp_rotations = np.array([wp.rotation().matrix() for wp in hemisphere_waypoints])
    np.save(outputs_dir / "hemisphere_waypoints_positions.npy", wp_positions)
    np.save(outputs_dir / "hemisphere_waypoints_rotations.npy", wp_rotations)
    print(colored(f"  Saved IK arrays and waypoints to {outputs_dir}", "cyan"))

    # Plot success / failed waypoints separately
    plot_hemisphere_ik_results(
        hemisphere_waypoints,
        failed_indices,
        hemisphere_pos,
        hemisphere_radius,
        outputs_dir,
    )

    # Draw only reachable waypoints in Meshcat
    failed_set = set(failed_indices)
    for i, wp in enumerate(hemisphere_waypoints):
        if i not in failed_set:
            draw_triad(
                meshcat,
                f"hemisphere_waypoint_{i}",
                wp,
                length=0.02,
                radius=0.001,
                opacity=0.1,
            )

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
        camera = cv2.VideoCapture(CAMERA_SOURCE)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not camera.isOpened():
            print(
                colored(
                    f"⚠ Could not open camera device {CAMERA_SOURCE} – frames will NOT be saved",
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

    # ==================================================================
    # State machine setup
    # ==================================================================
    state = State.WAITING_TO_GO_TO_START
    prev_state = State.WAITING_TO_GO_TO_START
    scan_idx = start_idx  # waypoint we are planning to visit next
    curr_idx = 0  # waypoint robot is currently at
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
    rrt_result = {
        "ready": False,
        "success": False,
        "trajectory": None,
        "path": None,
    }

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

    # Timing trackers
    scan_start_wall_time = None
    move_durations = []  # trajectory end_time() for each between-scan move
    optical_axis_traj_time = None  # set once from first optical axis trajectory

    # Button click trackers
    num_move_to_scan_clicks = 0
    num_preview_raw_clicks = 0
    num_preview_smooth_clicks = 0
    num_execute_clicks = 0

    print(colored("\nReady. Press 'Move to Scan' in Meshcat to begin.", "cyan"))

    # ==================================================================
    # Main simulation loop
    # ==================================================================
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        if state != prev_state:
            print(colored(f"  [{state.name}]", "grey"))
            prev_state = state

        # Refresh context and current joint positions every iteration
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

        # Live camera view with crosshairs
        if live_view and not no_cam and _latest_frame_lock is not None:
            with _latest_frame_lock:
                lf = _latest_frame.copy() if _latest_frame is not None else None
            if lf is not None:
                disp = cv2.resize(lf, (lf.shape[1] // 2, lf.shape[0] // 2))
                # Red crosshair at calibrated principal point
                cv2.drawMarker(
                    disp, (cx_disp, cy_disp), (0, 0, 255), cv2.MARKER_CROSS, 40, 4
                )
                # Green crosshair at image center
                img_cx = disp.shape[1] // 2
                img_cy = disp.shape[0] // 2
                cv2.drawMarker(
                    disp, (img_cx, img_cy), (0, 255, 0), cv2.MARKER_CROSS, 40, 2
                )
                cv2.imshow("Live View", disp)
                cv2.waitKey(1)

        # ------------------------------------------------------------------
        if state == State.WAITING_TO_GO_TO_START:
            if meshcat.GetButtonClicks("Move to Scan") <= num_move_to_scan_clicks:
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue
            num_move_to_scan_clicks += 1

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
                scan_start_wall_time = time.time()
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
                print(colored("✓ At first waypoint. Starting scan.", "green"))
                state = State.WAITING_FOR_NEXT_SCAN

        # ------------------------------------------------------------------
        elif state == State.WAITING_FOR_NEXT_SCAN:
            # Clear previous path visualizations
            meshcat.Delete("hemisphere_traj")
            meshcat.Delete("rrt_raw_path")
            meshcat.Delete("rrt_traj")

            # Skip pre-computation failures
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

            print(colored(f"\n── Waypoint {scan_idx}/{n - 1} ──", "cyan"))

            pose_target = hemisphere_waypoints[scan_idx] @ T_cam_to_tip
            draw_triad(
                meshcat, "next_scan_target", pose_target, length=0.1, radius=0.002
            )

            eef_pose = internal_plant.GetFrameByName(
                "microscope_tip_link"
            ).CalcPoseInWorld(internal_plant_context)
            pose_curr = hemisphere_waypoints[
                scan_idx
            ]  # raw waypoint at the target — optical axis z-axis must not be flipped

            # Launch hemisphere IK thread (skipped when --only_rrt)
            if only_rrt:
                hemisphere_ik_result["ready"] = True
                hemisphere_ik_result["valid_joints"] = False
                hemisphere_ik_result["valid_velocities"] = False
                hemisphere_ik_result["valid_collisions"] = False
                hemisphere_ik_result["trajectory"] = None
            else:
                hemisphere_ik_result["ready"] = False
                traj_save_path = (
                    scans_base / "trajectory_data" / f"waypoint_{scan_idx:05d}"
                )
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
                    kwargs={
                        "save_path": traj_save_path,
                        "plot_save_dir": scans_base / f"scan{scan_idx:02d}",
                    },
                    daemon=True,
                ).start()

            if not skip_opt:
                optical_axis_ik_result["ready"] = False
                threading.Thread(
                    target=compute_optical_axis_traj_async,
                    args=(
                        station,
                        pose_curr,
                        kinematics_solver,
                        q_array[scan_idx],
                        elbow_angle,
                        optical_axis_ik_result,
                        True,
                        scan_idx,
                        joint_lower_limits,
                        joint_upper_limits,
                        distance_along_optical_axis,
                        speed_factor,
                        max_joint_velocities,
                    ),
                    daemon=True,
                ).start()
            else:
                optical_axis_ik_result["ready"] = True

            state = State.COMPUTING_IKS

        # ------------------------------------------------------------------
        elif state == State.COMPUTING_IKS:
            both_ready = hemisphere_ik_result["ready"] and (
                skip_opt or optical_axis_ik_result["ready"]
            )
            if not both_ready:
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue

            hemisphere_trajectory = hemisphere_ik_result["trajectory"]
            hemisphere_valid = (
                hemisphere_ik_result["valid_joints"]
                and hemisphere_ik_result["valid_velocities"]
            )

            # Set up optical axis + photo state for this scan
            if not skip_opt:
                optical_axis_trajectory = optical_axis_ik_result["trajectory"]
                if optical_axis_traj_time is None:
                    optical_axis_traj_time = optical_axis_trajectory.end_time()
                optical_halfway_time = optical_axis_trajectory.end_time() / 2.0
                scan_frame_dir = scans_base / f"scan{scan_idx:02d}"
                scan_frame_dir.mkdir(parents=True, exist_ok=True)
                capture_traj_times = np.linspace(
                    0.0, optical_halfway_time, num_pictures
                )
                next_capture_idx = 0
                scan_frame_idx = 0
                is_pausing_for_capture = False
                pause_start_sim_time = 0.0
                hold_traj_time = 0.0
                print(colored(f"  Frame dir: {scan_frame_dir}", "cyan"))

            if hemisphere_valid:
                plot_trajectory_in_meshcat(
                    station,
                    hemisphere_trajectory,
                    rgba=Rgba(0, 1, 0, 1),
                    name="hemisphere_traj",
                )
                if wait:
                    print(
                        colored(
                            f"  ✓ Hemisphere trajectory ready for waypoint {scan_idx}.\n"
                            "    Press 'Execute Path' to run it.",
                            "green",
                        )
                    )
                    state = State.AWAITING_HEMISPHERE_CONFIRM
                else:
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
        elif state == State.AWAITING_HEMISPHERE_CONFIRM:
            execute = meshcat.GetButtonClicks("Execute Path") > num_execute_clicks
            if execute:
                num_execute_clicks += 1
                trajectory_start_time = simulator.get_context().get_time()
                print(colored("  Executing hemisphere trajectory...", "green"))
                state = State.MOVING_ALONG_HEMISPHERE

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

            # Draw both path visualizations
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
            if wait:
                print(
                    colored(
                        "  ✓ RRT*-Connect found path.\n"
                        "    Press 'Preview RRT* Raw', 'Preview RRT* Smooth', or 'Execute Path'.",
                        "green",
                    )
                )
                state = State.AWAITING_RRT_CONFIRM
            else:
                trajectory_start_time = simulator.get_context().get_time()
                print(colored("  ✓ RRT*-Connect found path. Executing...", "green"))
                state = State.MOVING_ALONG_RRT

        # ------------------------------------------------------------------
        elif state == State.AWAITING_RRT_CONFIRM:
            preview_raw = (
                meshcat.GetButtonClicks("Preview RRT* Raw") > num_preview_raw_clicks
            )
            preview_smooth = (
                meshcat.GetButtonClicks("Preview RRT* Smooth")
                > num_preview_smooth_clicks
            )
            execute = meshcat.GetButtonClicks("Execute Path") > num_execute_clicks

            if preview_raw:
                num_preview_raw_clicks += 1
                print(colored("  Animating raw RRT* waypoints...", "cyan"))
                _animate_configs(
                    rrt_result["path"], station, station_context, simulator, meshcat
                )
                print(colored("  ✓ Raw preview done.", "cyan"))

            elif preview_smooth:
                num_preview_smooth_clicks += 1
                print(colored("  Animating TOPPRA-smoothed trajectory...", "cyan"))
                spline = rrt_result["trajectory"]
                ts = np.linspace(spline.start_time(), spline.end_time(), 50)
                smooth_configs = [spline.value(t).flatten() for t in ts]
                _animate_configs(
                    smooth_configs, station, station_context, simulator, meshcat
                )
                print(colored("  ✓ Smooth preview done.", "cyan"))

            elif execute:
                num_execute_clicks += 1
                trajectory_start_time = simulator.get_context().get_time()
                print(colored("  Executing RRT* trajectory...", "green"))
                state = State.MOVING_ALONG_RRT

        # ------------------------------------------------------------------
        elif state == State.MOVING_ALONG_RRT:
            traj_complete = move_along_trajectory(
                rrt_result["trajectory"],
                trajectory_start_time,
                simulator,
                station,
            )
            if traj_complete:
                move_durations.append(rrt_result["trajectory"].end_time())
                curr_idx = scan_idx
                scan_idx += 1
                if skip_opt:
                    state = State.WAITING_FOR_NEXT_SCAN
                else:
                    optical_pause_start_time = simulator.get_context().get_time()
                    state = State.WAITING_BEFORE_OPTICAL

        # ------------------------------------------------------------------
        elif state == State.MOVING_ALONG_HEMISPHERE:
            traj_complete = move_along_trajectory(
                hemisphere_trajectory,
                trajectory_start_time,
                simulator,
                station,
            )
            if traj_complete:
                move_durations.append(hemisphere_trajectory.end_time())
                curr_idx = scan_idx
                scan_idx += 1
                if skip_opt:
                    state = State.WAITING_FOR_NEXT_SCAN
                else:
                    optical_pause_start_time = simulator.get_context().get_time()
                    state = State.WAITING_BEFORE_OPTICAL

        # ------------------------------------------------------------------
        elif state == State.WAITING_BEFORE_OPTICAL:
            if simulator.get_context().get_time() - optical_pause_start_time >= 2.0:
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
                            "optical_center"
                        ).CalcPoseInWorld(internal_plant_context)
                        np.save(
                            str(scan_frame_dir / f"pose_{scan_frame_idx:05d}.npy"),
                            cam_pose.GetAsMatrix4(),
                        )

                    if not no_cam:
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
                print(colored("  ✓ Optical axis trajectory complete.", "green"))
                state = State.WAITING_FOR_NEXT_SCAN

        # ------------------------------------------------------------------
        elif state == State.DONE:
            # Append timing data to params file
            total_run_s = (
                time.time() - scan_start_wall_time if scan_start_wall_time else 0.0
            )
            avg_move_s = float(np.mean(move_durations)) if move_durations else 0.0
            timing_text = (
                f"\n# Timing\n"
                f"total_run_time_s: {total_run_s:.2f}\n"
                f"num_moves: {len(move_durations)}\n"
                f"move_times_s: {[round(t, 3) for t in move_durations]}\n"
                f"avg_move_time_s: {avg_move_s:.3f}\n"
                f"optical_axis_traj_time_s: {round(optical_axis_traj_time, 3) if optical_axis_traj_time is not None else 'N/A'}\n"
            )
            with open(scans_base / "scan_params.txt", "a") as f:
                f.write(timing_text)
            print(colored(f"✓ Timing data appended to scan_params.txt", "cyan"))

            # Save joint trajectory log
            ctx = simulator.get_context()
            log = state_logger.FindLog(ctx)
            t_log = log.sample_times()
            data_log = log.data()
            out = np.vstack((t_log, data_log)).T
            log_path = outputs_dir / "joint_log.csv"
            np.savetxt(
                log_path,
                out,
                delimiter=",",
                header="time," + ",".join([f"q{i}" for i in range(data_log.shape[0])]),
                comments="",
            )
            print(colored(f"✓ Joint log saved → {log_path}", "cyan"))

            # Save combined joint positions + velocities plot
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            dt_log = np.diff(t_log)
            vel_log = np.diff(data_log, axis=1) / dt_log[np.newaxis, :]
            vel_t_log = t_log[:-1]

            fig = Figure(figsize=(18, 14))
            FigureCanvasAgg(fig)
            axes = fig.subplots(7, 2, sharex="col")
            fig.suptitle(
                "Joint Log - Positions & Velocities", fontsize=14, fontweight="bold"
            )
            for i in range(7):
                ax_p = axes[i, 0]
                ax_p.plot(t_log, np.rad2deg(data_log[i]), linewidth=1.0, color="C0")
                ax_p.set_ylabel(f"J{i+1}\n(deg)", fontsize=8)
                ax_p.grid(True, alpha=0.3)
                if i == 0:
                    ax_p.set_title("Joint Positions", fontsize=11)
                if i == 6:
                    ax_p.set_xlabel("Time (s)", fontsize=10)

                ax_v = axes[i, 1]
                ax_v.plot(vel_t_log, np.rad2deg(vel_log[i]), linewidth=1.0, color="C1")
                ax_v.set_ylabel("(deg/s)", fontsize=8)
                ax_v.grid(True, alpha=0.3)
                if i == 0:
                    ax_v.set_title("Joint Velocities", fontsize=11)
                if i == 6:
                    ax_v.set_xlabel("Time (s)", fontsize=10)
                if max_joint_velocities is not None:
                    lim = np.rad2deg(max_joint_velocities[i])
                    ax_v.axhline(
                        lim, color="red", linestyle="--", linewidth=1.0, alpha=0.8
                    )
                    ax_v.axhline(
                        -lim, color="red", linestyle="--", linewidth=1.0, alpha=0.8
                    )

            fig.tight_layout()
            log_plot_path = outputs_dir / "joint_log.png"
            fig.savefig(log_plot_path, dpi=150, bbox_inches="tight")
            print(colored(f"✓ Joint log plot saved → {log_plot_path}", "cyan"))
            break

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)

    # ==================================================================
    # Cleanup
    # ==================================================================
    for btn in [
        "Stop Simulation",
        "Move to Scan",
        "Preview RRT* Raw",
        "Preview RRT* Smooth",
        "Execute Path",
    ]:
        meshcat.DeleteButton(btn)
    for i in range(7):
        meshcat.DeleteSlider(f"Joint {i+1} (deg)")
    meshcat.DeleteSlider("Current PSI (deg)")

    if live_view:
        cv2.destroyAllWindows()

    if not no_cam and camera is not None:
        _capture_stop.set()
        _frame_queue.put(None)
        _capture_thread.join(timeout=5)
        _writer_thread.join(timeout=30)
        camera.release()
        print(colored("✓ Camera shut down cleanly.", "cyan"))

    print(colored("Simulation ended.", "cyan"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware", action="store_true", help="Connect to real iiwa hardware."
    )
    parser.add_argument("--no_cam", action="store_true", help="Disable camera capture.")
    parser.add_argument(
        "--live_view", action="store_true", help="Show live camera feed in a window."
    )
    parser.add_argument(
        "--skip_opt",
        action="store_true",
        help="Skip optical axis trajectory (no photos).",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Waypoint index to start scanning from (default: 0).",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Pause and wait for 'Execute Path' confirmation before each trajectory (default: execute immediately).",
    )
    parser.add_argument(
        "--num_scan_points",
        type=int,
        default=NUM_SCAN_POINTS,
        help=f"Number of hemisphere scan points (default: {NUM_SCAN_POINTS}).",
    )
    parser.add_argument(
        "--only_rrt",
        action="store_true",
        help="Always use RRT* for inter-waypoint moves; never use hemisphere trajectory.",
    )
    args = parser.parse_args()
    main(
        use_hardware=args.use_hardware,
        no_cam=args.no_cam,
        live_view=args.live_view,
        skip_opt=args.skip_opt,
        start_idx=args.start_idx,
        wait=args.wait,
        num_scan_points=args.num_scan_points,
        only_rrt=args.only_rrt,
    )
