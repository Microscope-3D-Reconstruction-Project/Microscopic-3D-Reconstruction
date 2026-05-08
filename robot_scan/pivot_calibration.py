"""
robot_scan/pivot_calibration.py

Pivot-point calibration for the microscope camera.

Moves the robot to the "straight" centering viewpoint, lets the user jog
until the object appears centred, then generates 6 end-effector roll poses
(60° increments around the tip's z-axis). At each pose the user clicks the
object centre in the live camera view. A circle is fitted to all accumulated
clicks to find the true optical-axis pixel offset relative to the calibrated
principal point (cx, cy).

Usage:
    python robot_scan/pivot_calibration.py --use_hardware
"""

import argparse
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
from pydrake.systems.primitives import VectorLogSink
from termcolor import colored

from iiwa_setup.iiwa import IiwaHardwareStationDiagram
from iiwa_setup.util.visualizations import draw_triad
from utils.centering_waypoints import generate_centering_waypoints
from utils.circle_fitting import fit_circle
from utils.jog import try_jog_tip
from utils.kuka_geo_kin import KinematicsSolver
from utils.planning import move_along_trajectory, plot_trajectory_in_meshcat
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
    AWAITING_JOG = auto()
    AWAITING_CLICK = auto()
    COMPUTING_MOVE_TO_POSE = auto()
    MOVING_TO_POSE = auto()
    COMPUTING_RESET = auto()
    RESETTING = auto()
    DONE = auto()


WIN = "pivot_calibration"
N_POSES = 6


def main(
    use_hardware: bool,
    no_cam: bool = False,
    camera_source: int = CAMERA_SOURCE,
) -> None:
    cfg = get_config(use_hardware)
    vel_limits = cfg["vel_limits"]
    acc_limits = cfg["acc_limits"]

    elbow_angle = ELBOW_ANGLE
    default_position = DEFAULT_POSITION
    r = R_AXIS
    v = V_AXIS
    T_cam_to_tip = T_CAM_TO_TIP

    camera_K = load_intrinsics()
    cx_full = int(camera_K[0, 2])
    cy_full = int(camera_K[1, 2])
    cx_disp = cx_full // 2
    cy_disp = cy_full // 2

    print(colored("=" * 60, "cyan"))
    print(colored("  pivot_calibration — configuration", "cyan"))
    print(colored("=" * 60, "cyan"))
    print(f"    use_hardware  : {use_hardware}")
    print(f"    no_cam        : {no_cam}")
    print(f"    camera_source : {camera_source}")
    print(colored("  Hemisphere:", "white"))
    print(f"    center : {HEMISPHERE_CENTER}")
    print(f"    radius : {HEMISPHERE_RADIUS}")
    print(colored("  Intrinsics (principal point):", "white"))
    print(f"    cx={cx_full}  cy={cy_full}  →  display ({cx_disp}, {cy_disp})")
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

    hemisphere_pos = np.asarray(HEMISPHERE_CENTER, dtype=float)
    hemisphere_radius = HEMISPHERE_RADIUS
    hemisphere_axis = np.array([-1.0, 0.0, 0.0])

    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = (
        Path(__file__).parent.parent
        / "microscope-data"
        / "pivot_calibration"
        / date_str
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    print(colored(f"✓ Results → {save_dir}", "cyan"))

    # Only the first (straight) centering waypoint is used as start pose
    centering_waypoints = generate_centering_waypoints(
        hemisphere_pos, hemisphere_radius, hemisphere_axis
    )
    start_waypoint = centering_waypoints[0]

    # Read hardware position via LCM before building the diagram
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
        print(colored(f"✓ Hardware q: {np.rad2deg(initial_q).round(1)} deg", "cyan"))
    else:
        initial_q = default_position

    # Diagram
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
        name="camera_link",
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    meshcat = station.internal_meshcat
    joint_lower_limits = station.get_internal_plant().GetPositionLowerLimits()
    joint_upper_limits = station.get_internal_plant().GetPositionUpperLimits()

    # Pre-compute IK for the start (straight) waypoint
    Q_start = kinematics_solver.IK_for_microscope(
        start_waypoint.rotation().matrix(),
        start_waypoint.translation(),
        psi=elbow_angle,
    )
    Q_start = filter_ik_solutions(
        station,
        Q_start,
        start_waypoint.rotation().matrix(),
        start_waypoint.translation(),
        joint_lower_limits,
        joint_upper_limits,
    )
    if Q_start.shape[0] == 0:
        print(colored("❌ No IK solution for start waypoint. Cannot proceed.", "red"))
        return
    q_start = kinematics_solver.find_closest_solution(Q_start, default_position)
    print(colored(f"✓ Start IK: {np.rad2deg(q_start).round(2)} deg", "cyan"))

    draw_triad(
        meshcat,
        "start_waypoint",
        start_waypoint @ T_cam_to_tip,
        length=0.05,
        radius=0.002,
    )

    # Buttons + sliders
    meshcat.AddButton("Stop Simulation")
    meshcat.AddButton("Move to Next")
    meshcat.AddButton("Generate Poses")
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

    for i in range(7):
        meshcat.AddSlider(
            f"Joint {i+1} (deg)",
            np.rad2deg(joint_lower_limits[i]),
            np.rad2deg(joint_upper_limits[i]),
            0.1,
            0,
        )
    meshcat.AddSlider("Current PSI (deg)", -180, 180, 0.1, 0)

    # Camera
    camera = None
    _latest_frame = None
    _latest_frame_lock = None
    _capture_stop = None
    _capture_thread = None
    _current_click = {"x": None, "y": None}  # half-res coords set by mouse callback

    def _mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            _current_click["x"] = x
            _current_click["y"] = y

    if not no_cam:
        camera = cv2.VideoCapture(camera_source)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not camera.isOpened():
            print(colored(f"⚠ Could not open camera {camera_source}", "yellow"))

        _latest_frame_lock = threading.Lock()
        _capture_stop = threading.Event()

        def _capture_loop():
            nonlocal _latest_frame
            while not _capture_stop.is_set():
                ret, frame = camera.read()
                if ret:
                    with _latest_frame_lock:
                        _latest_frame = frame

        _capture_thread = threading.Thread(target=_capture_loop, daemon=True)
        _capture_thread.start()

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, _mouse_callback)
        print(colored("✓ Camera started. Click image to mark object centre.", "cyan"))
    else:
        print(colored("✓ Camera disabled via --no_cam", "yellow"))

    # State machine
    state = State.WAITING_TO_GO_TO_START
    prev_state = state
    trajectory_start_time = 0.0
    pose_idx = 0
    q_roll = np.full((N_POSES, 7), np.nan)
    committed_clicks: list[tuple[int, int]] = []  # full-res (x, y)

    move_to_start_result = {
        "ready": False,
        "success": False,
        "trajectory": None,
        "path": None,
    }
    move_result = {"ready": False, "success": False, "trajectory": None, "path": None}
    reset_result = {"ready": False, "success": False, "trajectory": None, "path": None}

    num_move_to_next_clicks = 0
    num_generate_poses_clicks = 0
    num_reset_clicks = 0
    jog_click_counts = {name: 0 for name in JOG_BUTTONS}

    MOVING_STATES = {
        State.COMPUTING_MOVE_TO_START,
        State.MOVING_TO_START,
        State.COMPUTING_MOVE_TO_POSE,
        State.MOVING_TO_POSE,
        State.COMPUTING_RESET,
        State.RESETTING,
    }
    _jog_target_q: np.ndarray | None = None

    print(
        colored(
            "\nReady. Press 'Move to Next' to go to the centering waypoint.", "cyan"
        )
    )

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

        p_J2, p_J4, p_J6 = get_sew_joint_positions(
            internal_plant, internal_plant_context
        )
        R_WP_np, R_WR_np = compute_sew_and_ref_matrices(p_J2, p_J4, p_J6, r, v)
        psi_rad = compute_psi_from_matrices(R_WP_np, R_WR_np)
        if R_WR_np is not None:
            meshcat.SetSliderValue("Current PSI (deg)", np.rad2deg(psi_rad))

        if _jog_target_q is not None and np.all(np.abs(q_now - _jog_target_q) < 0.005):
            _jog_target_q = None

        # Live view with overlays
        if not no_cam and _latest_frame_lock is not None:
            with _latest_frame_lock:
                lf = _latest_frame.copy() if _latest_frame is not None else None
            if lf is not None:
                half = (lf.shape[1] // 2, lf.shape[0] // 2)
                disp = cv2.resize(lf, half)

                # Yellow crosshair at calibrated principal point
                cv2.drawMarker(
                    disp, (cx_disp, cy_disp), (0, 255, 255), cv2.MARKER_CROSS, 40, 2
                )

                # Red dots: committed clicks (half-res)
                for cx_c, cy_c in committed_clicks:
                    cv2.circle(disp, (cx_c // 2, cy_c // 2), 6, (0, 0, 255), -1)

                # Green dot: current pending click
                if _current_click["x"] is not None:
                    cv2.circle(
                        disp,
                        (_current_click["x"], _current_click["y"]),
                        6,
                        (0, 255, 0),
                        -1,
                    )

                # Blue fitted circle (≥ 3 committed, computed in full-res, displayed in half-res)
                if len(committed_clicks) >= 3:
                    fit = fit_circle(committed_clicks)
                    if fit is not None:
                        fx, fy, fr = fit
                        cv2.circle(
                            disp,
                            (int(fx) // 2, int(fy) // 2),
                            max(1, int(fr) // 2),
                            (255, 0, 0),
                            2,
                        )
                        cv2.circle(
                            disp, (int(fx) // 2, int(fy) // 2), 4, (255, 128, 0), -1
                        )

                label = f"Pose {pose_idx}/{N_POSES - 1}  clicks={len(committed_clicks)}"
                cv2.putText(
                    disp,
                    label,
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow(WIN, disp)
                cv2.waitKey(1)

        # Jog buttons (active in any non-moving state)
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

        # Reset button (any non-moving state)
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

        # ── state machine ──────────────────────────────────────────────────
        if state == State.WAITING_TO_GO_TO_START:
            if meshcat.GetButtonClicks("Move to Next") <= num_move_to_next_clicks:
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue
            num_move_to_next_clicks += 1
            print(colored("Planning RRT* to centering waypoint...", "cyan"))
            move_to_start_result["ready"] = False
            move_to_start_result["success"] = False
            threading.Thread(
                target=plan_rrt_star_async,
                args=(
                    station,
                    q_now,
                    q_start,
                    vel_limits,
                    acc_limits,
                    move_to_start_result,
                ),
                daemon=True,
            ).start()
            state = State.COMPUTING_MOVE_TO_START

        elif state == State.COMPUTING_MOVE_TO_START:
            if not move_to_start_result["ready"]:
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue
            if not move_to_start_result["success"]:
                print(colored("❌ RRT* to start failed. Retrying on next click.", "red"))
                state = State.WAITING_TO_GO_TO_START
            else:
                plot_rrt_raw_path_in_meshcat(
                    station,
                    move_to_start_result["path"],
                    name="rrt_to_start",
                    rgba=Rgba(1.0, 0.4, 0.0, 1.0),
                )
                plot_trajectory_in_meshcat(
                    station,
                    move_to_start_result["trajectory"],
                    rgba=Rgba(0, 1, 1, 1),
                    name="traj_to_start",
                )
                trajectory_start_time = simulator.get_context().get_time()
                state = State.MOVING_TO_START

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
                        "✓ At start waypoint. Jog to centre the object on the crosshair,\n"
                        "  then press 'Generate Poses'.",
                        "green",
                    )
                )
                state = State.AWAITING_JOG

        elif state == State.AWAITING_JOG:
            if meshcat.GetButtonClicks("Generate Poses") <= num_generate_poses_clicks:
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue
            num_generate_poses_clicks += 1

            print(colored("Generating 6 roll poses and pre-computing IK...", "cyan"))
            T_world_tip_ref = internal_plant.GetFrameByName(
                "microscope_tip_link"
            ).CalcPoseInWorld(internal_plant_context)
            T_tip_to_cam = T_cam_to_tip.inverse()

            q_roll[0] = q_now.copy()
            q_prev = q_now.copy()

            for i in range(1, N_POSES):
                Rz_i = RotationMatrix.MakeZRotation(i * np.pi / 3)
                R_tip_i = T_world_tip_ref.rotation() @ Rz_i
                T_world_tip_i = RigidTransform(R_tip_i, T_world_tip_ref.translation())
                T_world_cam_i = T_world_tip_i @ T_tip_to_cam

                draw_triad(
                    meshcat,
                    f"roll_pose_{i}",
                    T_world_tip_i,
                    length=0.04,
                    radius=0.001,
                    opacity=0.6,
                )

                rot_i = T_world_cam_i.rotation().matrix()
                pos_i = T_world_cam_i.translation()
                Q_i = kinematics_solver.IK_for_microscope(rot_i, pos_i, psi=elbow_angle)
                Q_i = filter_ik_solutions(
                    station, Q_i, rot_i, pos_i, joint_lower_limits, joint_upper_limits
                )
                if Q_i.shape[0] == 0:
                    print(colored(f"  [pose {i}] FAIL: no IK", "yellow"))
                else:
                    q_roll[i] = kinematics_solver.find_closest_solution(Q_i, q_prev)
                    q_prev = q_roll[i]
                    print(f"  [pose {i}] OK: {np.rad2deg(q_roll[i]).round(2)} deg")

            n_valid = int(np.sum(~np.isnan(q_roll).any(axis=1)))
            print(colored(f"✓ {n_valid}/{N_POSES} poses have valid IK.", "cyan"))
            print(
                colored(
                    "\nAt pose 0. Click on the object centre in the live view,\n"
                    "then press 'Move to Next' to commit and advance.",
                    "green",
                )
            )
            pose_idx = 0
            state = State.AWAITING_CLICK

        elif state == State.AWAITING_CLICK:
            if meshcat.GetButtonClicks("Move to Next") <= num_move_to_next_clicks:
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue
            num_move_to_next_clicks += 1

            # Commit the current click (full-res)
            if _current_click["x"] is not None:
                cx_c = _current_click["x"] * 2
                cy_c = _current_click["y"] * 2
                committed_clicks.append((cx_c, cy_c))
                _current_click["x"] = None
                _current_click["y"] = None
                print(
                    colored(
                        f"  ✓ Pose {pose_idx}: click committed ({cx_c}, {cy_c}) full-res px",
                        "green",
                    )
                )
            else:
                print(
                    colored(
                        f"  ⚠ No click for pose {pose_idx} — point skipped.", "yellow"
                    )
                )

            # Advance to the next pose with valid IK
            pose_idx += 1
            while pose_idx < N_POSES and np.isnan(q_roll[pose_idx]).any():
                print(colored(f"  Skipping pose {pose_idx} (no valid IK).", "yellow"))
                pose_idx += 1

            if pose_idx >= N_POSES:
                print(
                    colored(
                        "✓ All poses done. Returning to default_position...", "green"
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
                print(colored(f"Planning RRT* to roll pose {pose_idx}...", "cyan"))
                move_result["ready"] = False
                move_result["success"] = False
                threading.Thread(
                    target=plan_rrt_star_async,
                    args=(
                        station,
                        q_now,
                        q_roll[pose_idx],
                        vel_limits,
                        acc_limits,
                        move_result,
                    ),
                    daemon=True,
                ).start()
                state = State.COMPUTING_MOVE_TO_POSE

        elif state == State.COMPUTING_MOVE_TO_POSE:
            if not move_result["ready"]:
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue
            if not move_result["success"]:
                print(colored(f"❌ RRT* to pose {pose_idx} failed — skipping.", "red"))
                pose_idx += 1
                while pose_idx < N_POSES and np.isnan(q_roll[pose_idx]).any():
                    pose_idx += 1
                if pose_idx >= N_POSES:
                    print(colored("No more reachable poses. Resetting...", "cyan"))
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
                    print(colored(f"Planning RRT* to roll pose {pose_idx}...", "cyan"))
                    move_result["ready"] = False
                    move_result["success"] = False
                    threading.Thread(
                        target=plan_rrt_star_async,
                        args=(
                            station,
                            q_now,
                            q_roll[pose_idx],
                            vel_limits,
                            acc_limits,
                            move_result,
                        ),
                        daemon=True,
                    ).start()
            else:
                plot_rrt_raw_path_in_meshcat(
                    station,
                    move_result["path"],
                    name="rrt_roll",
                    rgba=Rgba(1.0, 0.4, 0.0, 1.0),
                )
                plot_trajectory_in_meshcat(
                    station,
                    move_result["trajectory"],
                    rgba=Rgba(0, 1, 1, 1),
                    name="traj_roll",
                )
                trajectory_start_time = simulator.get_context().get_time()
                state = State.MOVING_TO_POSE

        elif state == State.MOVING_TO_POSE:
            traj_complete = move_along_trajectory(
                move_result["trajectory"], trajectory_start_time, simulator, station
            )
            if traj_complete:
                print(
                    colored(
                        f"✓ At pose {pose_idx}. Click on object centre, then press 'Move to Next'.",
                        "green",
                    )
                )
                state = State.AWAITING_CLICK

        elif state == State.COMPUTING_RESET:
            if not reset_result["ready"]:
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue
            if not reset_result["success"]:
                print(colored("❌ RRT* reset failed.", "red"))
                state = State.DONE
            else:
                plot_rrt_raw_path_in_meshcat(
                    station,
                    reset_result["path"],
                    name="rrt_reset",
                    rgba=Rgba(1.0, 0.4, 0.0, 1.0),
                )
                plot_trajectory_in_meshcat(
                    station,
                    reset_result["trajectory"],
                    rgba=Rgba(0, 1, 1, 1),
                    name="traj_reset",
                )
                trajectory_start_time = simulator.get_context().get_time()
                state = State.RESETTING

        elif state == State.RESETTING:
            traj_complete = move_along_trajectory(
                reset_result["trajectory"], trajectory_start_time, simulator, station
            )
            if traj_complete:
                print(colored("✓ Reset to default_position.", "green"))
                state = State.DONE

        elif state == State.DONE:
            clicks_arr = (
                np.array(committed_clicks) if committed_clicks else np.zeros((0, 2))
            )
            np.save(save_dir / "committed_clicks.npy", clicks_arr)
            print(
                colored(
                    f"✓ {len(committed_clicks)} clicks saved → {save_dir / 'committed_clicks.npy'}",
                    "cyan",
                )
            )

            if len(committed_clicks) >= 3:
                fit = fit_circle(committed_clicks)
                if fit is not None:
                    fx, fy, fr = fit
                    dx = fx - cx_full
                    dy = fy - cy_full
                    dist = float(np.hypot(dx, dy))
                    print(
                        colored(
                            "\n  ── Circle fit result (full-res pixels) ──", "green"
                        )
                    )
                    print(f"    fitted centre : ({fx:.2f}, {fy:.2f})")
                    print(f"    fitted radius : {fr:.2f}")
                    print(f"    principal pt  : ({cx_full}, {cy_full})")
                    print(f"    offset (dx,dy): ({dx:+.2f}, {dy:+.2f})")
                    print(
                        colored(
                            f"\n  True optical axis is {dist:.2f} px from calibrated principal point.",
                            "cyan",
                        )
                    )
                    np.savez(
                        save_dir / "pivot_result.npz",
                        cx_fit=fx,
                        cy_fit=fy,
                        r_fit=fr,
                        cx_intrinsic=cx_full,
                        cy_intrinsic=cy_full,
                        dx=dx,
                        dy=dy,
                    )
                    print(
                        colored(
                            f"✓ Result saved → {save_dir / 'pivot_result.npz'}", "cyan"
                        )
                    )
                else:
                    print(
                        colored(
                            "  Circle fit returned no solution (degenerate).", "yellow"
                        )
                    )
            else:
                print(
                    colored(
                        f"  Only {len(committed_clicks)} clicks — need ≥ 3 for circle fit.",
                        "yellow",
                    )
                )

            print(colored("\nDone. Press 'Stop Simulation' to exit.", "cyan"))
            break

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)

    # Cleanup
    for btn in [
        "Stop Simulation",
        "Move to Next",
        "Generate Poses",
        "Reset",
        *JOG_BUTTONS,
    ]:
        meshcat.DeleteButton(btn)
    for i in range(7):
        meshcat.DeleteSlider(f"Joint {i+1} (deg)")
    meshcat.DeleteSlider("Current PSI (deg)")

    cv2.destroyAllWindows()

    if not no_cam and camera is not None:
        _capture_stop.set()
        _capture_thread.join(timeout=5)
        camera.release()
        print(colored("✓ Camera shut down cleanly.", "cyan"))

    print(colored("Session ended.", "cyan"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pivot-point calibration to find true optical axis offset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python robot_scan/pivot_calibration.py --use_hardware
  python robot_scan/pivot_calibration.py --no_cam
        """,
    )
    parser.add_argument(
        "--use_hardware", action="store_true", help="Connect to real iiwa hardware."
    )
    parser.add_argument("--no_cam", action="store_true", help="Disable camera.")
    parser.add_argument(
        "--camera_source",
        type=int,
        default=CAMERA_SOURCE,
        help="Camera device number.",
    )
    args = parser.parse_args()
    main(
        use_hardware=args.use_hardware,
        no_cam=args.no_cam,
        camera_source=args.camera_source,
    )
