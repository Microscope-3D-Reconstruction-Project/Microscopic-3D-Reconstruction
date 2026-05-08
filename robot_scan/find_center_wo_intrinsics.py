"""
robot_scan/find_center_wo_intrinsics.py

Moves the robot to 5 predefined hemisphere viewpoints. At each viewpoint the
user jogs the robot (camera frame) until the object appears centred on the
crosshair, then presses "Move to Next" to record the optical-axis ray from the
current camera pose. After all 5 viewpoints, the rays are intersected via
least-squares to estimate the 3D object centre.

The crosshair is drawn at (cx, cy) from the calibrated intrinsics, so the
user knows exactly where to aim. Assuming the user centres the object there,
the object lies on the optical axis ray — no pixel click needed.

Usage:
    python robot_scan/find_center_wo_intrinsics.py --use_hardware
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
    Box,
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
from utils.jog import try_jog_tip
from utils.kuka_geo_kin import KinematicsSolver
from utils.planning import (
    compute_hemisphere_traj_async,
    move_along_trajectory,
    plot_trajectory_in_meshcat,
)
from utils.plotting import plot_hemisphere_waypoints
from utils.ray_reconstruction import (
    least_squares_ray_intersection,
    load_intrinsics,
    print_ray,
    visualize_predicted_center,
    visualize_ray,
)
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
    AWAITING_HEMISPHERE_CONFIRM = auto()
    MOVING_ALONG_HEMISPHERE = auto()
    PLANNING_RRT_FALLBACK = auto()
    COMPUTING_RRT_FALLBACK = auto()
    AWAITING_RRT_CONFIRM = auto()
    MOVING_ALONG_RRT = auto()
    REVERTING_TO_SCAN_POSE = auto()
    AWAITING_JOG_CONFIRM = auto()
    COMPUTING_RESET = auto()
    RESETTING = auto()
    DONE = auto()


def _animate_configs(configs, station, station_context, simulator, meshcat):
    """Animate robot through configs forward then in reverse (for previewing)."""
    for q in list(configs) + list(reversed(configs)):
        station.GetInputPort("iiwa.position").FixValue(station_context, q)
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)
        for i, qi in enumerate(q):
            meshcat.SetSliderValue(f"Joint {i+1} (deg)", round(np.rad2deg(qi), 1))


def main(
    use_hardware: bool,
    no_cam: bool = False,
    start_idx: int = 0,
    live_view: bool = True,
    camera_source: int = CAMERA_SOURCE,
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

    # Load intrinsics for crosshair position
    camera_K = load_intrinsics()
    cx_full = int(camera_K[0, 2])
    cy_full = int(camera_K[1, 2])
    cx_disp = cx_full // 2
    cy_disp = cy_full // 2

    print(colored("=" * 60, "cyan"))
    print(colored("  find_center_wo_intrinsics — configuration", "cyan"))
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
    print(colored("  Intrinsics (crosshair):", "white"))
    print(f"    cx = {cx_full}  cy = {cy_full}  →  display ({cx_disp}, {cy_disp})")
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
    hemisphere_axis = np.array([-1.0, 0.0, 0.0])  # angle=0 always

    # ==================================================================
    # Outputs setup
    # ==================================================================
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    scans_base = (
        Path(__file__).parent.parent / "microscope-data" / "object_centering" / date_str
    )
    scans_base.mkdir(parents=True, exist_ok=True)
    print(colored(f"✓ Session outputs → {scans_base}", "cyan"))

    # ==================================================================
    # Waypoint generation (5 fixed centering poses)
    # ==================================================================
    hemisphere_waypoints = generate_centering_waypoints(
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
    )
    labels = ["straight", "up", "down", "right", "left"]
    print(colored("\n5 centering waypoints:", "cyan"))
    for i, (wp, label) in enumerate(zip(hemisphere_waypoints, labels)):
        # print(f"  [{i}] {label}: {np.round(wp.rotation().matrix(), 4)}, {np.round(wp.translation(), 4)}")
        print(f"  [{i}] {label}:")
        print("Rotation: ")
        print(np.round(wp.rotation().matrix(), 4))
        print("Translation: ")
        print(np.round(wp.translation(), 4))

    plot_hemisphere_waypoints(
        hemisphere_waypoints,
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
        output_path=outputs_dir / "centering_waypoints.png",
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

    # camera_frame = station.get_internal_plant().GetFrameByName("camera_link")
    # AddFrameTriadIllustration(
    #     scene_graph=station.internal_station.get_scene_graph(),
    #     plant=station.get_internal_plant(),
    #     frame=camera_frame,
    #     length=0.1,
    #     radius=0.002,
    #     name="camera_link",
    # )

    # flange_frame = station.get_internal_plant().GetFrameByName("iiwa_link_ee_kuka")
    # AddFrameTriadIllustration(
    #     scene_graph=station.internal_station.get_scene_graph(),
    #     plant=station.get_internal_plant(),
    #     frame=flange_frame,
    #     length=0.1,
    #     radius=0.002,
    #     name="camera_link",
    # )

    # Calibrated optical center frame — auto-updates with robot motion in Meshcat
    # because optical_center is a model link welded to the flange via the DMD.
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
    meshcat.AddButton("Move to Next")
    meshcat.AddButton("Preview RRT* Raw")
    meshcat.AddButton("Preview RRT* Smooth")
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
    _capture_thread = None

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

        def _capture_loop():
            nonlocal _latest_frame
            while not _capture_stop.is_set():
                ret, frame = camera.read()
                if ret:
                    with _latest_frame_lock:
                        _latest_frame = frame

        _capture_thread = threading.Thread(target=_capture_loop, daemon=True)
        _capture_thread.start()
        print(colored("✓ Camera thread started", "cyan"))
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

    collected_rays: list[tuple[np.ndarray, np.ndarray]] = []

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
    rrt_result = {"ready": False, "success": False, "trajectory": None, "path": None}
    reset_result = {"ready": False, "success": False, "trajectory": None, "path": None}

    hemisphere_trajectory = None

    num_move_to_next_clicks = 0
    num_preview_raw_clicks = 0
    num_preview_smooth_clicks = 0
    num_reset_clicks = 0
    jog_click_counts = {name: 0 for name in JOG_BUTTONS}

    MOVING_STATES = {
        State.COMPUTING_MOVE_TO_START,
        State.MOVING_TO_START,
        State.COMPUTING_IKS,
        State.PLANNING_RRT_FALLBACK,
        State.COMPUTING_RRT_FALLBACK,
        State.MOVING_ALONG_HEMISPHERE,
        State.MOVING_ALONG_RRT,
        State.REVERTING_TO_SCAN_POSE,
        State.COMPUTING_RESET,
        State.RESETTING,
    }
    _jog_target_q: np.ndarray | None = None
    _q_at_scan: np.ndarray | None = None
    _pending_traj_type: str = ""
    _at_scan_pose = False

    print(colored("\nReady. Press 'Move to Next' in Meshcat to begin.", "cyan"))
    print(
        colored(
            "  At each viewpoint: jog until object is on the crosshair, then press 'Move to Next'.",
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

        # Live camera view with crosshair at (cx, cy)
        if live_view and not no_cam and _latest_frame_lock is not None:
            with _latest_frame_lock:
                lf = _latest_frame.copy() if _latest_frame is not None else None
            if lf is not None:
                half = (lf.shape[1] // 2, lf.shape[0] // 2)
                disp = cv2.resize(lf, half)
                cv2.drawMarker(
                    disp,
                    (cx_disp, cy_disp),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    40,
                    4,
                )
                cv2.imshow("Live View", disp)
                cv2.waitKey(1)

        # ------------------------------------------------------------------
        # Jog buttons — active in any non-moving state
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
            if meshcat.GetButtonClicks("Move to Next") <= num_move_to_next_clicks:
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
                continue
            num_move_to_next_clicks += 1

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
                _at_scan_pose = True
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
        elif state == State.AWAITING_HEMISPHERE_CONFIRM:
            execute = meshcat.GetButtonClicks("Move to Next") > num_move_to_next_clicks
            if execute:
                num_move_to_next_clicks += 1
                if _q_at_scan is not None and np.any(
                    np.abs(q_now - _q_at_scan) > 0.005
                ):
                    print(
                        colored(
                            "  Robot was jogged — reverting to scan pose first...",
                            "cyan",
                        )
                    )
                    station.GetInputPort("iiwa.position").FixValue(
                        station_context, _q_at_scan
                    )
                    _pending_traj_type = "hemisphere"
                    state = State.REVERTING_TO_SCAN_POSE
                else:
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
            print(
                colored(
                    "  ✓ RRT*-Connect found path.\n"
                    "    Press 'Preview RRT* Raw', 'Preview RRT* Smooth', or 'Move to Next'.",
                    "green",
                )
            )
            state = State.AWAITING_RRT_CONFIRM

        # ------------------------------------------------------------------
        elif state == State.AWAITING_RRT_CONFIRM:
            preview_raw = (
                meshcat.GetButtonClicks("Preview RRT* Raw") > num_preview_raw_clicks
            )
            preview_smooth = (
                meshcat.GetButtonClicks("Preview RRT* Smooth")
                > num_preview_smooth_clicks
            )
            execute = meshcat.GetButtonClicks("Move to Next") > num_move_to_next_clicks

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
                num_move_to_next_clicks += 1
                if _q_at_scan is not None and np.any(
                    np.abs(q_now - _q_at_scan) > 0.005
                ):
                    print(
                        colored(
                            "  Robot was jogged — reverting to scan pose first...",
                            "cyan",
                        )
                    )
                    station.GetInputPort("iiwa.position").FixValue(
                        station_context, _q_at_scan
                    )
                    _pending_traj_type = "rrt"
                    state = State.REVERTING_TO_SCAN_POSE
                else:
                    trajectory_start_time = simulator.get_context().get_time()
                    print(colored("  Executing RRT* trajectory...", "green"))
                    state = State.MOVING_ALONG_RRT

        # ------------------------------------------------------------------
        elif state == State.MOVING_ALONG_RRT:
            traj = rrt_result["trajectory"]
            traj_complete = move_along_trajectory(
                traj, trajectory_start_time, simulator, station
            )
            if traj_complete:
                _q_at_scan = q_now.copy()
                curr_idx = scan_idx
                _at_scan_pose = True
                state = State.AWAITING_JOG_CONFIRM

        # ------------------------------------------------------------------
        elif state == State.MOVING_ALONG_HEMISPHERE:
            traj = hemisphere_trajectory
            traj_complete = move_along_trajectory(
                traj, trajectory_start_time, simulator, station
            )
            if traj_complete:
                _q_at_scan = q_now.copy()
                curr_idx = scan_idx
                _at_scan_pose = True
                state = State.AWAITING_JOG_CONFIRM

        # ------------------------------------------------------------------
        elif state == State.REVERTING_TO_SCAN_POSE:
            station.GetInputPort("iiwa.position").FixValue(station_context, _q_at_scan)
            if np.all(np.abs(q_now - _q_at_scan) < 0.01):
                print(
                    colored("  ✓ Back at scan pose. Executing trajectory...", "green")
                )
                trajectory_start_time = simulator.get_context().get_time()
                state = (
                    State.MOVING_ALONG_HEMISPHERE
                    if _pending_traj_type == "hemisphere"
                    else State.MOVING_ALONG_RRT
                )

        # ------------------------------------------------------------------
        elif state == State.AWAITING_JOG_CONFIRM:
            # Jog until object is centred on crosshair, then record
            if meshcat.GetButtonClicks("Move to Next") > num_move_to_next_clicks:
                num_move_to_next_clicks += 1
                T_world_cam = internal_plant.GetFrameByName(
                    "camera_link"
                ).CalcPoseInWorld(internal_plant_context)
                origin = T_world_cam.translation()
                # Optical axis = camera +z rotated to world frame
                direction = T_world_cam.rotation().matrix()[:, 2]
                direction = direction / np.linalg.norm(direction)
                print_ray(origin, direction, scan_idx=curr_idx)
                meshcat.Delete(f"ray_{curr_idx}")
                visualize_ray(meshcat, origin, direction, name=f"ray_{curr_idx}")
                collected_rays.append((origin.copy(), direction.copy()))
                scan_idx += 1
                if scan_idx >= n:
                    print(
                        colored(
                            "✓ All rays recorded. Returning to default_position...",
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
                _at_scan_pose = False
                _q_at_scan = None
                state = State.DONE if scan_idx >= n else State.WAITING_TO_GO_TO_START

        # ------------------------------------------------------------------
        elif state == State.DONE:
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

            if collected_rays:
                origins = np.array([ray[0] for ray in collected_rays])
                directions = np.array([ray[1] for ray in collected_rays])
                rays_path = scans_base / "rays.npy"
                np.save(rays_path, np.stack([origins, directions], axis=1))
                print(
                    colored(
                        f"✓ Rays saved → {rays_path}  ({len(collected_rays)} rays)",
                        "cyan",
                    )
                )

                if len(collected_rays) >= 2:
                    predicted = least_squares_ray_intersection(origins, directions)
                    print(
                        colored(
                            f"\n  Predicted object centre (LS):\n"
                            f"    x = {predicted[0]:.6f}\n"
                            f"    y = {predicted[1]:.6f}\n"
                            f"    z = {predicted[2]:.6f}",
                            "green",
                        )
                    )
                    np.save(scans_base / "predicted_center.npy", predicted)
                    print(
                        colored(
                            f"✓ Predicted center saved → {scans_base / 'predicted_center.npy'}",
                            "cyan",
                        )
                    )
                    visualize_predicted_center(meshcat, predicted)
                else:
                    print(colored("  Need ≥ 2 rays for LS intersection.", "yellow"))
            else:
                print(
                    colored("  No rays collected — skipping LS intersection.", "yellow")
                )

            print(colored("\nScan complete. Press 'Stop Simulation' to exit.", "cyan"))
            break

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)

    # ==================================================================
    # Cleanup
    # ==================================================================
    for btn in [
        "Stop Simulation",
        "Move to Next",
        "Preview RRT* Raw",
        "Preview RRT* Smooth",
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
        _capture_thread.join(timeout=5)
        camera.release()
        print(colored("✓ Camera shut down cleanly.", "cyan"))

    print(colored("Session ended.", "cyan"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find object centre by shooting optical-axis rays from 5 centering viewpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python robot_scan/find_center_wo_intrinsics.py --use_hardware
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

    args = parser.parse_args()
    main(
        use_hardware=args.use_hardware,
        no_cam=args.no_cam,
        start_idx=args.start_idx,
        live_view=not args.no_live_view,
        camera_source=args.camera_source,
    )
