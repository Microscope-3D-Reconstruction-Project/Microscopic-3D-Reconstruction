"""
microscope_scripts/eye_in_hand_calibration.py

Offline eye-in-hand calibration. Reads data collected by
robot_scan/collect_calib_data.py and solves for T_cam_to_flange — the rigid
transform from the camera optical frame to the iiwa_link_ee_kuka media flange.

Expected folder layout:
    <calib_dir>/
        images/
            camera_calib_intrinsic.json
            frame_00000.png
            frame_00001.png
            ...
        poses/
            frame_00000.npy   ← 4x4 T_flange_in_world at capture time
            frame_00001.npy
            ...

Usage:
    python microscope_scripts/eye_in_hand_calibration.py \\
        --calib_dir microscope-data/calibrations/20240501_123456 \\
        --checker_size 2.0 \\
        --corners_h 6 \\
        --corners_w 9
"""

import argparse
import json
import sys

from pathlib import Path

import cv2
import numpy as np

METHODS = {
    # "TSAI":      cv2.CALIB_HAND_EYE_TSAI,
    # "PARK":      cv2.CALIB_HAND_EYE_PARK,
    # "HORAUD":    cv2.CALIB_HAND_EYE_HORAUD,
    # "ANDREFF":   cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def load_intrinsics(images_dir: Path):
    json_path = images_dir / "camera_calib_intrinsic.json"
    if not json_path.exists():
        sys.exit(f"ERROR: intrinsics not found at {json_path}")
    with open(json_path) as f:
        calib = json.load(f)
    K = np.array(calib["camera_matrix"], dtype=np.float64)
    dist = np.array(calib["distortion_coefficients"], dtype=np.float64)
    return K, dist


def build_object_points(corners_h: int, corners_w: int, checker_size_mm: float):
    objp = np.zeros((corners_h * corners_w, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:corners_h, 0:corners_w].T.reshape(-1, 2)
    objp *= checker_size_mm / 1000.0  # convert to metres to match Drake's flange poses
    return objp


def reprojection_error(
    R_cam2flange,
    t_cam2flange,
    R_flange2base_list,
    t_flange2base_list,
    R_target2cam_list,
    t_target2cam_list,
    objpoints,
    K,
    dist,
):
    errors = []
    for R_f2b, t_f2b, R_t2c, t_t2c, objp in zip(
        R_flange2base_list,
        t_flange2base_list,
        R_target2cam_list,
        t_target2cam_list,
        objpoints,
    ):
        # Transform: target → camera → flange → base
        R_t2f = R_cam2flange @ R_t2c
        t_t2f = R_cam2flange @ t_t2c + t_cam2flange
        R_t2b = R_f2b @ R_t2f
        t_t2b = R_f2b @ t_t2f + t_f2b

        # Project object points through the full chain back to image
        rvec_t2c, _ = cv2.Rodrigues(R_t2c)
        imgpts, _ = cv2.projectPoints(objp, rvec_t2c, t_t2c, K, dist)
        # Reproject from base frame via inverse of flange2base
        R_b2f = R_f2b.T
        t_b2f = -R_b2f @ t_f2b
        R_b2c = R_cam2flange.T @ R_b2f  # cam2flange inverse
        # (reprojection error measured purely in camera frame via solvePnP residual)
        errors.append(np.linalg.norm(imgpts))  # placeholder — see below

    # Simpler and standard: use PnP reprojection residuals per image
    return None  # computed per-image below


def compute_pnp_reprojection(objp, imgp, rvec, tvec, K, dist):
    projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    return float(
        np.mean(np.linalg.norm(imgp.reshape(-1, 2) - projected.reshape(-1, 2), axis=1))
    )


def main(calib_dir: Path, checker_size: float, corners_h: int, corners_w: int):
    calib_dir = calib_dir.resolve()
    images_dir = calib_dir / "images"
    poses_dir = calib_dir / "poses"

    print(f"calib_dir  : {calib_dir}")
    print(f"images_dir : {images_dir}  (exists: {images_dir.exists()})")
    print(f"poses_dir  : {poses_dir}   (exists: {poses_dir.exists()})")

    K, dist = load_intrinsics(images_dir)
    print(f"\nIntrinsics loaded:")
    print(f"  K:\n{K}")
    print(f"  dist: {dist.flatten()}")

    objp = build_object_points(corners_h, corners_w, checker_size)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    image_files = sorted(images_dir.glob("frame_*.png"))
    print(f"\nFound {len(image_files)} frame images.")

    R_flange2base_list = []
    t_flange2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []
    objpoints_list = []
    pnp_errors = []
    skipped = 0

    for img_path in image_files:
        stem = img_path.stem
        pose_path = poses_dir / f"{stem}.npy"

        if not pose_path.exists():
            print(f"  SKIP {stem}: no matching pose file")
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  SKIP {stem}: could not read image")
            skipped += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (corners_h, corners_w), None)
        if not found:
            print(f"  SKIP {stem}: checkerboard not detected")
            skipped += 1
            continue

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        ret, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist)
        if not ret:
            print(f"  SKIP {stem}: solvePnP failed")
            skipped += 1
            continue

        R_t2c, _ = cv2.Rodrigues(rvec)
        t_t2c = tvec  # shape (3,1)

        pose = np.load(str(pose_path))  # 4x4 T_flange_in_world
        R_f2b = pose[:3, :3]
        t_f2b = pose[:3, 3:4]  # shape (3,1)

        pnp_err = compute_pnp_reprojection(objp, corners2, rvec, tvec, K, dist)
        pnp_errors.append(pnp_err)

        R_flange2base_list.append(R_f2b)
        t_flange2base_list.append(t_f2b)
        R_target2cam_list.append(R_t2c)
        t_target2cam_list.append(t_t2c)
        objpoints_list.append(objp)

    n = len(R_flange2base_list)
    print(f"\nUsable pairs: {n}  |  skipped: {skipped}")
    if n < 3:
        sys.exit(
            "ERROR: need at least 3 usable image/pose pairs for hand-eye calibration."
        )

    print(
        f"Mean PnP reprojection error across used images: {np.mean(pnp_errors):.3f} px"
    )

    print("\n" + "=" * 60)
    print("  Hand-eye calibration results (cv2.calibrateHandEye)")
    print("=" * 60)

    best_method = None
    best_T = None
    best_err = np.inf

    n_pairs = len(R_flange2base_list)

    for name, method in METHODS.items():
        try:
            R_c2f, t_c2f = cv2.calibrateHandEye(
                R_flange2base_list,
                t_flange2base_list,
                R_target2cam_list,
                t_target2cam_list,
                method=method,
            )
        except Exception as e:
            print(f"  {name:12s}: FAILED ({e})")
            continue

        T_c2f = np.eye(4)
        T_c2f[:3, :3] = R_c2f
        T_c2f[:3, 3] = t_c2f.flatten()

        # Quality metric: AX = XB residual over all consecutive pose pairs.
        # A = relative flange motion (world frame), B = relative target motion
        # (camera frame). A good X minimises ||AX - XB||_F across all pairs.
        ax_xb_rot_errors = []
        ax_xb_trans_errors_mm = []
        for i in range(n_pairs - 1):
            # Relative flange motion A: pose i → pose i+1 in world frame
            T_f2b_i = np.eye(4)
            T_f2b_i[:3, :3] = R_flange2base_list[i]
            T_f2b_i[:3, 3:] = t_flange2base_list[i]
            T_f2b_j = np.eye(4)
            T_f2b_j[:3, :3] = R_flange2base_list[i + 1]
            T_f2b_j[:3, 3:] = t_flange2base_list[i + 1]
            A = np.linalg.inv(T_f2b_i) @ T_f2b_j

            # AX=XB form (OpenCV convention):
            #   inv(T_g2b_i) @ T_g2b_j @ X = X @ T_t2c_i @ inv(T_t2c_j)
            # so B = T_t2c_i @ inv(T_t2c_j), NOT T_t2c_j @ inv(T_t2c_i).
            T_t2c_i = np.eye(4)
            T_t2c_i[:3, :3] = R_target2cam_list[i]
            T_t2c_i[:3, 3:] = t_target2cam_list[i]
            T_t2c_j = np.eye(4)
            T_t2c_j[:3, :3] = R_target2cam_list[i + 1]
            T_t2c_j[:3, 3:] = t_target2cam_list[i + 1]
            B = T_t2c_i @ np.linalg.inv(T_t2c_j)

            residual = A @ T_c2f - T_c2f @ B
            ax_xb_rot_errors.append(np.linalg.norm(residual[:3, :3], "fro"))
            ax_xb_trans_errors_mm.append(np.linalg.norm(residual[:3, 3]) * 1000.0)

        mean_rot_residual = float(np.mean(ax_xb_rot_errors))
        mean_trans_residual_mm = float(np.mean(ax_xb_trans_errors_mm))
        t_norm = np.linalg.norm(t_c2f)

        if mean_rot_residual < best_err:
            best_err = mean_rot_residual
            best_method = name
            best_T = T_c2f

        print(
            f"  {name:12s}:  AX=XB rot residual = {mean_rot_residual:.5f}  |  "
            f"trans residual = {mean_trans_residual_mm:.4f} mm  |  |t| = {t_norm*1000:.2f} mm"
        )
        print(f"               R =\n{np.round(R_c2f, 5)}")
        print(f"               t = {np.round(t_c2f.flatten()*1000, 3)} mm")

    print("\n" + "=" * 60)
    if best_T is None:
        sys.exit("ERROR: all methods failed.")

    R_best = best_T[:3, :3]
    t_best = best_T[:3, 3]

    print(
        f"  Recommended method: {best_method}  (smallest AX=XB rot residual = {best_err:.5f})"
    )
    print(f"\n  T_cam_to_flange (4x4):\n{np.round(best_T, 6)}")

    print("\n" + "=" * 60)
    print("  Paste into robot_scan/demo_config.py:")
    print("=" * 60)
    r_rows = ",\n                    ".join(
        "[" + ", ".join(f"{v:.8f}" for v in row) + "]" for row in R_best
    )
    t_vals = ", ".join(f"{v:.8f}" for v in t_best)
    print(
        f"""
T_CAM_TO_FLANGE = RigidTransform(
    RotationMatrix(np.array([
                    {r_rows}
                   ])),
    np.array([{t_vals}]),
)
"""
    )

    out_path = calib_dir / "T_cam_to_flange.npy"
    np.save(str(out_path), best_T)
    print(f"  Saved T_cam_to_flange → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Eye-in-hand calibration: solve T_cam_to_flange from collected data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python microscope_scripts/eye_in_hand_calibration.py \\
      --calib_dir microscope-data/calibrations/20240501_123456 \\
      --checker_size 2.0
        """,
    )
    parser.add_argument("--calib_dir", type=Path, required=True)
    parser.add_argument(
        "--checker_size",
        type=float,
        default=2.0,
        help="Checkerboard square size in mm (default: 2.0)",
    )
    parser.add_argument(
        "--corners_h",
        type=int,
        default=6,
        help="Internal corners along height (default: 6)",
    )
    parser.add_argument(
        "--corners_w",
        type=int,
        default=9,
        help="Internal corners along width (default: 9)",
    )
    args = parser.parse_args()
    main(args.calib_dir, args.checker_size, args.corners_h, args.corners_w)
