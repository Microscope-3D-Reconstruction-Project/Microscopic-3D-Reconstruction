import json
import os
import shutil
import sqlite3
import time

from dataclasses import dataclass
from pathlib import Path

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pycolmap

from omegaconf import DictConfig
from pycolmap import logging
from scipy.spatial.transform import Rotation as R
from visualizers.feature_viewer import FeatureViewer


@dataclass
class ScanPaths:
    # feature extraction / matching
    feat_images_path: Path  # input: images directory
    feat_masks_path: Path  # input: masks directory
    feat_database_path: Path  # output: database.db
    # reconstruction
    poses_json_path: Path  # input: poses.json written by focus_stack
    intrinsics_json_path: Path  # input: intrinsics.json written by focus_stack
    recon_images_path: Path  # input: images directory
    recon_database_path: Path  # input: database.db
    model_dir: Path  # output: sparse model directory
    recon_output_dir: Path  # output: parent dir (for temp files)
    colmap_dir: Path  # output: colmap summary/timings directory


def create_scan_paths(cfg: DictConfig) -> ScanPaths:
    feat_out_dir = Path(cfg.feat_extract_match.output_paths.output_dir)
    feat_out_dir.mkdir(parents=True, exist_ok=True)

    recon_out_dir = Path(cfg.reconstruction.output_paths.output_dir)
    recon_out_dir.mkdir(parents=True, exist_ok=True)

    colmap_dir = recon_out_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    return ScanPaths(
        feat_images_path=Path(cfg.feat_extract_match.input_paths.images_dir),
        feat_masks_path=Path(cfg.feat_extract_match.input_paths.masks_dir),
        feat_database_path=feat_out_dir
        / cfg.feat_extract_match.output_paths.database_filename,
        poses_json_path=Path(cfg.reconstruction.input_paths.poses_json),
        intrinsics_json_path=Path(
            cfg.reconstruction.input_paths.prior_intrinsics_filepath
        ),
        recon_images_path=Path(cfg.reconstruction.input_paths.images_dir),
        recon_database_path=Path(cfg.reconstruction.input_paths.database_filepath),
        model_dir=recon_out_dir / cfg.reconstruction.output_paths.model_dirname,
        recon_output_dir=recon_out_dir,
        colmap_dir=colmap_dir,
    )


def _init_timings(colmap_dir: Path) -> None:
    timings_path = colmap_dir / "timings.json"
    if not timings_path.exists():
        with open(timings_path, "w") as f:
            json.dump(
                {"feature_extraction": 0, "feature_matching": 0, "reconstruction": 0},
                f,
                indent=2,
            )


def _write_timing_entry(colmap_dir: Path, key: str, value: float) -> None:
    timings_path = colmap_dir / "timings.json"
    with open(timings_path) as f:
        timings = json.load(f)
    timings[key] = value
    with open(timings_path, "w") as f:
        json.dump(timings, f, indent=2)


def read_poses_dict(poses_json_path, is_camera_to_world=False, as_quaternion=False):
    """
    Reads robot poses from a poses.json file written by run_focus_stack.

    Args:
        poses_json_path (Path | str): Path to poses.json.
        is_camera_to_world (bool): If True, the stored poses are Camera-to-World and
                                   will be inverted to World-to-Camera before returning.
        as_quaternion (bool): If True, each pose is returned as a dict
                              {'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz'}.
                              If False, each pose is returned as a 4x4 numpy array.

    Returns:
        dict: Mapping image names (e.g. 'scan00.jpg') to either 4x4 numpy arrays
              or quaternion+translation dicts, depending on as_quaternion.
    """
    poses = {}
    print(
        f"Reading poses as {'quaternion' if as_quaternion else 'matrix'} "
        f"({'Camera-to-World' if is_camera_to_world else 'World-to-Camera'}) "
        f"from {poses_json_path}..."
    )
    with open(poses_json_path) as f:
        raw = json.load(f)
    for image_name, matrix_list in raw.items():
        matrix = np.array(matrix_list)
        if is_camera_to_world:
            matrix = np.linalg.inv(matrix)
        if as_quaternion:
            qx, qy, qz, qw = R.from_matrix(matrix[:3, :3]).as_quat()
            tx, ty, tz = matrix[:3, 3]
            poses[image_name] = {
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "tx": tx,
                "ty": ty,
                "tz": tz,
            }
        else:
            poses[image_name] = matrix
    return poses


def read_intrinsics_json(intrinsics_json_path):
    """Read camera intrinsics from an intrinsics.json file.

    Args:
        intrinsics_json_path (Path | str): Path to intrinsics.json.

    Returns:
        tuple: OPENCV camera params (fx, fy, cx, cy, k1, k2, p1, p2).
    """
    print(f"Reading intrinsics from {intrinsics_json_path}...")
    with open(intrinsics_json_path) as f:
        data = json.load(f)

    cam = data["camera_matrix"]
    fx, fy = cam[0][0], cam[1][1]
    cx, cy = cam[0][2], cam[1][2]

    dist = data.get("distortion_coefficients", [[0, 0, 0, 0, 0]])
    coeffs = dist[0] if dist else [0, 0, 0, 0, 0]
    # Pad to at least 5 coefficients
    while len(coeffs) < 5:
        coeffs.append(0.0)
    k1, k2, p1, p2 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]

    print(f"  fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    print(f"  k1={k1}, k2={k2}, p1={p1}, p2={p2}")
    return (fx, fy, cx, cy, k1, k2, p1, p2)


def apply_prior_intrinsics_to_database(db_path, intrinsics_params):
    """Update camera intrinsics in a COLMAP database.

    Overwrites the params blob for every camera in the database with the
    given OPENCV parameters.

    Args:
        db_path (Path | str): Path to the COLMAP database.db.
        intrinsics_params (tuple): (fx, fy, cx, cy, k1, k2, p1, p2).
    """
    params_array = np.array(intrinsics_params, dtype=np.float64)
    params_blob = params_array.tobytes()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT camera_id FROM cameras")
    camera_ids = [row[0] for row in cursor.fetchall()]

    for cam_id in camera_ids:
        cursor.execute(
            "UPDATE cameras SET params = ?, prior_focal_length = 1 WHERE camera_id = ?",
            (params_blob, cam_id),
        )
        print(f"  Updated camera {cam_id} intrinsics in database.")

    conn.commit()
    conn.close()
    print(f"Applied prior intrinsics to {len(camera_ids)} camera(s) in {db_path}.")


def create_reference_reconstruction(
    db_path, output_dir, pose_dict, intrinsics_params=None
):
    """
    Creates a COLMAP text model from the database and known poses.

    Args:
        db_path: Path to the COLMAP database.db
        output_dir: Directory to write cameras.txt, images.txt, points3D.txt
        pose_dict (dict): Mapping image names to quaternion+translation dicts
                          {'qw','qx','qy','qz','tx','ty','tz'}, as returned by
                          read_poses_from_scans(as_quaternion=True).
        intrinsics_params (tuple | None): Optional OPENCV params
                          (fx, fy, cx, cy, k1, k2, p1, p2). If provided,
                          overrides camera params read from the DB.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT camera_id, model, width, height, params FROM cameras")
    db_cameras = cursor.fetchall()

    cursor.execute("SELECT image_id, name, camera_id FROM images")
    db_images = cursor.fetchall()
    conn.close()

    # Write cameras.txt
    with open(output_dir / "cameras.txt", "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam_id, model_id, width, height, params_blob in db_cameras:
            if intrinsics_params is not None:
                params = np.array(intrinsics_params, dtype=np.float64)
            else:
                params = np.frombuffer(params_blob, dtype=np.float64)
            model_name = pycolmap.CameraModelId(model_id).name
            params_str = " ".join([str(p) for p in params])
            f.write(f"{cam_id} {model_name} {width} {height} {params_str}\n")

    # Write images.txt
    with open(output_dir / "images.txt", "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for image_id, name, camera_id in db_images:
            pose = pose_dict[name]
            qw, qx, qy, qz = pose["qw"], pose["qx"], pose["qy"], pose["qz"]
            tx, ty, tz = pose["tx"], pose["ty"], pose["tz"]
            f.write(
                f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {name}\n"
            )
            f.write("\n")

    # Write empty points3D.txt
    with open(output_dir / "points3D.txt", "w") as f:
        pass


def create_dummy_model(output_dir):
    """
    Creates an empty COLMAP text model (no cameras, images, or points).
    Passing this as input_path to incremental_mapping forces COLMAP to write
    the result directly into output_path instead of numbered subdirectories.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with open(output_dir / "cameras.txt", "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    with open(output_dir / "images.txt", "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    with open(output_dir / "points3D.txt", "w") as f:
        pass


def run_feature_extraction_and_matching(paths: ScanPaths, cfg: DictConfig) -> None:
    # Always delete an existing database and re-extract when this stage runs.
    if paths.feat_database_path.exists():
        print(f"Deleting existing database: {paths.feat_database_path}")
        paths.feat_database_path.unlink()

    print(f"Extracting features from {paths.feat_images_path}...")
    sift_options = pycolmap.SiftExtractionOptions()
    sift_options.estimate_affine_shape = (
        cfg.feat_extract_match.extraction.estimate_affine_shape
    )
    sift_options.max_num_features = cfg.feat_extract_match.extraction.max_num_features

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift = sift_options
    extraction_options.num_threads = cfg.feat_extract_match.extraction.num_threads

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = cfg.feat_extract_match.extraction.camera_model
    reader_options.mask_path = paths.feat_masks_path

    extraction_start_time = time.time()
    pycolmap.extract_features(
        database_path=paths.feat_database_path,
        image_path=paths.feat_images_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=cfg.feat_extract_match.extraction.device,
    )
    extraction_end_time = time.time()
    extraction_duration = extraction_end_time - extraction_start_time
    print(f"Feature extraction took {extraction_duration:.2f} seconds")
    _write_timing_entry(paths.colmap_dir, "feature_extraction", extraction_duration)

    # Load prior intrinsics if enabled
    intrinsics_params = None
    if cfg.reconstruction.load_prior_intrinsics:
        intrinsics_params = read_intrinsics_json(paths.intrinsics_json_path)
        apply_prior_intrinsics_to_database(paths.feat_database_path, intrinsics_params)

    feature_matching_options = pycolmap.FeatureMatchingOptions()
    feature_matching_options.use_gpu = cfg.feat_extract_match.matching.use_gpu
    feature_matching_options.num_threads = cfg.feat_extract_match.matching.num_threads
    feature_matching_options.guided_matching = (
        cfg.feat_extract_match.matching.guided_matching
    )

    verification_options = pycolmap.TwoViewGeometryOptions()
    verification_options.ransac.random_seed = cfg.random_seed

    matching_start_time = time.time()
    pycolmap.match_exhaustive(
        database_path=paths.feat_database_path,
        matching_options=feature_matching_options,
        verification_options=verification_options,
        device="cuda" if cfg.feat_extract_match.matching.use_gpu else "cpu",
    )
    matching_end_time = time.time()
    matching_duration = matching_end_time - matching_start_time
    print(f"Feature matching took {matching_duration:.2f} seconds")
    _write_timing_entry(paths.colmap_dir, "feature_matching", matching_duration)


def run_triangulation(paths: ScanPaths, cfg: DictConfig) -> Path:
    if paths.model_dir.exists():
        shutil.rmtree(paths.model_dir)
    paths.model_dir.mkdir(parents=True)

    # Load prior intrinsics if enabled
    intrinsics_params = None
    if cfg.reconstruction.load_prior_intrinsics:
        intrinsics_params = read_intrinsics_json(paths.intrinsics_json_path)
        apply_prior_intrinsics_to_database(paths.recon_database_path, intrinsics_params)

    # Use known robot poses: lock cameras, triangulate points.
    poses = read_poses_dict(
        paths.poses_json_path, is_camera_to_world=True, as_quaternion=True
    )

    reference_model_path = paths.recon_output_dir / "_reference_model"
    if reference_model_path.exists():
        shutil.rmtree(reference_model_path)
    reference_model_path.mkdir()

    print("Creating reference reconstruction...")
    create_reference_reconstruction(
        paths.recon_database_path,
        reference_model_path,
        poses,
        intrinsics_params=intrinsics_params,
    )

    reference = pycolmap.Reconstruction()
    reference.read(str(reference_model_path))

    print("Triangulating 3D points from known poses...")
    triangulation_options = pycolmap.IncrementalPipelineOptions()
    triangulation_options.random_seed = cfg.random_seed
    result = pycolmap.triangulate_points(
        reconstruction=reference,
        database_path=paths.recon_database_path,
        image_path=paths.recon_images_path,
        output_path=paths.model_dir,
        clear_points=True,
        options=triangulation_options,
        refine_intrinsics=cfg.reconstruction.refine_intrinsics,
    )
    summary = result.summary()
    print(summary)
    with open(paths.colmap_dir / "reconstruction_summary.txt", "w") as f:
        f.write(summary)

    shutil.rmtree(reference_model_path)
    return paths.model_dir


def run_automatic_reconstruction(paths: ScanPaths, cfg: DictConfig) -> Path:
    if paths.model_dir.exists():
        shutil.rmtree(paths.model_dir)
    paths.model_dir.mkdir(parents=True)

    # Load prior intrinsics if enabled
    intrinsics_params = None
    if cfg.reconstruction.load_prior_intrinsics:
        intrinsics_params = read_intrinsics_json(paths.intrinsics_json_path)
        apply_prior_intrinsics_to_database(paths.recon_database_path, intrinsics_params)

    mapping_options = pycolmap.IncrementalPipelineOptions()

    # ModifyForIndividualData
    mapping_options.min_focal_length_ratio = (
        cfg.reconstruction.mapping_params.min_focal_length_ratio
    )
    mapping_options.max_focal_length_ratio = (
        cfg.reconstruction.mapping_params.max_focal_length_ratio
    )
    mapping_options.max_extra_param = cfg.reconstruction.mapping_params.max_extra_param
    mapping_options.random_seed = cfg.random_seed
    # ModifyForHighQuality
    mapping_options.ba_local_max_num_iterations = (
        cfg.reconstruction.mapping_params.ba_local_max_num_iterations
    )
    mapping_options.ba_local_max_refinements = (
        cfg.reconstruction.mapping_params.ba_local_max_refinements
    )
    mapping_options.ba_global_max_num_iterations = (
        cfg.reconstruction.mapping_params.ba_global_max_num_iterations
    )
    mapping_options.ba_use_gpu = cfg.reconstruction.mapping_params.ba_use_gpu
    mapping_options.num_threads = cfg.reconstruction.mapping_params.num_threads
    mapping_options.ba_refine_focal_length = cfg.reconstruction.refine_intrinsics
    mapping_options.ba_refine_principal_point = cfg.reconstruction.refine_intrinsics
    mapping_options.ba_refine_extra_params = cfg.reconstruction.refine_intrinsics

    if cfg.reconstruction.load_prior_poses:
        poses = read_poses_dict(
            paths.poses_json_path, is_camera_to_world=True, as_quaternion=True
        )
        reference_model_path = paths.recon_output_dir / "_reference_model"
        if reference_model_path.exists():
            shutil.rmtree(reference_model_path)
        reference_model_path.mkdir()

        print("Creating reference reconstruction...")
        create_reference_reconstruction(
            paths.recon_database_path,
            reference_model_path,
            poses,
            intrinsics_params=intrinsics_params,
        )

        print("Running automatic reconstruction with prior poses...")
        recs = pycolmap.incremental_mapping(
            paths.recon_database_path,
            paths.recon_images_path,
            paths.model_dir,
            options=mapping_options,
            input_path=reference_model_path,
        )
        shutil.rmtree(reference_model_path)
    else:
        dummy_model_path = paths.recon_output_dir / "_tmp_dummy"
        create_dummy_model(dummy_model_path)

        print("Running automatic reconstruction (poses unknown)...")
        recs = pycolmap.incremental_mapping(
            paths.recon_database_path,
            paths.recon_images_path,
            paths.model_dir,
            options=mapping_options,
            input_path=dummy_model_path,
        )
        shutil.rmtree(dummy_model_path)
    summary_lines = []
    for idx, rec in recs.items():
        summary = f"#{idx} {rec.summary()}"
        logging.info(summary)
        summary_lines.append(summary)
    with open(paths.colmap_dir / "reconstruction_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    return paths.model_dir


def run_colmap_pipeline(cfg: DictConfig) -> None:
    """Run the COLMAP pipeline. Can be called standalone or from run_pipeline.py."""
    pycolmap.set_random_seed(cfg.random_seed)
    paths = create_scan_paths(cfg)
    _init_timings(paths.colmap_dir)

    if cfg.run.colmap_feat_extract_match:
        run_feature_extraction_and_matching(paths, cfg)

    if cfg.run.colmap_reconstruct:
        reconstruction_start_time = time.time()
        if cfg.reconstruction.mode == "triangulate":
            output_model_dir = run_triangulation(paths, cfg)
        else:
            output_model_dir = run_automatic_reconstruction(paths, cfg)
        reconstruction_end_time = time.time()
        reconstruction_duration = reconstruction_end_time - reconstruction_start_time
        _write_timing_entry(paths.colmap_dir, "reconstruction", reconstruction_duration)

        if (
            cfg.reconstruction.mode == "automatic"
            and cfg.reconstruction.load_prior_poses
        ):
            print(
                f"Total time for automatic mode with prior poses: "
                f"{reconstruction_duration:.2f} seconds"
            )
        else:
            print(
                f"Total time for {cfg.reconstruction.mode} mode: "
                f"{reconstruction_duration:.2f} seconds"
            )

        output_model = pycolmap.Reconstruction(output_model_dir)
        print("Exporting final model to TXT format...")
        output_model.write_text(output_model_dir)
        print("Exporting model sparse pointcloud as PLY...")
        output_model.export_PLY(output_model_dir / "sparse_pointcloud.ply")


@hydra.main(config_path="../configs/colmap", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_colmap_pipeline(cfg)


if __name__ == "__main__":
    main()
