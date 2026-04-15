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
    dataset_path: Path
    image_dir: Path
    colmap_dir: Path
    masks_dir: Path
    database_path: Path
    workspace_dir: Path


def create_scan_paths(cfg: DictConfig) -> ScanPaths:
    dataset_path = Path(cfg.paths.dataset)
    colmap_dir = dataset_path / cfg.paths.colmap_subdir
    colmap_dir.mkdir(parents=True, exist_ok=True)
    return ScanPaths(
        dataset_path=dataset_path,
        image_dir=dataset_path / cfg.paths.images_subdir,
        colmap_dir=colmap_dir,
        masks_dir=dataset_path / cfg.paths.masks_subdir,
        database_path=colmap_dir / cfg.paths.database_filename,
        workspace_dir=colmap_dir / cfg.mode,
    )


def read_poses_from_scans(dataset_path, is_camera_to_world=False, as_quaternion=False):
    """
    Reads robot poses from each scan subdirectory.

    Args:
        dataset_path (Path): Root dataset directory containing scan* subdirectories.
        is_camera_to_world (bool): If True, the stored poses are Camera-to-World and
                                   will be inverted to World-to-Camera before returning.
        as_quaternion (bool): If True, each pose is returned as a dict
                              {'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz'}.
                              If False, each pose is returned as a 4x4 numpy array.

    Returns:
        dict: Mapping image names (e.g. 'scan001.jpg') to either 4x4 numpy arrays
              or quaternion+translation dicts, depending on as_quaternion.
    """
    poses = {}
    print(
        f"Reading poses as {'quaternion' if as_quaternion else 'matrix'} ({'Camera-to-World' if is_camera_to_world else 'World-to-Camera'}) from scans in {dataset_path}..."
    )
    for scan_dir in sorted(dataset_path.glob("scan*")):
        pose_file = scan_dir / "pose.npy"
        if pose_file.exists():
            image_name = f"{scan_dir.name}.jpg"
            matrix = np.load(pose_file)
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
            fmt = "quaternion" if as_quaternion else "matrix"
    return poses


def create_reference_reconstruction(db_path, output_dir, pose_dict):
    """
    Creates a COLMAP text model from the database and known poses.

    Args:
        db_path: Path to the COLMAP database.db
        output_dir: Directory to write cameras.txt, images.txt, points3D.txt
        pose_dict (dict): Mapping image names to quaternion+translation dicts
                          {'qw','qx','qy','qz','tx','ty','tz'}, as returned by
                          read_poses_from_scans(as_quaternion=True).
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
    # Feature extraction/matching are rerun when rebuilding or if DB is missing.
    if cfg.rebuild_database and paths.database_path.exists():
        print(f"Deleting existing database: {paths.database_path}")
        paths.database_path.unlink()

    should_extract_and_match = cfg.rebuild_database or not paths.database_path.exists()
    if not should_extract_and_match:
        print(f"Using existing database without re-extraction: {paths.database_path}")
        return

    print(f"Extracting features from {paths.image_dir}...")
    sift_options = pycolmap.SiftExtractionOptions()
    sift_options.estimate_affine_shape = cfg.extraction.estimate_affine_shape
    sift_options.max_num_features = cfg.extraction.max_num_features

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift = sift_options
    extraction_options.num_threads = cfg.extraction.num_threads

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = cfg.extraction.camera_model
    reader_options.mask_path = paths.masks_dir

    extraction_start_time = time.time()
    pycolmap.extract_features(
        database_path=paths.database_path,
        image_path=paths.image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=cfg.extraction.device,
    )
    extraction_end_time = time.time()
    print(
        f"Feature extraction took {extraction_end_time - extraction_start_time:.2f} seconds"
    )

    feature_matching_options = pycolmap.FeatureMatchingOptions()
    feature_matching_options.use_gpu = cfg.matching.use_gpu
    feature_matching_options.num_threads = cfg.matching.num_threads
    feature_matching_options.guided_matching = cfg.matching.guided_matching

    verification_options = pycolmap.TwoViewGeometryOptions()
    verification_options.ransac.random_seed = cfg.random_seed

    matching_start_time = time.time()
    pycolmap.match_exhaustive(
        database_path=paths.database_path,
        matching_options=feature_matching_options,
        verification_options=verification_options,
        device="cuda" if cfg.matching.use_gpu else "cpu",
    )
    matching_end_time = time.time()
    print(
        f"Feature matching took {matching_end_time - matching_start_time:.2f} seconds"
    )


def run_triangulation(paths: ScanPaths, cfg: DictConfig) -> Path:
    # triangulate only has 1 mode so we can safely delete the entire workspace
    if paths.workspace_dir.exists():
        shutil.rmtree(paths.workspace_dir)
    paths.workspace_dir.mkdir(parents=True)

    # Use known robot poses: lock cameras, triangulate points.
    poses = read_poses_from_scans(
        paths.dataset_path, is_camera_to_world=True, as_quaternion=True
    )

    reference_model_path = paths.workspace_dir / "reference_model"
    if reference_model_path.exists():
        shutil.rmtree(reference_model_path)
    reference_model_path.mkdir()

    print("Creating reference reconstruction...")
    create_reference_reconstruction(paths.database_path, reference_model_path, poses)

    reference = pycolmap.Reconstruction()
    reference.read(str(reference_model_path))

    triangulated_model_path = paths.workspace_dir / "triangulated_model"
    if triangulated_model_path.exists():
        shutil.rmtree(triangulated_model_path)
    triangulated_model_path.mkdir()

    print("Triangulating 3D points from known poses...")
    triangulation_options = pycolmap.IncrementalPipelineOptions()
    triangulation_options.random_seed = cfg.random_seed
    result = pycolmap.triangulate_points(
        reconstruction=reference,
        database_path=paths.database_path,
        image_path=paths.image_dir,
        output_path=triangulated_model_path,
        clear_points=True,
        options=triangulation_options,
        refine_intrinsics=True,
    )
    print(result.summary())
    return triangulated_model_path


def run_automatic_reconstruction(paths: ScanPaths, cfg: DictConfig) -> Path:
    paths.workspace_dir.mkdir(exist_ok=True)

    # Standard automatic reconstruction SfM — poses estimated from images.
    if cfg.load_prior_poses:
        sfm_path = paths.workspace_dir / "with_prior_poses_model"
    else:
        sfm_path = paths.workspace_dir / "no_prior_poses_model"
    if sfm_path.exists():
        shutil.rmtree(sfm_path)
    sfm_path.mkdir()

    mapping_options = pycolmap.IncrementalPipelineOptions()

    # ModifyForIndividualData
    mapping_options.min_focal_length_ratio = cfg.mapping.min_focal_length_ratio
    mapping_options.max_focal_length_ratio = cfg.mapping.max_focal_length_ratio
    mapping_options.max_extra_param = cfg.mapping.max_extra_param
    mapping_options.random_seed = cfg.random_seed
    # ModifyForHighQuality
    mapping_options.ba_local_max_num_iterations = (
        cfg.mapping.ba_local_max_num_iterations
    )
    mapping_options.ba_local_max_refinements = cfg.mapping.ba_local_max_refinements
    mapping_options.ba_global_max_num_iterations = (
        cfg.mapping.ba_global_max_num_iterations
    )
    mapping_options.ba_use_gpu = cfg.mapping.ba_use_gpu
    mapping_options.num_threads = cfg.mapping.num_threads

    if cfg.load_prior_poses:
        poses = read_poses_from_scans(
            paths.dataset_path, is_camera_to_world=True, as_quaternion=True
        )
        reference_model_path = paths.workspace_dir / "reference_model"
        if reference_model_path.exists():
            shutil.rmtree(reference_model_path)
        reference_model_path.mkdir()

        print("Creating reference reconstruction...")
        create_reference_reconstruction(
            paths.database_path, reference_model_path, poses
        )

        print("Running automatic reconstruction with prior poses...")
        recs = pycolmap.incremental_mapping(
            paths.database_path,
            paths.image_dir,
            sfm_path,
            options=mapping_options,
            input_path=reference_model_path,
        )
    else:
        dummy_model_path = paths.workspace_dir / "dummy_model"
        create_dummy_model(dummy_model_path)

        print("Running automatic reconstruction (poses unknown)...")
        recs = pycolmap.incremental_mapping(
            paths.database_path,
            paths.image_dir,
            sfm_path,
            options=mapping_options,
            input_path=dummy_model_path,
        )
    for idx, rec in recs.items():
        logging.info(f"#{idx} {rec.summary()}")

    return sfm_path


@hydra.main(config_path="configs/colmap", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pycolmap.set_random_seed(cfg.random_seed)

    paths = create_scan_paths(cfg)

    run_feature_extraction_and_matching(paths, cfg)

    reconstruction_start_time = time.time()
    if cfg.mode == "triangulate":
        output_model_dir = run_triangulation(paths, cfg)
    else:
        output_model_dir = run_automatic_reconstruction(paths, cfg)
    reconstruction_end_time = time.time()

    if cfg.mode == "automatic" and cfg.load_prior_poses:
        print(
            f"Total time for automatic mode with prior poses: {reconstruction_end_time - reconstruction_start_time:.2f} seconds"
        )
    else:
        print(
            f"Total time for {cfg.mode} mode: {reconstruction_end_time - reconstruction_start_time:.2f} seconds"
        )

    # Convert the model to TXT and export
    output_model = pycolmap.Reconstruction(output_model_dir)
    print("Exporting final model to TXT format...")
    output_model.write_text(output_model_dir)
    print("Exporting model sparse pointcloud as PLY...")
    output_model.export_PLY(output_model_dir / "sparse_pointcloud.ply")


if __name__ == "__main__":
    main()
