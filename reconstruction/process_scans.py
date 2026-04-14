import argparse
import os
import shutil
import sqlite3
import time

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycolmap

from pycolmap import logging
from scipy.spatial.transform import Rotation as R
from visualizers.feature_viewer import FeatureViewer


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


def main():
    parser = argparse.ArgumentParser(
        description="Process microscope scans with COLMAP."
    )
    parser.add_argument(
        "--mode",
        choices=["triangulate", "automatic"],
        default="triangulate",
        help=(
            "'triangulate': use known robot poses to lock cameras and triangulate points. "
            "'automatic': run standard incremental SfM (poses estimated from images)."
        ),
    )
    parser.add_argument(
        "--load_prior_poses",
        action="store_true",
        default=False,
        help=(
            "For automatic mode: load known robot poses and create a reference reconstruction "
            "to use as a prior for incremental SfM. Output saved to 'with_prior_poses_model'. "
            "If not set, output is saved to 'no_prior_poses_model'."
        ),
    )
    parser.add_argument(
        "--rebuild_database",
        action="store_true",
        default=False,
        help=(
            "If set, delete existing database.db and rerun feature extraction/matching. "
            "If not set, extraction/matching only runs when database.db is missing."
        ),
    )
    args = parser.parse_args()

    pycolmap.set_random_seed(0)

    dataset_path = Path(
        "/home/codaero/Projects/Microscopic-3D-Reconstruction/microscope-data/scans/20260401_205638"
    )
    image_dir = dataset_path / "images"
    colmap_dir = dataset_path / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = dataset_path / "masks"

    # Shared database used by both triangulation and incremental mapping.
    # Feature extraction/matching are only rerun when rebuilding or missing DB.
    database_path = colmap_dir / "database.db"
    if args.rebuild_database and database_path.exists():
        print(f"Deleting existing database: {database_path}")
        database_path.unlink()

    workspace_dir = colmap_dir / args.mode

    should_extract_and_match = args.rebuild_database or not database_path.exists()
    if should_extract_and_match:
        print(f"Extracting features from {image_dir}...")
        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.estimate_affine_shape = True
        sift_options.max_num_features = 8192

        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.sift = sift_options

        reader_options = pycolmap.ImageReaderOptions()
        reader_options.camera_model = "OPENCV"
        reader_options.mask_path = masks_dir

        extraction_start_time = time.time()
        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
            extraction_options=extraction_options,
            device="cuda",
        )
        extraction_end_time = time.time()
        print(
            f"Feature extraction took {extraction_end_time - extraction_start_time:.2f} seconds"
        )

        feature_matching_options = pycolmap.FeatureMatchingOptions()
        feature_matching_options.use_gpu = True
        feature_matching_options.guided_matching = True

        matching_start_time = time.time()
        pycolmap.match_exhaustive(
            database_path=database_path,
            matching_options=feature_matching_options,
            device="cuda",
        )
        matching_end_time = time.time()
        print(
            f"Feature matching took {matching_end_time - matching_start_time:.2f} seconds"
        )
    else:
        print(f"Using existing database without re-extraction: {database_path}")

    output_model_dir = None

    reconstruction_start_time = time.time()
    if args.mode == "triangulate":
        # triangulate only has 1 mode so we can safely delete the entire workspace
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True)

        # Use known robot poses: lock cameras, triangulate points.
        poses = read_poses_from_scans(
            dataset_path, is_camera_to_world=True, as_quaternion=True
        )

        reference_model_path = workspace_dir / "reference_model"
        if reference_model_path.exists():
            shutil.rmtree(reference_model_path)
        reference_model_path.mkdir()

        print("Creating reference reconstruction...")
        create_reference_reconstruction(database_path, reference_model_path, poses)

        reference = pycolmap.Reconstruction()
        reference.read(str(reference_model_path))

        triangulated_model_path = workspace_dir / "triangulated_model"
        if triangulated_model_path.exists():
            shutil.rmtree(triangulated_model_path)
        triangulated_model_path.mkdir()

        print("Triangulating 3D points from known poses...")
        result = pycolmap.triangulate_points(
            reconstruction=reference,
            database_path=database_path,
            image_path=image_dir,
            output_path=triangulated_model_path,
            clear_points=True,
            refine_intrinsics=True,
        )
        print(result.summary())
        output_model_dir = triangulated_model_path

    else:  # automatic
        workspace_dir.mkdir(exist_ok=True)

        # Standard automatic reconstruction SfM — poses estimated from images.
        if args.load_prior_poses:
            sfm_path = workspace_dir / "with_prior_poses_model"
        else:
            sfm_path = workspace_dir / "no_prior_poses_model"
        if sfm_path.exists():
            shutil.rmtree(sfm_path)
        sfm_path.mkdir()

        mapping_options = pycolmap.IncrementalPipelineOptions()

        # ModifyForIndividualData
        mapping_options.min_focal_length_ratio = 0.1
        mapping_options.max_focal_length_ratio = 10.0
        mapping_options.max_extra_param = float("inf")
        # ModifyForHighQuality
        mapping_options.ba_local_max_num_iterations = 30
        mapping_options.ba_local_max_refinements = 3
        mapping_options.ba_global_max_num_iterations = 75
        mapping_options.ba_use_gpu = True

        if args.load_prior_poses:
            poses = read_poses_from_scans(
                dataset_path, is_camera_to_world=True, as_quaternion=True
            )
            reference_model_path = workspace_dir / "reference_model"
            if reference_model_path.exists():
                shutil.rmtree(reference_model_path)
            reference_model_path.mkdir()

            print("Creating reference reconstruction...")
            create_reference_reconstruction(database_path, reference_model_path, poses)

            print("Running automatic reconstruction with prior poses...")
            recs = pycolmap.incremental_mapping(
                database_path,
                image_dir,
                sfm_path,
                options=mapping_options,
                input_path=reference_model_path,
            )
        else:
            dummy_model_path = workspace_dir / "dummy_model"
            create_dummy_model(dummy_model_path)

            print("Running automatic reconstruction (poses unknown)...")
            recs = pycolmap.incremental_mapping(
                database_path,
                image_dir,
                sfm_path,
                options=mapping_options,
                input_path=dummy_model_path,
            )
        for idx, rec in recs.items():
            logging.info(f"#{idx} {rec.summary()}")

        output_model_dir = sfm_path
    reconstruction_end_time = time.time()

    if args.mode == "automatic" and args.load_prior_poses:
        print(
            f"Total time for automatic mode with prior poses: {reconstruction_end_time - reconstruction_start_time:.2f} seconds"
        )
    else:
        print(
            f"Total time for {args.mode} mode: {reconstruction_end_time - reconstruction_start_time:.2f} seconds"
        )

    # Convert the model to TXT and export
    output_model = pycolmap.Reconstruction(output_model_dir)
    print("Exporting final model to TXT format...")
    output_model.write_text(output_model_dir)
    print("Exporting model sparse pointcloud as PLY...")
    output_model.export_PLY(output_model_dir / "sparse_pointcloud.ply")


if __name__ == "__main__":
    main()
