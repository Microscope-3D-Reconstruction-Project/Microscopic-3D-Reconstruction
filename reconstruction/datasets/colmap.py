import json
import os

from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import pycolmap
import torch

from PIL import Image
from tqdm import tqdm
from typing_extensions import assert_never

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder by *factor* and write PNGs to *resized_dir*."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)
    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        os.makedirs(os.path.dirname(resized_path), exist_ok=True)
        image = Image.open(image_path)
        resized_size = (
            int(round(image.width / factor)),
            int(round(image.height / factor)),
        )
        if image.mode == "RGBA":
            rgb = image.convert("RGB").resize(resized_size, Image.BICUBIC)
            alpha = image.getchannel("A").resize(resized_size, Image.NEAREST)
            rgba = np.dstack((np.array(rgb), np.array(alpha)))
            rgba[rgba[..., 3] == 0, :3] = 0
            imageio.imwrite(resized_path, rgba.astype(np.uint8))
        else:
            resized_image = np.array(
                image.convert("RGB").resize(resized_size, Image.BICUBIC)
            )
            imageio.imwrite(resized_path, resized_image)
    return resized_dir


def _load_rgb_image(path: str) -> np.ndarray:
    """Load RGB image data, applying alpha as black transparency when present."""
    image = imageio.imread(path)
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] >= 4:
        rgb = image[..., :3].astype(np.float32)
        alpha = image[..., 3:4].astype(np.float32) / 255.0
        return np.clip(rgb * alpha, 0, 255).astype(image.dtype)
    return image[..., :3]


def _resize_mask_folder(mask_dir: str, resized_dir: str, factor: int) -> str:
    """Resize binary mask folder by *factor* with nearest-neighbor sampling."""
    print(f"Downscaling masks by {factor}x from {mask_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)
    mask_files = _get_rel_paths(mask_dir)
    for mask_file in tqdm(mask_files):
        mask_path = os.path.join(mask_dir, mask_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(mask_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        os.makedirs(os.path.dirname(resized_path), exist_ok=True)
        mask = _load_binary_mask(mask_path)
        resized_size = (
            int(round(mask.shape[1] / factor)),
            int(round(mask.shape[0] / factor)),
        )
        resized_mask = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            resized_size, Image.NEAREST
        )
        imageio.imwrite(resized_path, np.array(resized_mask, dtype=np.uint8))
    return resized_dir


def _resolve_downscaled_folder(path_dir: str, factor: int, resize_fn) -> str:
    """Return ``path_dir`` or a ``_<factor>`` copy, creating it if missing."""
    if factor <= 1:
        return path_dir
    resized_dir = f"{path_dir}_{factor}"
    if not os.path.exists(resized_dir):
        resize_fn(path_dir, resized_dir, factor)
    return resized_dir


def _resolve_mask_path(masks_dir: str, image_name: str) -> Optional[str]:
    """Resolve a per-image mask path, preferring a PNG with the same stem."""
    candidates = [
        os.path.join(masks_dir, image_name),
        os.path.join(masks_dir, os.path.splitext(image_name)[0] + ".png"),
        os.path.join(masks_dir, os.path.splitext(image_name)[0] + ".jpg"),
        os.path.join(masks_dir, os.path.splitext(image_name)[0] + ".jpeg"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _load_binary_mask(path: str) -> np.ndarray:
    """Load a per-image mask as a boolean numpy array."""
    mask = imageio.imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def _remap_binary_mask(
    mask: np.ndarray, mapx: np.ndarray, mapy: np.ndarray
) -> np.ndarray:
    """Apply camera undistortion maps to a boolean mask."""
    return (
        cv2.remap(
            mask.astype(np.uint8),
            mapx,
            mapy,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        > 0
    )


def _resolve_mask_paths(
    masks_dir: Optional[str],
    image_names: List[str],
    label: str,
) -> List[Optional[str]]:
    """Resolve optional per-image mask paths and report missing masks."""
    if masks_dir is None:
        return [None] * len(image_names)

    mask_paths = [
        _resolve_mask_path(masks_dir, image_name) for image_name in image_names
    ]
    num_missing_masks = sum(mask_path is None for mask_path in mask_paths)
    if num_missing_masks == len(mask_paths):
        print(f"Warning: no per-image {label} masks were found in {masks_dir!r}.")
    elif num_missing_masks > 0:
        print(
            f"Warning: missing {label} masks for "
            f"{num_missing_masks} / {len(mask_paths)} images."
        )
    return mask_paths


def _has_valid_undistortion(
    K_undist: np.ndarray, roi_undist: tuple, width: int, height: int
) -> bool:
    """Return whether OpenCV undistortion produced a usable camera/ROI."""
    if not np.isfinite(K_undist).all():
        return False
    if K_undist[0, 0] <= 0 or K_undist[1, 1] <= 0:
        return False
    x, y, w, h = roi_undist
    if w <= 0 or h <= 0:
        return False
    if x < 0 or y < 0:
        return False
    if x + w > width or y + h > height:
        return False
    return True


def _camera_model_name(cam) -> str:
    """Return the camera model as a plain string regardless of pycolmap version."""
    model = cam.model
    # pycolmap >= 4.0: model is a CameraModelId enum
    if hasattr(model, "name"):
        return model.name
    # some builds expose model_name directly on the camera
    if hasattr(cam, "model_name"):
        return cam.model_name
    return str(model).split(".")[-1]


def _parse_camera(cam):
    """
    Return (K_3x3, distortion_params, camtype, model_name) for a pycolmap Camera.

    K is the un-scaled calibration matrix (divide by factor after this call).
    distortion_params is a float32 array of [k1, k2, p1, p2] (zeros where unused),
    or empty for pinhole models.
    camtype is "perspective" or "fisheye".

    Parameter layout by model (from COLMAP docs):
      SIMPLE_PINHOLE   [f, cx, cy]
      PINHOLE          [fx, fy, cx, cy]
      SIMPLE_RADIAL    [f, cx, cy, k1]
      RADIAL           [f, cx, cy, k1, k2]
      OPENCV           [fx, fy, cx, cy, k1, k2, p1, p2]
      OPENCV_FISHEYE   [fx, fy, cx, cy, k1, k2, k3, k4]
    """
    model_name = _camera_model_name(cam)
    p = np.array(cam.params, dtype=np.float64)

    if model_name == "SIMPLE_PINHOLE":
        K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]], dtype=np.float64)
        distortion = np.empty(0, dtype=np.float32)
        camtype = "perspective"
    elif model_name == "PINHOLE":
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], dtype=np.float64)
        distortion = np.empty(0, dtype=np.float32)
        camtype = "perspective"
    elif model_name == "SIMPLE_RADIAL":
        K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]], dtype=np.float64)
        distortion = np.array([p[3], 0.0, 0.0, 0.0], dtype=np.float32)
        camtype = "perspective"
    elif model_name == "RADIAL":
        K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]], dtype=np.float64)
        distortion = np.array([p[3], p[4], 0.0, 0.0], dtype=np.float32)
        camtype = "perspective"
    elif model_name == "OPENCV":
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], dtype=np.float64)
        distortion = np.array([p[4], p[5], p[6], p[7]], dtype=np.float32)
        camtype = "perspective"
    elif model_name == "OPENCV_FISHEYE":
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], dtype=np.float64)
        distortion = np.array([p[4], p[5], p[6], p[7]], dtype=np.float32)
        camtype = "fisheye"
    else:
        raise ValueError(
            f"Unsupported camera model: {model_name}. "
            "Supported: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, OPENCV_FISHEYE"
        )

    return K, distortion, camtype, model_name


class Parser:
    """COLMAP parser compatible with pycolmap >= 4.0 (Reconstruction API).

    Args:
        colmap_dir: Path to the COLMAP sparse model directory (e.g.
            ``sparse/0`` or ``sparse``).  The caller is responsible for
            providing the exact path — no sub-directory guessing is done.
        images_dir: Path to the directory containing the source images that
            COLMAP registered.
        masks_dir: Optional path to a directory containing per-image binary
            foreground/object masks.
        valid_region_masks_dir: Optional path to per-image focus-stack valid masks.
        factor: Integer downsampling factor applied to intrinsics and image
            sizes.  ``1`` means no downsampling.
        normalize: If ``True``, apply a similarity transform so that camera
            positions are centred and axis-aligned.
        test_every: Every *N*-th image (by sorted filename) is held out for
            validation; the rest are used for training.
    """

    def __init__(
        self,
        colmap_dir: str,
        images_dir: str,
        masks_dir: Optional[str] = None,
        valid_region_masks_dir: Optional[str] = None,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ) -> None:
        self.colmap_dir = colmap_dir
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.valid_region_masks_dir = valid_region_masks_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir!r} does not exist."
        assert os.path.exists(
            images_dir
        ), f"Images directory {images_dir!r} does not exist."
        if masks_dir is not None:
            assert os.path.exists(
                masks_dir
            ), f"Masks directory {masks_dir!r} does not exist."
        if valid_region_masks_dir is not None:
            assert os.path.exists(
                valid_region_masks_dir
            ), f"Valid region masks directory {valid_region_masks_dir!r} does not exist."

        reconstruction = pycolmap.Reconstruction(colmap_dir)
        imdata = reconstruction.images  # dict[image_id -> Image]

        # Only use images with a registered pose.
        valid_ids = [k for k, im in imdata.items() if im.has_pose]
        if len(valid_ids) == 0:
            raise ValueError("No images with a valid pose found in COLMAP.")

        w2c_mats: List[np.ndarray] = []
        camera_ids: List[int] = []
        Ks_dict: Dict[int, np.ndarray] = {}
        params_dict: Dict[int, np.ndarray] = {}
        imsize_dict: Dict[int, tuple] = {}
        camtype_dict: Dict[int, str] = {}
        bottom = np.array([0, 0, 0, 1], dtype=np.float64).reshape(1, 4)

        for k in valid_ids:
            im = imdata[k]
            pose = im.cam_from_world()
            rot = pose.rotation.matrix()  # (3, 3)
            trans = pose.translation.reshape(3, 1)
            w2c_mats.append(
                np.concatenate([np.concatenate([rot, trans], axis=1), bottom], axis=0)
            )
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            if camera_id not in Ks_dict:
                cam = reconstruction.cameras[camera_id]
                K, distortion, camtype, model_name = _parse_camera(cam)
                K = K.copy()
                K[:2, :] /= factor
                Ks_dict[camera_id] = K
                params_dict[camera_id] = distortion
                camtype_dict[camera_id] = camtype
                imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
                if model_name not in ("SIMPLE_PINHOLE", "PINHOLE"):
                    print(
                        f"Warning: camera {camera_id} uses {model_name}. "
                        "Images have distortion."
                    )

        print(
            f"[Parser] {len(w2c_mats)} images, taken by {len(set(camera_ids))} cameras."
        )

        w2c_mats_np = np.stack(w2c_mats, axis=0)
        camtoworlds = np.linalg.inv(w2c_mats_np)
        image_names = [imdata[k].name for k in valid_ids]

        # Sort by filename for reproducible train/test splits.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Extended metadata (Bilarf dataset).
        self.extconf: Dict[str, Any] = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(images_dir, "..", "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Forward-facing scene bounds.
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(images_dir, "..", "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Resolve actual image/mask directories, creating downsampled copies
        # with a _<factor> suffix when needed.
        colmap_image_dir = images_dir
        folder_factor = 1 if self.extconf["no_factor_suffix"] else factor
        image_dir = (
            images_dir
            if folder_factor <= 1
            else _resolve_downscaled_folder(
                images_dir, folder_factor, _resize_image_folder
            )
        )
        if not os.path.exists(image_dir):
            raise ValueError(f"Image folder {image_dir!r} does not exist.")
        masks_dir = (
            _resolve_downscaled_folder(masks_dir, folder_factor, _resize_mask_folder)
            if masks_dir is not None
            else None
        )
        valid_region_masks_dir = (
            _resolve_downscaled_folder(
                valid_region_masks_dir, folder_factor, _resize_mask_folder
            )
            if valid_region_masks_dir is not None
            else None
        )

        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
        mask_paths = _resolve_mask_paths(
            masks_dir=masks_dir,
            image_names=image_names,
            label="foreground",
        )
        valid_region_mask_paths = _resolve_mask_paths(
            masks_dir=valid_region_masks_dir,
            image_names=image_names,
            label="valid-region",
        )

        # 3-D points.
        point3D_ids = sorted(reconstruction.points3D.keys())
        point3D_id_to_idx = {pid: i for i, pid in enumerate(point3D_ids)}
        points = np.array(
            [reconstruction.points3D[pid].xyz for pid in point3D_ids], dtype=np.float32
        )
        points_err = np.array(
            [reconstruction.points3D[pid].error for pid in point3D_ids],
            dtype=np.float32,
        )
        points_rgb = np.array(
            [reconstruction.points3D[pid].color for pid in point3D_ids], dtype=np.uint8
        )

        # Map each 3-D point to the images that observe it.
        image_id_to_name = {
            img_id: img.name for img_id, img in reconstruction.images.items()
        }
        point_indices: Dict[str, List[int]] = {}
        for pid in point3D_ids:
            p3d = reconstruction.points3D[pid]
            for elem in p3d.track.elements:
                img_name = image_id_to_name.get(elem.image_id)
                if img_name is not None:
                    point_indices.setdefault(img_name, []).append(
                        point3D_id_to_idx[pid]
                    )
        point_indices_np: Dict[str, np.ndarray] = {
            k: np.array(v, dtype=np.int32) for k, v in point_indices.items()
        }

        # Optional world-space normalisation.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)
            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)
            transform = T2 @ T1
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names: List[str] = image_names
        self.image_paths: List[str] = image_paths
        self.camtoworlds: np.ndarray = camtoworlds
        self.camera_ids: List[int] = camera_ids
        self.Ks_dict: Dict[int, np.ndarray] = Ks_dict
        self.params_dict: Dict[int, np.ndarray] = params_dict
        self.imsize_dict: Dict[int, tuple] = imsize_dict
        self.mask_paths: List[Optional[str]] = mask_paths
        self.valid_region_mask_paths: List[Optional[str]] = valid_region_mask_paths
        self.points: np.ndarray = points
        self.points_err: np.ndarray = points_err
        self.points_rgb: np.ndarray = points_rgb
        self.point_indices: Dict[str, np.ndarray] = point_indices_np
        self.transform: np.ndarray = transform

        # Scale K / imsize to match the actual on-disk resolution (handles
        # datasets where COLMAP intrinsics were estimated at a different res).
        actual_image = _load_rgb_image(self.image_paths[0])
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_width = actual_width / colmap_width
        s_height = actual_height / colmap_height
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            w, h = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(w * s_width), int(h * s_height))

        # Build undistortion maps for cameras that have distortion parameters.
        self.mapx_dict: Dict[int, np.ndarray] = {}
        self.mapy_dict: Dict[int, np.ndarray] = {}
        self.roi_undist_dict: Dict[int, list] = {}
        for camera_id, dist_params in self.params_dict.items():
            if len(dist_params) == 0:
                continue
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            camtype = camtype_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, dist_params, (width, height), 0
                )
                if _has_valid_undistortion(K_undist, roi_undist, width, height):
                    mapx, mapy = cv2.initUndistortRectifyMap(
                        K, dist_params, None, K_undist, (width, height), cv2.CV_32FC1
                    )
                else:
                    print(
                        f"Warning: skipping undistortion for camera {camera_id} "
                        f"because OpenCV returned an invalid ROI {roi_undist} "
                        f"or intrinsics."
                    )
                    self.Ks_dict[camera_id] = K
                    self.params_dict[camera_id] = np.empty(0, dtype=np.float32)
                    self.imsize_dict[camera_id] = (width, height)
                    continue
            elif camtype == "fisheye":
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + dist_params[0] * theta**2
                    + dist_params[1] * theta**4
                    + dist_params[2] * theta**6
                    + dist_params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)
                mask = (
                    (mapx > 0) & (mapy > 0) & (mapx < width - 1) & (mapy < height - 1)
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        self.scene_scale: float = float(
            np.max(np.linalg.norm(camera_locations - scene_center, axis=1))
        )


class Dataset:
    """A simple dataset class for iterating over registered images."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ) -> None:
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        self.indices = (
            indices[indices % self.parser.test_every != 0]
            if split == "train"
            else indices[indices % self.parser.test_every == 0]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = _load_rgb_image(self.parser.image_paths[index])
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask_path = self.parser.mask_paths[index]
        valid_region_mask_path = self.parser.valid_region_mask_paths[index]
        mask = None
        if mask_path is not None:
            mask = _load_binary_mask(mask_path)
        valid_region_mask = None
        if valid_region_mask_path is not None:
            valid_region_mask = _load_binary_mask(valid_region_mask_path)

        if len(params) > 0:
            mapx = self.parser.mapx_dict[camera_id]
            mapy = self.parser.mapy_dict[camera_id]
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            if mask is not None:
                mask = _remap_binary_mask(mask, mapx, mapy)
            if valid_region_mask is not None:
                valid_region_mask = _remap_binary_mask(valid_region_mask, mapx, mapy)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]
            if mask is not None:
                mask = mask[y : y + h, x : x + w]
            if valid_region_mask is not None:
                valid_region_mask = valid_region_mask[y : y + h, x : x + w]

        if self.patch_size is not None:
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            if mask is not None:
                mask = mask[y : y + self.patch_size, x : x + self.patch_size]
            if valid_region_mask is not None:
                valid_region_mask = valid_region_mask[
                    y : y + self.patch_size, x : x + self.patch_size
                ]
            K[0, 2] -= x
            K[1, 2] -= y

        data: Dict[str, Any] = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()
        if valid_region_mask is not None:
            data["valid_region_mask"] = torch.from_numpy(valid_region_mask).bool()

        if self.load_depths:
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices.get(
                image_name, np.empty(0, dtype=np.int32)
            )
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]
            depths = points_cam[:, 2]
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            if valid_region_mask is not None:
                x_idx = np.clip(
                    np.round(points[:, 0]).astype(np.int32),
                    0,
                    image.shape[1] - 1,
                )
                y_idx = np.clip(
                    np.round(points[:, 1]).astype(np.int32),
                    0,
                    image.shape[0] - 1,
                )
                selector = selector & valid_region_mask[y_idx, x_idx]
            data["points"] = torch.from_numpy(points[selector]).float()
            data["depths"] = torch.from_numpy(depths[selector]).float()

        return data
