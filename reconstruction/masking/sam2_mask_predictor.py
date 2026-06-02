import os
import shutil
import tempfile

import numpy as np
import torch

from PIL import Image

# HuggingFace model ID → (config relative path, checkpoint filename)
# Mirrors HF_MODEL_ID_TO_FILENAMES in sam2/build_sam.py without needing Hydra.
_HF_MODEL_REGISTRY = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


def _build_sam2_from_hf(model_id: str, device: str, for_video: bool = False):
    """Build a SAM2 model from HuggingFace without using Hydra's compose().

    The standard build_sam2_hf() calls hydra.compose() which conflicts with an
    already-running Hydra context (e.g. run_pipeline.py). This function avoids
    that by loading the YAML config directly via OmegaConf and instantiating
    the model with hydra.utils.instantiate.
    """
    import sam2

    from huggingface_hub import hf_hub_download
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if model_id not in _HF_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown SAM2 model id {model_id!r}. " f"Known: {list(_HF_MODEL_REGISTRY)}"
        )

    config_rel, checkpoint_name = _HF_MODEL_REGISTRY[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)

    sam2_pkg_dir = os.path.dirname(os.path.abspath(sam2.__file__))
    config_path = os.path.join(sam2_pkg_dir, config_rel)
    cfg = OmegaConf.load(config_path)

    # Replicate the postprocessing overrides from build_sam2 / build_sam2_video_predictor.
    decoder_args = cfg.model.get("sam_mask_decoder_extra_args", {})
    cfg.model.sam_mask_decoder_extra_args = OmegaConf.merge(
        decoder_args,
        {
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
    )
    if for_video:
        cfg.model._target_ = "sam2.sam2_video_predictor.SAM2VideoPredictor"
        cfg.model.binarize_mask_from_pts_for_mem_enc = True
        cfg.model.fill_hole_area = 8

    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    return model


class Sam2MaskPredictor:
    """SAM2 image and video prediction for the masking pipeline."""

    def __init__(self, cfg):
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.cfg = cfg
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_id = cfg.sam2_model_id

        print(f"Loading SAM2 image predictor ({model_id}) on {self.device}...")
        self.image_predictor = SAM2ImagePredictor(
            _build_sam2_from_hf(model_id, self.device, for_video=False)
        )

        print(f"Loading SAM2 video predictor ({model_id}) on {self.device}...")
        self.video_predictor = _build_sam2_from_hf(
            model_id, self.device, for_video=True
        )

    def predict_image_from_points(self, image_np: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        if len(points_xy) == 0:
            return np.zeros(image_np.shape[:2], dtype=bool)

        point_coords = np.array(points_xy, dtype=np.float32)   # (N, 2)
        point_labels = np.ones(len(points_xy), dtype=np.int32)  # all foreground

        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if str(self.device).startswith("cuda") else "cpu",
            dtype=torch.bfloat16,
            enabled=str(self.device).startswith("cuda"),
        ):
            self.image_predictor.set_image(image_np)
            masks, scores, _ = self.image_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool)

    def predict_image_from_mask(self, image_np: np.ndarray, prompt_mask: np.ndarray) -> np.ndarray:
        if not prompt_mask.any():
            return np.zeros(prompt_mask.shape, dtype=bool)

        # SAM2 expects a low-res (1, 256, 256) float32 logit mask as input.
        mask_tensor = torch.from_numpy(prompt_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mask_input = torch.nn.functional.interpolate(
            mask_tensor, size=(256, 256), mode="bilinear", align_corners=False
        ).squeeze(0).numpy()  # (1, 256, 256)

        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if str(self.device).startswith("cuda") else "cpu",
            dtype=torch.bfloat16,
            enabled=str(self.device).startswith("cuda"),
        ):
            self.image_predictor.set_image(image_np)
            masks, scores, _ = self.image_predictor.predict(
                mask_input=mask_input,
                multimask_output=False,
            )
        return masks[0].astype(bool)

    def predict_video_from_mask(
        self,
        image_paths: list,
        bootstrap_frame_idx: int,
        bootstrap_mask: np.ndarray,
        offload_to_cpu: bool = True,
    ) -> dict:
        """Run SAM2 video predictor over all frames.

        Writes frames to a temporary JPEG directory, propagates the bootstrap
        mask through the entire sequence, and returns per-frame masks.

        Args:
            image_paths: sorted list of all input image paths.
            bootstrap_frame_idx: index in image_paths of the bootstrap image.
            bootstrap_mask: precise boolean mask (H, W) for the bootstrap frame.
            offload_to_cpu: if True, offload video and state tensors to CPU to
                conserve GPU memory.

        Returns:
            dict mapping int frame_idx (0-based in image_paths order) to a
            boolean numpy mask array (H, W).
        """
        jpeg_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        try:
            print(f"Writing {len(image_paths)} JPEG frames to temp dir...")
            for i, path in enumerate(image_paths):
                img = Image.open(path).convert("RGB")
                img.save(
                    os.path.join(jpeg_dir, f"{i:05d}.jpg"), format="JPEG", quality=95
                )

            print("Initializing SAM2 video inference state...")
            inference_state = self.video_predictor.init_state(
                video_path=jpeg_dir,
                offload_video_to_cpu=offload_to_cpu,
                offload_state_to_cpu=offload_to_cpu,
            )

            mask_tensor = torch.from_numpy(bootstrap_mask.astype(bool))
            self.video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=bootstrap_frame_idx,
                obj_id=1,
                mask=mask_tensor,
            )

            frame_masks = {}

            print("Propagating forward from bootstrap frame...")
            for (
                frame_idx,
                _obj_ids,
                video_res_masks,
            ) in self.video_predictor.propagate_in_video(
                inference_state, start_frame_idx=bootstrap_frame_idx
            ):
                frame_masks[frame_idx] = (
                    (video_res_masks[0] > 0).squeeze().cpu().numpy()
                )

            if bootstrap_frame_idx > 0:
                print("Propagating backward to frame 0...")
                for (
                    frame_idx,
                    _obj_ids,
                    video_res_masks,
                ) in self.video_predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=bootstrap_frame_idx,
                    reverse=True,
                ):
                    if frame_idx not in frame_masks:
                        frame_masks[frame_idx] = (
                            (video_res_masks[0] > 0).squeeze().cpu().numpy()
                        )

            return frame_masks
        finally:
            shutil.rmtree(jpeg_dir, ignore_errors=True)

    def predict_video_from_points(
        self,
        image_paths: list,
        bootstrap_frame_idx: int,
        points_xy: np.ndarray,
        offload_to_cpu: bool = True,
    ) -> dict:
        """Run SAM2 video predictor over all frames using point prompts on the bootstrap frame.

        Args:
            image_paths: sorted list of all input image paths.
            bootstrap_frame_idx: index in image_paths of the bootstrap image.
            points_xy: (N, 2) float32 array of (x, y) foreground point prompts.
            offload_to_cpu: if True, offload video and state tensors to CPU to
                conserve GPU memory.

        Returns:
            dict mapping int frame_idx (0-based in image_paths order) to a
            boolean numpy mask array (H, W).
        """
        jpeg_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        try:
            print(f"Writing {len(image_paths)} JPEG frames to temp dir...")
            for i, path in enumerate(image_paths):
                img = Image.open(path).convert("RGB")
                img.save(
                    os.path.join(jpeg_dir, f"{i:05d}.jpg"), format="JPEG", quality=95
                )

            print("Initializing SAM2 video inference state...")
            inference_state = self.video_predictor.init_state(
                video_path=jpeg_dir,
                offload_video_to_cpu=offload_to_cpu,
                offload_state_to_cpu=offload_to_cpu,
            )

            point_coords = np.array(points_xy, dtype=np.float32)   # (N, 2)
            point_labels = np.ones(len(points_xy), dtype=np.int32)  # all foreground
            self.video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=bootstrap_frame_idx,
                obj_id=1,
                points=point_coords,
                labels=point_labels,
            )

            frame_masks = {}

            print("Propagating forward from bootstrap frame...")
            for (
                frame_idx,
                _obj_ids,
                video_res_masks,
            ) in self.video_predictor.propagate_in_video(
                inference_state, start_frame_idx=bootstrap_frame_idx
            ):
                frame_masks[frame_idx] = (
                    (video_res_masks[0] > 0).squeeze().cpu().numpy()
                )

            if bootstrap_frame_idx > 0:
                print("Propagating backward to frame 0...")
                for (
                    frame_idx,
                    _obj_ids,
                    video_res_masks,
                ) in self.video_predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=bootstrap_frame_idx,
                    reverse=True,
                ):
                    if frame_idx not in frame_masks:
                        frame_masks[frame_idx] = (
                            (video_res_masks[0] > 0).squeeze().cpu().numpy()
                        )

            return frame_masks
        finally:
            shutil.rmtree(jpeg_dir, ignore_errors=True)
