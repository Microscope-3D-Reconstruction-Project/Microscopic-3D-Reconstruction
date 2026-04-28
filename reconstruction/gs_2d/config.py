from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from gsplat.strategy import DefaultStrategy, MCMCStrategy
from typing_extensions import Literal, assert_never


@dataclass
class Config:
    # ── Miscellaneous ─────────────────────────────────────────────────────────
    random_seed: int = 0

    # ── Viewer ────────────────────────────────────────────────────────────────
    disable_viewer: bool = False

    # ── Checkpoint / export ───────────────────────────────────────────────────
    ckpt: Optional[List[str]] = None
    compression: Optional[Literal["png"]] = None
    render_traj_path: str = "interp"
    render_traj_trim: Optional[
        int
    ] = 5  # frames to drop from each end before path generation; None disables trim
    # ── Input / output paths ────────────────────────────────────────────────────────────
    model_dir: str = "outputs/sparse_model"
    images_dir: str = "outputs/images"
    masks_dir: Optional[str] = None
    splat_dir: str = "outputs/gs_2d"

    # ── Data ───────────────────────────────────────────────────────────────────────
    data_factor: int = 4
    test_every: int = 8
    patch_size: Optional[int] = None
    global_scale: float = 1.0
    normalize_world_space: bool = True
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # ── Viewer server ─────────────────────────────────────────────────────────
    port: int = 8080

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int = 1
    steps_scaler: float = 1.0
    max_steps: int = 30_000
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_ply: bool = False
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    disable_video: bool = False

    # ── Gaussian initialization ───────────────────────────────────────────────
    init_type: str = "sfm"
    init_num_pts: int = 100_000
    init_extent: float = 3.0
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1
    init_scale: float = 1.0
    ssim_lambda: float = 0.2
    model_type: Literal["2dgs", "2dgs-inria"] = "2dgs"  # Model for splatting.

    # ── Clipping planes ───────────────────────────────────────────────────────
    near_plane: float = 0.2
    far_plane: float = 200

    # ── Rasterization ─────────────────────────────────────────────────────────
    packed: bool = False  # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    sparse_grad: bool = False  # Use sparse gradients for optimization. (experimental)
    absgrad: bool = False  # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    antialiased: bool = False  # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    revised_opacity: bool = False  # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
    random_bkgd: bool = (
        False  # Use random background for training to discourage transparency
    )

    # ── Pruning ───────────────────────────────────────────────────────────
    prune_opa: float = 0.05  # GSs with opacity below this value will be pruned
    grow_grad2d: float = 0.0002  # GSs with image plane gradient above this value will be split/duplicated
    grow_scale3d: float = (
        0.01  # GSs with scale below this value will be duplicated. Above will be split
    )
    prune_scale3d: float = 0.1  # GSs with scale above this value will be pruned.

    # ── Refinement ───────────────────────────────────────────────────────────
    refine_start_iter: int = 500  # Start refining GSs after this iteration
    refine_stop_iter: int = 15_000  # Stop refining GSs after this iteration
    reset_every: int = 3000  # Reset opacities every this steps
    refine_every: int = 100  # Refine GSs every this steps

    # ── Camera Pose Optimization ────────────────────────────────────────────────────
    pose_opt: bool = False  # Enable camera optimization.
    pose_opt_lr: float = 1e-5  # Learning rate for camera optimization
    pose_opt_reg: float = 1e-6  # Regularization for camera optimization as weight decay
    pose_noise: float = 0.0  # Add noise to camera extrinsics. This is only to test the camera pose optimization.

    # ── Appearance Optimization ────────────────────────────────────────────────────
    app_opt: bool = False  # Enable appearance optimization. (experimental)
    app_embed_dim: int = 16  # Appearance embedding dimension
    app_opt_lr: float = 1e-3  # Learning rate for appearance optimization
    app_opt_reg: float = (
        1e-6  # Regularization for appearance optimization as weight decay
    )

    # ── Depth Loss ────────────────────────────────────────────────────────────
    depth_loss: bool = False  # Enable depth loss. (experimental)
    depth_lambda: float = 1e-2  # Weight for depth loss

    #  Background Alpha Loss (drives background Gaussians to be transparent, only works if masks are provided)
    bg_alpha_loss: bool = True
    bg_alpha_lambda: float = 0.5

    # ── Normal Consistency Loss ────────────────────────────────────────────────
    normal_loss: bool = (
        False  # Enable normal consistency loss. (Currently for 2DGS only)
    )
    normal_lambda: float = 5e-2  # Weight for normal loss
    normal_start_iter: int = (
        7_000  # Iteration to start normal consistency regulerization
    )

    # ── Distortion Loss ─────────────────────────────────────────────────────────
    dist_loss: bool = False  # Enable distortion loss. (experimental)
    dist_lambda: float = 1e-2  # Weight for distortion loss
    dist_start_iter: int = 3_000  # Iteration to start distortion loss regularization

    # ── Tensorboard ───────────────────────────────────────────────────────────
    tb_every: int = 100
    tb_save_image: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)
