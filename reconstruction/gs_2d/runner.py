"""Runner — orchestrates setup, rasterization, and the training loop for 2DGS.
"""

import math
import os
import time

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import tqdm
import viser
import yaml

from datasets.colmap import Dataset, Parser
from gsplat.compression import PngCompression
from gsplat.rendering import rasterization_2dgs, rasterization_2dgs_inria_wrapper
from gsplat.strategy import DefaultStrategy
from nerfview import CameraState, RenderTabState, apply_float_colormap
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import StructuralSimilarityIndexMeasure
from visualizers.gsplat_viewer_2dgs import GsplatRenderTabState2D, GsplatViewer2D

from .config import Config
from .evaluator import Evaluator
from .logger import TrainingLogger
from .model import create_splats_with_optimizers
from .utils import AppearanceOptModule, CameraOptModule, set_random_seed


class Runner:
    """Orchestrates model setup and the training loop for 2DGS."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(cfg.random_seed)

        self.cfg = cfg
        self.device = "cuda"

        self._setup_dirs()
        self._load_data()
        self._build_model()
        self._build_aux_modules()
        self._build_viewer()

        self.logger = TrainingLogger(
            cfg=cfg,
            splats=self.splats,
            stats_dir=self.stats_dir,
            ckpt_dir=self.ckpt_dir,
            ply_dir=self.ply_dir,
            writer=self.writer,
        )
        self.evaluator = Evaluator(
            cfg=cfg,
            rasterize_fn=self._rasterize_splats,
            splats=self.splats,
            valset=self.valset,
            parser=self.parser,
            scene_scale=self.scene_scale,
            render_dir=self.render_dir,
            stats_dir=self.stats_dir,
            writer=self.writer,
            device=self.device,
            compression_method=self.compression_method,
        )

    def _setup_dirs(self) -> None:
        cfg = self.cfg
        for subdir in ("", "ckpts", "stats", "renders", "ply"):
            path = os.path.join(cfg.splat_dir, subdir)
            os.makedirs(path, exist_ok=True)
        self.ckpt_dir = f"{cfg.splat_dir}/ckpts"
        self.stats_dir = f"{cfg.splat_dir}/stats"
        self.render_dir = f"{cfg.splat_dir}/renders"
        self.ply_dir = f"{cfg.splat_dir}/ply"
        self.writer = SummaryWriter(log_dir=f"{cfg.splat_dir}/tb")

    def _load_data(self) -> None:
        cfg = self.cfg
        self.parser = Parser(
            colmap_dir=cfg.model_dir,
            images_dir=cfg.images_dir,
            masks_dir=cfg.masks_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

    def _build_model(self) -> None:
        cfg = self.cfg
        feature_dim = 32 if cfg.app_opt else None

        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
        )
        print("Model initialised. Number of GS:", len(self.splats["means"]))
        self.model_type = cfg.model_type

        if self.model_type == "2dgs":
            key_for_gradient = "gradient_2dgs"
        else:
            key_for_gradient = "means2d"

        # Densification Strategy
        self.strategy = DefaultStrategy(
            verbose=True,
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d,
            prune_scale3d=cfg.prune_scale3d,
            # refine_scale2d_stop_iter=4000, # splatfacto behavior
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter,
            reset_every=cfg.reset_every,
            refine_every=cfg.refine_every,
            absgrad=cfg.absgrad,
            revised_opacity=cfg.revised_opacity,
            key_for_gradient=key_for_gradient,
        )

        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()

        self.compression_method = None
        if cfg.compression == "png":
            self.compression_method = PngCompression()

    def _build_aux_modules(self) -> None:
        cfg = self.cfg
        self.pose_optimizers: List[torch.optim.Optimizer] = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers: List[torch.optim.Optimizer] = []
        feature_dim = 32 if cfg.app_opt else None
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

    def _build_viewer(self) -> None:
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=self.cfg.port, verbose=False)
            self.viewer = GsplatViewer2D(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(self.cfg.splat_dir),
                mode="training",
            )

    def _rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        assert self.cfg.antialiased is False, "Antialiased is not supported for 2DGS"

        if self.model_type == "2dgs":
            (
                render_colors,
                render_alphas,
                render_normals,
                normals_from_depth,
                render_distort,
                render_median,
                info,
            ) = rasterization_2dgs(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
        elif self.model_type == "2dgs-inria":
            renders, info = rasterization_2dgs_inria_wrapper(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
            render_colors, render_alphas = renders
            render_normals = info["normals_rend"]
            normals_from_depth = info["normals_surf"]
            render_distort = info["render_distloss"]
            render_median = render_colors[..., 3]

        if masks is not None:
            render_colors = render_colors * masks[..., None]
        return (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        )

    rasterize_splats = _rasterize_splats

    def _build_schedulers(self, max_steps: int) -> List:
        cfg = self.cfg
        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        return schedulers

    def _train_step(self, step: int, data: dict, schedulers: List) -> dict:
        cfg = self.cfg
        device = self.device

        camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
        Ks = data["K"].to(device)  # [1, 3, 3]
        pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
        num_train_rays_per_step = pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
        image_ids = data["image_id"].to(device)
        masks = data["mask"].to(device) if "mask" in data else None
        if cfg.depth_loss:
            points = data["points"].to(device)  # [1, M, 2]
            depths_gt = data["depths"].to(device)  # [1, M]

        height, width = pixels.shape[1:3]

        if cfg.pose_noise:
            camtoworlds = self.pose_perturb(camtoworlds, image_ids)

        if cfg.pose_opt:
            camtoworlds = self.pose_adjust(camtoworlds, image_ids)

        # sh schedule
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

        # forward
        (
            renders,
            alphas,
            normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = self._rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            image_ids=image_ids,
            render_mode="RGB+ED" if cfg.depth_loss else "RGB+D",
            distloss=self.cfg.dist_loss,
            masks=masks,
        )
        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3:4]
        else:
            colors, depths = renders, None

        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=device)
            colors = colors + bkgd * (1.0 - alphas)

        self.strategy.step_pre_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=step,
            info=info,
        )

        if masks is not None:
            pixels = pixels * masks[..., None]
            colors = colors * masks[..., None]

        # ── Loss ──────────────────────────────────────────────────────────
        if masks is not None:
            l1loss = F.l1_loss(colors[masks], pixels[masks])
            colors_ssim = colors * masks[..., None]
            pixels_ssim = pixels * masks[..., None]
        else:
            l1loss = F.l1_loss(colors, pixels)
            colors_ssim = colors
            pixels_ssim = pixels
        ssimloss = 1.0 - self.ssim(
            colors_ssim.permute(0, 3, 1, 2),
            pixels_ssim.permute(0, 3, 1, 2),
        )
        loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)

        bg_alpha_loss = None
        if masks is not None and cfg.bg_alpha_loss:
            bg_alpha_loss = alphas[~masks].mean()
            loss = loss + (
                bg_alpha_loss * cfg.bg_alpha_lambda
                if cfg.bg_alpha_lambda > 0.0
                else 0.0
            )

        depthloss = None
        if cfg.depth_loss:
            # query depths from depth map
            points = torch.stack(
                [
                    points[:, :, 0] / (width - 1) * 2 - 1,
                    points[:, :, 1] / (height - 1) * 2 - 1,
                ],
                dim=-1,
            )  # normalize to [-1, 1]
            grid = points.unsqueeze(2)  # [1, M, 1, 2]
            depths = F.grid_sample(
                depths.permute(0, 3, 1, 2), grid, align_corners=True
            )  # [1, 1, M, 1]
            depths = depths.squeeze(3).squeeze(1)  # [1, M]
            # calculate loss in disparity space
            disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
            disp_gt = 1.0 / depths_gt  # [1, M]
            depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
            loss += depthloss * cfg.depth_lambda

        if cfg.normal_loss:
            if step > cfg.normal_start_iter:
                curr_normal_lambda = cfg.normal_lambda
            else:
                curr_normal_lambda = 0.0
            # normal consistency loss
            normals = normals.squeeze(0).permute((2, 0, 1))
            normals_from_depth = normals_from_depth * alphas.squeeze(0).detach()
            if len(normals_from_depth.shape) == 4:
                normals_from_depth = normals_from_depth.squeeze(0)
            normals_from_depth = normals_from_depth.permute((2, 0, 1))
            normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
            normalloss = curr_normal_lambda * normal_error.mean()
            loss += normalloss

        if cfg.dist_loss:
            if step > cfg.dist_start_iter:
                curr_dist_lambda = cfg.dist_lambda
            else:
                curr_dist_lambda = 0.0
            distloss = render_distort.mean()
            loss += distloss * curr_dist_lambda

        loss.backward()

        # ── Progress-bar description ───────────────────────────────────────
        desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
        if cfg.depth_loss:
            desc += f"depth loss={depthloss.item():.6f}| "
        if cfg.dist_loss:
            desc += f"dist loss={distloss.item():.6f}"
        if cfg.pose_opt and cfg.pose_noise:
            # monitor the pose error if we inject noise
            pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
            desc += f"pose err={pose_err.item():.6f}| "

        # ── Strategy post-backward ─────────────────────────────────────────
        self.strategy.step_post_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=step,
            info=info,
            packed=cfg.packed,
        )

        # ── Sparsify gradients (experimental) ─────────────────────────────
        if cfg.sparse_grad:
            assert cfg.packed, "Sparse gradients only work with packed mode."
            gaussian_ids = info["gaussian_ids"]
            for k in self.splats.keys():
                grad = self.splats[k].grad
                if grad is None or grad.is_sparse:
                    continue
                self.splats[k].grad = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=grad[gaussian_ids],  # [nnz, ...]
                    size=self.splats[k].size(),  # [N, ...]
                    is_coalesced=len(Ks) == 1,
                )

        # optimize
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for optimizer in self.pose_optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for optimizer in self.app_optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()

        return {
            "desc": desc,
            "num_rays": num_train_rays_per_step,
            "sh_degree": sh_degree_to_use,
            "loss": loss,
            "l1loss": l1loss,
            "ssimloss": ssimloss,
            "depthloss": depthloss,
            "bg_alpha_loss": bg_alpha_loss,
            "pixels": pixels,
            "colors": colors,
        }

    def train(self) -> None:
        cfg = self.cfg
        with open(f"{cfg.splat_dir}/cfg.yml", "w") as f:
            yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        schedulers = self._build_schedulers(max_steps)

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Precompute step sets for O(1) membership checks
        save_at = {i - 1 for i in cfg.save_steps}
        ply_at = {i - 1 for i in cfg.ply_steps}
        eval_at = {i - 1 for i in cfg.eval_steps}

        global_tic = time.time()
        pbar = tqdm.tqdm(range(max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            out = self._train_step(step, data, schedulers)
            pbar.set_description(out["desc"])

            # ── Logging ───────────────────────────────────────────────────
            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                self.logger.log_step(
                    step,
                    loss=out["loss"],
                    l1loss=out["l1loss"],
                    ssimloss=out["ssimloss"],
                    sh_degree=out["sh_degree"],
                    pixels=out["pixels"],
                    colors=out["colors"],
                    depthloss=out["depthloss"],
                    bg_alpha_loss=out["bg_alpha_loss"],
                )

            # ── Checkpointing ─────────────────────────────────────────────
            if step in save_at or step == max_steps - 1:
                self.logger.save_checkpoint(
                    step,
                    elapsed=time.time() - global_tic,
                    pose_adjust=getattr(self, "pose_adjust", None),
                    app_module=getattr(self, "app_module", None),
                )
            if (step in ply_at or step == max_steps - 1) and cfg.save_ply:
                self.logger.export_ply(
                    step,
                    out["sh_degree"],
                    app_module=getattr(self, "app_module", None),
                )

            # ── Evaluation ────────────────────────────────────────────────
            if step in eval_at or step == max_steps - 1:
                self.evaluator.eval(step)
                self.evaluator.render_traj(step)
                if cfg.compression is not None:
                    self.evaluator.run_compression(step)

            # ── Viewer update ─────────────────────────────────────────────
            if not cfg.disable_viewer:
                self.viewer.lock.release()
                steps_per_sec = 1.0 / max(time.time() - tic, 1e-10)
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    out["num_rays"] * steps_per_sec
                )
                self.viewer.update(step, out["num_rays"])

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState2D)
        if render_tab_state.preview_render:
            width, height = (
                render_tab_state.render_width,
                render_tab_state.render_height,
            )
        else:
            width, height = (
                render_tab_state.viewer_width,
                render_tab_state.viewer_height,
            )

        c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
        K = (
            torch.from_numpy(camera_state.get_K((width, height)))
            .float()
            .to(self.device)
        )

        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = self._rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            render_mode="RGB+ED",
            backgrounds=(
                torch.tensor([render_tab_state.backgrounds], device=self.device) / 255.0
            ),
        )
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "depth":
            # normalize depth to [0, 1]
            depth = render_median
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "normal":
            render_normals = render_normals * 0.5 + 0.5  # normalize to [0, 1]
            renders = render_normals.cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        else:
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        return renders
