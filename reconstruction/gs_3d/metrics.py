import torch
import torch.nn.functional as F
from torch import nn


class GSS(nn.Module):
    """
    Gradient Structural Similarity (GSS) Metric.

    Measures how well the edge structures in the reconstructed image align with those in the reference image.
    Uses Gaussian smoothing to suppress noise and a specific gradient formulation.
    """

    def __init__(self, sigma: float = 1.5, kernel_size: int = 5):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Create Gaussian Kernel
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.0
        variance = sigma**2.0

        gaussian_kernel = (1.0 / (2.0 * torch.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
        )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        self.register_buffer("gaussian_kernel", gaussian_kernel.view(1, 1, kernel_size, kernel_size))

        # Gradient Kernels for 2nd derivative approximation as per description
        # Nx = Is(m, n+1) + Is(m, n-1) - 2Is(m, n) -> Kernel: [1, -2, 1] horizontal
        self.register_buffer("gx_kernel", torch.tensor([[1.0, -2.0, 1.0]]).view(1, 1, 1, 3))

        # Ny = Is(m+1, n) + Is(m-1, n) - 2Is(m, n) -> Kernel: [1, -2, 1] vertical
        self.register_buffer("gy_kernel", torch.tensor([[1.0], [-2.0], [1.0]]).view(1, 1, 3, 1))

    def forward(self, gt_rgb: torch.Tensor, predicted_rgb: torch.Tensor) -> torch.Tensor:
        """
        Compute GSS metric.
        Args:
            gt_rgb: Ground truth images (B, C, H, W)
            predicted_rgb: Predicted images (B, C, H, W)
        Returns:
            gss_score: Scalar GSS score
        """
        b, c, h, w = gt_rgb.shape

        # 1. Gaussian Smoothing
        # Expand kernel to match channels
        kernel = self.gaussian_kernel.expand(c, 1, self.kernel_size, self.kernel_size)

        # Zero padding to maintain original image dimensions
        gt_s = F.conv2d(gt_rgb, kernel, padding=self.padding, groups=c)
        pred_s = F.conv2d(predicted_rgb, kernel, padding=self.padding, groups=c)

        # 2. Image Gradients
        gx_k = self.gx_kernel.expand(c, 1, 1, 3)
        gy_k = self.gy_kernel.expand(c, 1, 3, 1)

        # Gradient X (horizontal) - Padding (0, 1) for 1x3 kernel
        gt_gx = F.conv2d(gt_s, gx_k, padding=(0, 1), groups=c)
        pred_gx = F.conv2d(pred_s, gx_k, padding=(0, 1), groups=c)

        # Gradient Y (vertical) - Padding (1, 0) for 3x1 kernel
        gt_gy = F.conv2d(gt_s, gy_k, padding=(1, 0), groups=c)
        pred_gy = F.conv2d(pred_s, gy_k, padding=(1, 0), groups=c)

        # 3. Gradient Magnitude
        gt_mag = torch.sqrt(gt_gx**2 + gt_gy**2)
        pred_mag = torch.sqrt(pred_gx**2 + pred_gy**2)

        # 4. GSS Computation
        # Sum over spatial dimensions (H, W)

        # Numerator: Sum |G(I) - G(I_hat)|
        diff = torch.abs(gt_mag - pred_mag)
        numerator = torch.sum(diff, dim=[2, 3])

        # Denominator: Sum (|G(I)| + |G(I_hat)|)
        denom_term = torch.abs(gt_mag) + torch.abs(pred_mag)
        denominator = torch.sum(denom_term, dim=[2, 3])

        # GSS = 1 - (Num / Denom)
        # Process per batch and channel, then scalarize
        gss_per_channel = 1.0 - (numerator / (denominator + 1e-8))

        return torch.mean(gss_per_channel)

import torch
import torch.nn.functional as F
from torch import nn


class LSS(nn.Module):
    """
    Laplacian Similarity Score (LSS) Metric.

    Measures second-order intensity variations to quantify curvature consistency and fine texture preservation.
    """

    def __init__(self, sigma: float = 1.5, kernel_size: int = 5):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Create Gaussian Kernel for smoothing
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.0
        variance = sigma**2.0

        gaussian_kernel = (1.0 / (2.0 * torch.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
        )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        self.register_buffer("gaussian_kernel", gaussian_kernel.view(1, 1, kernel_size, kernel_size))

        # Laplacian Kernel
        # delta I(m,n) = I(m+1, n) + I(m-1, n) + I(m, n+1) + I(m, n-1) - 4I(m, n)
        # This corresponds to the standard 3x3 Laplacian filter with center -4
        self.register_buffer(
            "laplacian_kernel",
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3),
        )

    def forward(self, gt_rgb: torch.Tensor, predicted_rgb: torch.Tensor) -> torch.Tensor:
        """
        Compute LSS metric.
        Args:
            gt_rgb: Ground truth images (B, C, H, W)
            predicted_rgb: Predicted images (B, C, H, W)
        Returns:
            lss_score: Scalar LSS score
        """
        b, c, h, w = gt_rgb.shape

        # 1. Gaussian Smoothing (Is = I * W_sigma)
        # Expand kernel to match channels
        g_kernel = self.gaussian_kernel.expand(c, 1, self.kernel_size, self.kernel_size)

        # Zero padding to maintain original image dimensions
        gt_s = F.conv2d(gt_rgb, g_kernel, padding=self.padding, groups=c)
        pred_s = F.conv2d(predicted_rgb, g_kernel, padding=self.padding, groups=c)

        # 2. Laplacian Operator
        l_kernel = self.laplacian_kernel.expand(c, 1, 3, 3)
        # Padding 1 for 3x3 kernel to maintain dimensions
        gt_lap = F.conv2d(gt_s, l_kernel, padding=1, groups=c)
        pred_lap = F.conv2d(pred_s, l_kernel, padding=1, groups=c)

        # 3. Normalization L(I) = delta I / (1 + Is)
        # 1 + Is acts as an intensity-dependent normalization term
        gt_L = gt_lap / (1.0 + gt_s)
        pred_L = pred_lap / (1.0 + pred_s)

        # 4. LSS Computation
        # Sum over spatial dimensions (H, W) per channel
        diff = torch.abs(gt_L - pred_L)
        numerator = torch.sum(diff, dim=[2, 3])

        denom_term = torch.abs(gt_L) + torch.abs(pred_L)
        denominator = torch.sum(denom_term, dim=[2, 3])

        # S_LSS = 1 - (Num / Denom)
        # Add epsilon to denominator to avoid division by zero
        lss_per_channel = 1.0 - (numerator / (denominator + 1e-8))

        return torch.mean(lss_per_channel)