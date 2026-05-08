import numpy as np


def fit_circle(
    points: list[tuple[float, float]],
) -> tuple[float, float, float] | None:
    """
    Algebraic least-squares circle fit to 2D points.

    Solves (u - cx)^2 + (v - cy)^2 = r^2 in the form:
        [2u  2v  1] [cx; cy; D] = u^2 + v^2,   D = cx^2 + cy^2 - r^2

    Returns (cx, cy, r) or None if fewer than 3 non-collinear points.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return None

    u, v = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * u, 2 * v, np.ones(len(u))])
    b = u**2 + v**2

    x, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        return None

    cx, cy = float(x[0]), float(x[1])
    r_sq = cx**2 + cy**2 - float(x[2])
    if r_sq < 0:
        return None

    return cx, cy, float(np.sqrt(r_sq))
