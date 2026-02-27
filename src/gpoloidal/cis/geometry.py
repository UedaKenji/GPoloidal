from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sps

from .types import CISObservationGeometry


def toroidal_directional_cosine_from_midpoints(
    x_mid: np.ndarray,
    y_mid: np.ndarray,
    ray_dirs_xyz: np.ndarray,
    *,
    eps_r: float = 1e-10,
) -> np.ndarray:
    """Compute LOS directional cosine to the toroidal unit vector at ray midpoints."""
    x_mid = np.asarray(x_mid, dtype=float)
    y_mid = np.asarray(y_mid, dtype=float)
    ray_dirs_xyz = np.asarray(ray_dirs_xyz, dtype=float)
    if x_mid.shape != y_mid.shape:
        raise ValueError("x_mid and y_mid must have the same shape")
    if ray_dirs_xyz.ndim != 2 or ray_dirs_xyz.shape[1] != 3:
        raise ValueError("ray_dirs_xyz must have shape (M, 3)")
    if ray_dirs_xyz.shape[0] != x_mid.shape[0]:
        raise ValueError("ray_dirs_xyz row count must match midpoint row count")

    r_mid = np.sqrt(x_mid**2 + y_mid**2)
    valid = np.isfinite(r_mid) & (r_mid >= eps_r) & np.isfinite(x_mid) & np.isfinite(y_mid)

    ephi_x = np.zeros_like(x_mid, dtype=float)
    ephi_y = np.zeros_like(y_mid, dtype=float)
    ephi_x[valid] = -y_mid[valid] / r_mid[valid]
    ephi_y[valid] = x_mid[valid] / r_mid[valid]

    dcos = ray_dirs_xyz[:, [0]] * ephi_x + ray_dirs_xyz[:, [1]] * ephi_y
    dcos = np.clip(np.asarray(dcos, dtype=float), -1.0, 1.0)
    dcos[~valid] = 0.0
    return dcos


def coerce_observation_geometry(
    *,
    H: np.ndarray | sps.spmatrix,
    Dcos: np.ndarray | sps.spmatrix,
    mask: np.ndarray | None = None,
    im_shape: tuple[int, int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> CISObservationGeometry:
    if H.shape != Dcos.shape:
        raise ValueError(f"H and Dcos shapes must match, got {H.shape} vs {Dcos.shape}")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
    return CISObservationGeometry(H=H, Dcos=Dcos, mask=mask, im_shape=im_shape, metadata=dict(metadata or {}))


def build_kernel_weighting_geometry(
    *,
    kernel: Any,
    ray: Any,
    Lnum: int,
    mask: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> CISObservationGeometry:
    """Build ``CISObservationGeometry`` using kernel-weighting observation matrices."""
    H, Dcos = kernel.create_obs_and_dcos_kernel_weighting(ray=ray, Lnum=Lnum)
    im_shape = tuple(getattr(getattr(ray, "Length", None), "im_shape", ())) or None
    return coerce_observation_geometry(H=H, Dcos=Dcos, mask=mask, im_shape=im_shape, metadata=metadata)


def build_grid_binning_geometry(
    *,
    kernel: Any,
    ray: Any,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    sample_count: int = 400,
    column_mask: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> CISObservationGeometry:
    """Build ``CISObservationGeometry`` using grid-binned observation matrices."""
    H, Dcos = kernel.create_obs_and_dcos_grid_binning(
        ray=ray,
        r_grid=r_grid,
        z_grid=z_grid,
        sample_count=sample_count,
        column_mask=column_mask,
        sparse_output=False,
    )
    im_shape = tuple(getattr(getattr(ray, "Length", None), "im_shape", ())) or None
    return coerce_observation_geometry(H=H, Dcos=Dcos, mask=mask, im_shape=im_shape, metadata=metadata)
