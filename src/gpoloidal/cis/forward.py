from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sps


def _as_dense_matrix(mat: np.ndarray | sps.spmatrix) -> np.ndarray:
    if sps.issparse(mat):
        return np.asarray(mat.toarray(), dtype=float)
    return np.asarray(mat, dtype=float)


def forward_emit(H: np.ndarray | sps.spmatrix, e_hat: np.ndarray) -> np.ndarray:
    """Forward model for line-integrated intensity ``I0 = H exp(e_hat)``."""
    e_hat = np.asarray(e_hat, dtype=float)
    if sps.issparse(H):
        return np.asarray(H @ np.exp(e_hat), dtype=float).reshape(-1)
    return np.asarray(_as_dense_matrix(H) @ np.exp(e_hat), dtype=float).reshape(-1)


def forward_cis_from_av(
    H: np.ndarray | sps.spmatrix,
    Dcos: np.ndarray | sps.spmatrix,
    a_hat: np.ndarray,
    v_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward model for CIS real/imaginary channels from local amplitude and velocity.

    Computes
      IRe_i = sum_j H_ij exp(a_j) cos(Dcos_ij v_j)
      IIm_i = sum_j H_ij exp(a_j) sin(Dcos_ij v_j)
    """
    H_d = _as_dense_matrix(H)
    D_d = _as_dense_matrix(Dcos)
    if H_d.shape != D_d.shape:
        raise ValueError(f"H and Dcos must have the same shape, got {H_d.shape} vs {D_d.shape}")

    a_hat = np.asarray(a_hat, dtype=float).reshape(-1)
    v_hat = np.asarray(v_hat, dtype=float).reshape(-1)
    if H_d.shape[1] != a_hat.size or H_d.shape[1] != v_hat.size:
        raise ValueError(
            f"Column size mismatch: H has {H_d.shape[1]} columns but a_hat={a_hat.size}, v_hat={v_hat.size}"
        )

    amp = np.exp(a_hat)[None, :]
    phase = D_d * v_hat[None, :]
    weighted = H_d * amp
    IRe = np.sum(weighted * np.cos(phase), axis=1)
    IIm = np.sum(weighted * np.sin(phase), axis=1)
    return np.asarray(IRe, dtype=float), np.asarray(IIm, dtype=float)


def forward_cis_from_etv(
    H: np.ndarray | sps.spmatrix,
    Dcos: np.ndarray | sps.spmatrix,
    e_hat: np.ndarray,
    T_hat: np.ndarray,
    v_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward model for CIS channels from log-emissivity, temperature, velocity."""
    e_hat = np.asarray(e_hat, dtype=float)
    T_hat = np.asarray(T_hat, dtype=float)
    if e_hat.shape != T_hat.shape:
        raise ValueError("e_hat and T_hat must have the same shape")
    a_hat = e_hat - T_hat
    I0 = forward_emit(H, e_hat)
    IRe, IIm = forward_cis_from_av(H, Dcos, a_hat, np.asarray(v_hat, dtype=float))
    return I0, IRe, IIm


def stack_cis_channels(IRe: np.ndarray, IIm: np.ndarray) -> np.ndarray:
    return np.hstack((np.asarray(IRe, dtype=float).reshape(-1), np.asarray(IIm, dtype=float).reshape(-1)))


def projected_temperature_velocity_proxies(
    *,
    I0: np.ndarray,
    IRe: np.ndarray,
    IIm: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Diagnostic proxy images used in notebook-style CIS visualization."""
    I0 = np.asarray(I0, dtype=float)
    IRe = np.asarray(IRe, dtype=float)
    IIm = np.asarray(IIm, dtype=float)
    amp = np.sqrt(IRe**2 + IIm**2)
    T_proxy = -np.log(np.clip(amp, eps, None) / np.clip(np.abs(I0), eps, None))
    v_proxy = np.arctan2(IIm, IRe)
    return {
        "amplitude": amp,
        "temperature_proxy": T_proxy,
        "velocity_proxy": v_proxy,
    }


def channels_as_image_dict(
    *,
    I0: np.ndarray,
    IRe: np.ndarray,
    IIm: np.ndarray,
    im_shape: tuple[int, int] | None,
) -> dict[str, Any]:
    """Small helper for plotting/report code."""
    if im_shape is None:
        return {"I0": I0, "IRe": IRe, "IIm": IIm}
    return {
        "I0": np.asarray(I0).reshape(im_shape),
        "IRe": np.asarray(IRe).reshape(im_shape),
        "IIm": np.asarray(IIm).reshape(im_shape),
    }

