from __future__ import annotations

import numpy as np


def sort_profile_by_coordinate(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.argsort(x)
    return x[idx], y[idx]


def extract_band_profile(
    x_coord: np.ndarray,
    axis_coord: np.ndarray,
    values: np.ndarray,
    *,
    target: float,
    half_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and sort values whose axis coordinate lies in [target-half_width, target+half_width]."""
    x_coord = np.asarray(x_coord, dtype=float)
    axis_coord = np.asarray(axis_coord, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = (axis_coord >= target - half_width) & (axis_coord <= target + half_width)
    x_sorted, v_sorted = sort_profile_by_coordinate(x_coord[mask], values[mask])
    return x_sorted, v_sorted, mask


def lognormal_bands_from_latent(latent_mean: np.ndarray, latent_std: np.ndarray, nsigma: float = 1.0) -> dict[str, np.ndarray]:
    """Return mean/median and ±nsigma multiplicative bands for a latent Gaussian log-field."""
    m = np.asarray(latent_mean, dtype=float)
    s = np.asarray(latent_std, dtype=float)
    return {
        "mean": np.exp(m + 0.5 * s**2),
        "median": np.exp(m),
        "lower": np.exp(m - nsigma * s),
        "upper": np.exp(m + nsigma * s),
    }

