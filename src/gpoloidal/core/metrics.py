from __future__ import annotations

import numpy as np


def safe_corrcoef(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) < eps or np.std(y) < eps:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def field_metrics(f_est: np.ndarray, f_true: np.ndarray, *, prefix: str) -> dict[str, float]:
    f_est = np.asarray(f_est, dtype=float)
    f_true = np.asarray(f_true, dtype=float)
    diff = f_est - f_true
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(diff**2))),
        f"{prefix}_mae": float(np.mean(np.abs(diff))),
        f"{prefix}_rel_rmse": float(np.linalg.norm(diff) / (np.linalg.norm(f_true) + 1e-12)),
        f"{prefix}_corr": safe_corrcoef(f_est, f_true),
        f"{prefix}_neg_frac": float(np.mean(f_est < 0)),
    }


def mean_chi2(g_pred: np.ndarray, g_obs: np.ndarray, obs_noise_std: np.ndarray) -> float:
    g_pred = np.asarray(g_pred, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)
    obs_noise_std = np.asarray(obs_noise_std, dtype=float)
    return float(np.mean(((g_pred - g_obs) / (obs_noise_std + 1e-12)) ** 2))


def weighted_mae(y_est: np.ndarray, y_true: np.ndarray, weights: np.ndarray, eps: float = 1e-12) -> float:
    y_est = np.asarray(y_est, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(np.sum(np.abs(y_est - y_true) * w) / (np.sum(w) + eps))


def weighted_rmse(y_est: np.ndarray, y_true: np.ndarray, weights: np.ndarray, eps: float = 1e-12) -> float:
    y_est = np.asarray(y_est, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.sum((y_est - y_true) ** 2 * w) / (np.sum(w) + eps)))

