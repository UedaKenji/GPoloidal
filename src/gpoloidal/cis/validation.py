from __future__ import annotations

from typing import Callable

import numpy as np


def finite_difference_gradient(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    *,
    step: float = 1e-6,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = step
        g[i] = (func(x + dx) - func(x - dx)) / (2.0 * step)
    return g


def finite_difference_jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    step: float = 1e-6,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(func(x), dtype=float).reshape(-1)
    J = np.zeros((y0.size, x.size), dtype=float)
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = step
        yp = np.asarray(func(x + dx), dtype=float).reshape(-1)
        ym = np.asarray(func(x - dx), dtype=float).reshape(-1)
        J[:, i] = (yp - ym) / (2.0 * step)
    return J


def symmetric_part(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def max_abs(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(A, dtype=float))))


def relative_l2_error(x: np.ndarray, y: np.ndarray, *, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    den = np.linalg.norm(y) + eps
    return float(np.linalg.norm(x - y) / den)

