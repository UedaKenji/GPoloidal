from __future__ import annotations

import numpy as np

from gpoloidal.rt1 import phantom


def test_double_phantom_applies_upper_z_mask_and_returns_finite_values():
    f = phantom.get_phantom_funtion("double")
    r = np.array([0.7, 0.7, 0.7], dtype=float)
    z = np.array([0.0, 0.49, 0.6], dtype=float)

    y = np.asarray(f(r, z), dtype=float)

    assert y.shape == z.shape
    assert np.all(np.isfinite(y))
    assert y[1] == 0.0
    assert y[2] == 0.0
