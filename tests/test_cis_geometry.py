from __future__ import annotations

import numpy as np

from gpoloidal.cis.geometry import coerce_observation_geometry, toroidal_directional_cosine_from_midpoints


def test_toroidal_directional_cosine_bounds_and_basic_orientation():
    # Midpoints on x-axis => e_phi = +y. Choose ray dirs along +y and -y.
    x_mid = np.array([[1.0, 1.0], [1.0, 1.0]])
    y_mid = np.array([[0.0, 0.0], [0.0, 0.0]])
    ray_dirs = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    dcos = toroidal_directional_cosine_from_midpoints(x_mid, y_mid, ray_dirs)
    assert dcos.shape == x_mid.shape
    assert np.all(dcos <= 1.0 + 1e-12)
    assert np.all(dcos >= -1.0 - 1e-12)
    assert np.allclose(dcos[0], 1.0)
    assert np.allclose(dcos[1], -1.0)


def test_toroidal_directional_cosine_zero_when_r_too_small():
    x_mid = np.array([[0.0, 1e-12]])
    y_mid = np.array([[0.0, 0.0]])
    ray_dirs = np.array([[1.0, 0.0, 0.0]])
    dcos = toroidal_directional_cosine_from_midpoints(x_mid, y_mid, ray_dirs, eps_r=1e-10)
    assert np.allclose(dcos, 0.0)


def test_coerce_observation_geometry_validates_shapes():
    H = np.zeros((3, 2))
    D = np.zeros((3, 2))
    g = coerce_observation_geometry(H=H, Dcos=D, im_shape=(1, 3))
    assert g.H.shape == (3, 2)
    assert g.Dcos.shape == (3, 2)

