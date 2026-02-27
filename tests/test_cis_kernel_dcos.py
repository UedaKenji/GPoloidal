from __future__ import annotations

import numpy as np

from gpoloidal.kernel import Kernel2D_scatter_grid


def test_create_obs_and_dcos_kernel_weighting_shapes_bounds_and_zero_rows():
    class DummyLength(np.ndarray):
        def __new__(cls, arr, im_shape):
            obj = np.asarray(arr, dtype=float).view(cls)
            obj.im_shape = im_shape
            return obj

    class DummyRay:
        def __init__(self):
            self.Length = DummyLength(np.array([[1.0, 0.0]]), (1, 2))
            self.Direction_xyz = np.array(
                [
                    [0.0, 1.0, 0.0],  # along +phi at x>0,y=0 midpoint -> dcos ~ +1
                    [1.0, 0.0, 0.0],
                ],
                dtype=float,
            )

        def generate_xyz(self, Lnum):
            # shape (M, Lnum)
            assert Lnum == 3
            M = 2
            X = np.zeros((M, Lnum), dtype=float)
            Y = np.zeros((M, Lnum), dtype=float)
            Z = np.zeros((M, Lnum), dtype=float)
            # ray 0: x=1, y varies from 0 to 0.2 (midpoints x>0)
            X[0, :] = 1.0
            Y[0, :] = np.linspace(0.0, 0.2, Lnum)
            Z[0, :] = 0.0
            # ray 1: arbitrary, but length=0 so row should vanish
            X[1, :] = 1.0
            Y[1, :] = 0.0
            Z[1, :] = 0.0
            return X, Y, Z

        def generate_rz(self, Lnum):
            X, Y, Z = self.generate_xyz(Lnum)
            R = np.sqrt(X**2 + Y**2)
            return R, Z

    k = Kernel2D_scatter_grid(vessel=None)  # type: ignore[arg-type]
    k.load_point(
        r_idc=np.array([0.9, 1.1]),
        z_idc=np.array([0.0, 0.0]),
        r_bd=np.array([0.8, 1.2]),
        z_bd=np.array([0.1, -0.1]),
        length_sq_fuction=Kernel2D_scatter_grid.constant_length_scale_sq_function(length_scale=0.2),
        is_plot=False,
    )
    H, Dcos = k.create_obs_and_dcos_kernel_weighting(DummyRay(), Lnum=2)
    assert H.shape == Dcos.shape == (2, 2)
    assert np.all(np.isfinite(H))
    assert np.all(np.isfinite(Dcos))
    assert np.all(np.abs(Dcos) <= 1.0 + 1e-12)
    # Zero-length ray should have zero contribution and zero Dcos row.
    assert np.allclose(H[1], 0.0)
    assert np.allclose(Dcos[1], 0.0)

