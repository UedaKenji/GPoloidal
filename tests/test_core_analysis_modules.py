from __future__ import annotations

import numpy as np

from gpoloidal.analysis.hparam_sweep import SweepGrid, pivot_metric, run_2d_hparam_sweep
from gpoloidal.analysis.profiles import extract_band_profile, lognormal_bands_from_latent, sort_profile_by_coordinate
from gpoloidal.core.kernel import Kernel2D_scatter_grid
from gpoloidal.core.metrics import field_metrics, mean_chi2, weighted_mae, weighted_rmse


def test_core_metrics_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_est = np.array([1.0, 1.0, 4.0])
    m = field_metrics(y_est, y_true, prefix="t")
    assert set(["t_rmse", "t_mae", "t_rel_rmse", "t_corr", "t_neg_frac"]) <= set(m)
    assert m["t_neg_frac"] == 0.0

    sigma = np.ones_like(y_true) * 0.5
    chi2 = mean_chi2(y_est, y_true, sigma)
    assert chi2 > 0

    w = np.array([1.0, 2.0, 1.0])
    assert weighted_mae(y_est, y_true, w) > 0
    assert weighted_rmse(y_est, y_true, w) > 0


def test_analysis_profiles_helpers():
    x = np.array([0.4, 0.2, 0.3])
    y = np.array([4.0, 2.0, 3.0])
    xs, ys = sort_profile_by_coordinate(x, y)
    assert np.allclose(xs, [0.2, 0.3, 0.4])
    assert np.allclose(ys, [2.0, 3.0, 4.0])

    r = np.array([0.2, 0.3, 0.4, 0.5])
    z = np.array([0.0, 0.01, 0.03, 0.2])
    v = np.array([2.0, 3.0, 4.0, 5.0])
    xr, vr, mask = extract_band_profile(r, z, v, target=0.02, half_width=0.02)
    assert mask.sum() == 3
    assert np.allclose(xr, [0.2, 0.3, 0.4])
    assert np.allclose(vr, [2.0, 3.0, 4.0])

    bands = lognormal_bands_from_latent(np.array([0.0]), np.array([0.1]), nsigma=2.0)
    assert set(bands.keys()) == {"mean", "median", "lower", "upper"}
    assert float(bands["upper"][0]) > float(bands["lower"][0])


def test_analysis_hparam_sweep_helpers():
    grid = SweepGrid("length_scale", (0.04, 0.05), "obs_noise_level", (0.1, 0.2))

    def evaluator(length_scale, obs_noise_level):
        return {"mll": -(length_scale - 0.05) ** 2 - (obs_noise_level - 0.1) ** 2}

    df = run_2d_hparam_sweep(grid=grid, evaluator=evaluator)
    assert df.shape[0] == 4
    piv = pivot_metric(df, grid=grid, metric="mll")
    assert list(piv.index) == [0.04, 0.05]
    assert list(piv.columns) == [0.1, 0.2]


def test_core_grid_inducing_helpers():
    f = Kernel2D_scatter_grid.constant_length_scale_sq_function(length_scale=0.05)
    out = f(np.array([0.1, 0.2]), np.array([0.0, 0.1]))
    assert np.allclose(out, 0.05**2)


def test_core_grid_inducing_kernel_adapters():
    class DummyScatter(Kernel2D_scatter_grid):
        def __init__(self):
            super().__init__(vessel=None)  # type: ignore[arg-type]

        def load_point(self, **kwargs):
            self.kwargs = kwargs  # type: ignore[attr-defined]

    scatter = DummyScatter()
    r_grid = np.array([0.0, 0.1, 0.2])
    z_grid = np.array([-0.1, 0.0, 0.1])
    fill = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])
    r_idc, z_idc, r_bd, z_bd = scatter.set_inducing_point_from_grid_fill(
        r_grid=r_grid,
        z_grid=z_grid,
        fill=fill,
        length_scale=0.03,
    )
    assert r_idc.size == 1
    assert z_idc.size == 1
    assert r_bd.size >= 4 and z_bd.size >= 4
    assert set(["r_idc", "z_idc", "r_bd", "z_bd", "length_sq_fuction"]) <= set(scatter.kwargs)

    class DummyGridKernel(Kernel2D_scatter_grid):
        def __init__(self):
            super().__init__(vessel=None)  # type: ignore[arg-type]

        def set_uniform_kernel(self, **kwargs):  # type: ignore[override]
            self.kwargs = kwargs
            return "grid"

    class DummyLegacyKernel(Kernel2D_scatter_grid):
        def __init__(self):
            super().__init__(vessel=None)  # type: ignore[arg-type]

        def set_unifom_kernel(self, **kwargs):
            self.kwargs = kwargs
            return "legacy"

    gk = DummyGridKernel()
    assert gk.set_uniform_kernel(length_scale=0.1, static=False) == "grid"
    assert gk.kwargs["length_scale"] == 0.1

    lk = DummyLegacyKernel()
    # Use inherited alias to hit legacy set_unifom_kernel
    assert Kernel2D_scatter_grid.set_uniform_kernel(lk, length_scale=0.2, is_static_kernel=False) == "legacy"
    assert lk.kwargs["length_scale"] == 0.2


def test_core_grid_obsmatrix_builder_dense_and_masked_sparse():
    class DummyLength(np.ndarray):
        def __new__(cls, arr, im_shape):
            obj = np.asarray(arr, dtype=float).view(cls)
            obj.im_shape = im_shape
            return obj

    class DummyRay:
        def __init__(self):
            self.Length = DummyLength(np.array([[2.0, 2.0], [2.0, 2.0]]), (2, 2))

        def generate_rz(self, Lnum):
            # sample-major image shape: (sample_count, ny, nx)
            # each ray stays in a single cell so accumulated weight should be Length.
            assert Lnum == 3
            R = np.zeros((Lnum, 2, 2), dtype=float)
            Z = np.zeros((Lnum, 2, 2), dtype=float)
            # cells: (z,r) = (0,0), (0,1), (1,0), (1,1)
            R[:, 0, 0] = 0.1
            Z[:, 0, 0] = 0.1
            R[:, 0, 1] = 0.9
            Z[:, 0, 1] = 0.1
            R[:, 1, 0] = 0.1
            Z[:, 1, 0] = 0.9
            R[:, 1, 1] = 0.9
            Z[:, 1, 1] = 0.9
            return R, Z

    ray = DummyRay()
    r_grid = np.array([0.0, 1.0])
    z_grid = np.array([0.0, 1.0])

    kernel = Kernel2D_scatter_grid(vessel=None)  # type: ignore[arg-type]
    res = kernel.create_obs_matrix_grid_binning(
        ray,
        r_grid=r_grid,
        z_grid=z_grid,
        sample_count=3,
        sparse_output=False,
        return_grid4d=True,
        show_progress=False,
    )
    H_flat, H_grid4d = res
    assert isinstance(H_flat, np.ndarray)
    assert H_flat.shape == (4, 4)
    assert np.allclose(H_flat, np.eye(4) * 2.0)
    assert H_grid4d.shape == (2, 2, 2, 2)

    keep = np.array([[True, False], [False, True]])
    H2_sparse = kernel.create_obs_matrix_grid_binning(
        ray,
        r_grid=r_grid,
        z_grid=z_grid,
        sample_count=3,
        column_mask=keep,
        sparse_output=True,
        show_progress=False,
    )
    H2 = H2_sparse.toarray()
    assert H2.shape == (4, 2)
    assert np.allclose(H2[[0, 3], [0, 1]], [2.0, 2.0])
    assert np.allclose(H2[[1, 2]], 0.0)
