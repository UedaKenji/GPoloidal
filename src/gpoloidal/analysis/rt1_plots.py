from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_rt1_truth_and_observation(
    *,
    kernel,
    flux_kernel,
    f_true: np.ndarray,
    g_obs: np.ndarray,
    resolution: tuple[int, int],
    truth_title: str = "Truth (inducing -> grid)",
    obs_title: str = "Observed image g_obs",
) -> plt.Figure:
    """Create a 2-panel figure for truth field and observation image."""
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    axs[0].imshow(
        kernel.convert_grid(np.asarray(f_true, dtype=float), boundary=0.0) * kernel.mask,
        **kernel.im_kwargs,
        cmap="turbo",
    )
    flux_kernel.plt_rt1_flux(ax=axs[0], linewidths=0.7)
    axs[0].set_title(truth_title)
    axs[1].imshow(np.asarray(g_obs, dtype=float).reshape(resolution), origin="upper", cmap="magma")
    axs[1].set_title(obs_title)
    return fig


def plot_rt1_reconstruction_panels(
    *,
    kernel,
    flux_kernel,
    f_true: np.ndarray,
    f_recon: np.ndarray,
    f_std: np.ndarray,
    fit_label: str,
    value_vmin: float | None = 0.0,
    value_vmax: float | None = 1.0,
    error_percentile: float = 99.0,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    """Create 2x2 reconstruction panels and return plotted grid arrays."""
    truth_grid = kernel.convert_grid(np.asarray(f_true, dtype=float), boundary=0.0) * kernel.mask
    recon_grid = kernel.convert_grid(np.asarray(f_recon, dtype=float), boundary=0.0) * kernel.mask
    err_grid = kernel.convert_grid(np.asarray(f_recon, dtype=float) - np.asarray(f_true, dtype=float), boundary=0.0) * kernel.mask
    std_grid = kernel.convert_grid(np.asarray(f_std, dtype=float), boundary=0.0) * kernel.mask

    if value_vmin is None or value_vmax is None:
        stacked = np.concatenate([
            np.ravel(np.asarray(truth_grid, dtype=float)),
            np.ravel(np.asarray(recon_grid, dtype=float)),
        ])
        finite = stacked[np.isfinite(stacked)]
        if finite.size == 0:
            auto_vmin, auto_vmax = 0.0, 1.0
        else:
            auto_vmin = float(np.nanmin(finite))
            auto_vmax = float(np.nanmax(finite))
            if not np.isfinite(auto_vmin):
                auto_vmin = 0.0
            if not np.isfinite(auto_vmax) or auto_vmax <= auto_vmin:
                auto_vmax = max(auto_vmin + 1e-3, 1.0)
        if value_vmin is None:
            value_vmin = auto_vmin
        if value_vmax is None:
            value_vmax = auto_vmax

    fig, axs = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    im_truth = axs[0, 0].imshow(truth_grid, **kernel.im_kwargs, cmap="turbo", vmin=value_vmin, vmax=value_vmax)
    axs[0, 0].set_title("Truth")
    im_recon = axs[0, 1].imshow(recon_grid, **kernel.im_kwargs, cmap="turbo", vmin=value_vmin, vmax=value_vmax)
    axs[0, 1].set_title(f"{fit_label} reconstruction")

    vmax = float(np.nanpercentile(np.abs(err_grid), error_percentile)) if np.any(np.isfinite(err_grid)) else 1.0
    vmax = max(vmax, 1e-3)
    try:
        from gpoloidal.plot_utils import cmocean_balance as _cm_balance
        err_cmap = _cm_balance
    except Exception:
        err_cmap = "RdBu_r"
    im_err = axs[1, 0].imshow(err_grid, **kernel.im_kwargs, cmap=err_cmap, vmin=-vmax, vmax=vmax)
    axs[1, 0].set_title("Error")
    im_std = axs[1, 1].imshow(std_grid, **kernel.im_kwargs, cmap="viridis")
    axs[1, 1].set_title("Posterior std")

    for ax in axs.flat:
        flux_kernel.plt_rt1_flux(ax=ax, linewidths=0.6)
    for ax, im in ((axs[0, 0], im_truth), (axs[0, 1], im_recon), (axs[1, 0], im_err), (axs[1, 1], im_std)):
        fig.colorbar(im, ax=ax, shrink=0.9)

    return fig, {"truth_grid": truth_grid, "recon_grid": recon_grid, "err_grid": err_grid, "std_grid": std_grid}


def plot_loggp_loss_history(*, loss_history: list[float] | np.ndarray) -> plt.Figure:
    """Create a simple logGP loss history plot."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.2), constrained_layout=True)
    ax.plot(np.asarray(loss_history, dtype=float), marker="o", ms=3)
    ax.set_title("logGP loss history")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    return fig
