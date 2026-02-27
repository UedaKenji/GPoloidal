from __future__ import annotations

import numpy as np

from gpoloidal.cis.forward import forward_cis_from_etv
from gpoloidal.cis.geometry import coerce_observation_geometry
from gpoloidal.cis.pipeline import CISStepwiseReconstructor
from gpoloidal.cis.types import CISObservedChannels


def test_cis_stepwise_reconstructor_smoke():
    rng = np.random.default_rng(0)
    H = np.abs(rng.normal(size=(7, 4))) + 0.05
    Dcos = rng.uniform(-0.8, 0.8, size=(7, 4))
    e_hat_true = np.array([-1.0, -1.2, -0.8, -1.1])
    T_true = np.array([0.2, 0.1, 0.3, 0.15])
    v_true = np.array([0.4, -0.1, 0.2, -0.2])
    I0, IRe, IIm = forward_cis_from_etv(H, Dcos, e_hat_true, T_true, v_true)
    sigma_I0 = np.full(H.shape[0], 0.03)
    sigma_I1 = np.full(H.shape[0], 0.02)
    obs = CISObservedChannels(
        I0_obs=I0 + sigma_I0 * rng.normal(size=H.shape[0]),
        IRe_obs=IRe + sigma_I1 * rng.normal(size=H.shape[0]),
        IIm_obs=IIm + sigma_I1 * rng.normal(size=H.shape[0]),
        sigma_I0=sigma_I0,
        sigma_I1=sigma_I1,
    )
    geom = coerce_observation_geometry(H=H, Dcos=Dcos)

    recon = CISStepwiseReconstructor(emit_max_iters=5, av_max_iters=6)
    emit_post, av_post, tv_post = recon.fit(
        geometry=geom,
        obs=obs,
        K_e_pri=np.eye(4) * 0.8,
        mu_e_pri=np.full(4, -1.0),
        K_T_pri=np.eye(4) * 0.5,
        mu_T_pri=np.zeros(4),
        K_v_pri=np.eye(4) * 0.5,
        mu_v_pri=np.zeros(4),
    )
    assert emit_post.mu_e.shape == (4,)
    assert av_post.mu_a.shape == (4,)
    assert av_post.mu_v.shape == (4,)
    assert tv_post.mu_T.shape == (4,)
    assert tv_post.mu_v.shape == (4,)
    assert np.all(np.isfinite(emit_post.mu_e))
    assert np.all(np.isfinite(tv_post.mu_T))
    assert np.all(np.isfinite(tv_post.mu_v))
    d = recon.diagnostics_as_dict()
    assert "emit_iters" in d and "av_iters" in d

