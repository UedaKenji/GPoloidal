from __future__ import annotations

import numpy as np

from gpoloidal.cis.validation import finite_difference_gradient
from gpoloidal.tomography import GPT_cis_av_general


def _make_model(seed: int = 0) -> GPT_cis_av_general:
    rng = np.random.default_rng(seed)
    H = np.abs(rng.normal(size=(5, 3))) + 0.05
    Dcos = rng.uniform(-0.8, 0.8, size=(5, 3))
    model = GPT_cis_av_general(H=H, Dcos=Dcos)
    model.set_kernel(
        Ka_pri=np.eye(3) * 0.7,
        Kv_pri=np.eye(3) * 0.9,
        mua_pri=np.zeros(3),
        muv_pri=np.zeros(3),
    )
    a_true = np.array([-0.5, -0.2, -0.7])
    v_true = np.array([0.8, -0.4, 0.2])
    amp = np.exp(a_true)[None, :]
    phase = Dcos * v_true[None, :]
    IRe = np.sum(H * amp * np.cos(phase), axis=1)
    IIm = np.sum(H * amp * np.sin(phase), axis=1)
    model.set_obs(IRe_obs=IRe, IIm_obs=IIm, obs_noise_profile=np.ones(5), obs_noise_level=0.05)
    return model


def test_gpt_cis_av_general_smoke_update_postprocess():
    model = _make_model()
    a = np.array([-0.8, -0.8, -0.8])
    v = np.zeros(3)
    for _ in range(4):
        da, dv, loss = model.update(a, v)
        a = np.clip(a + 0.5 * da, -8.0, 8.0)
        v = np.clip(v + 0.5 * dv, -8.0, 8.0)
        assert np.isfinite(loss)
    model.postprocess(a, v, consider_w2=True)
    assert model.Kf_pos.shape == (6, 6)
    assert np.all(np.isfinite(model.sig_a_pos))
    assert np.all(np.isfinite(model.sig_v_pos))
    assert np.isfinite(model.mll)


def test_gpt_cis_av_general_gradient_matches_finite_difference():
    model = _make_model(seed=2)
    x0 = np.array([-0.3, -0.6, -0.2, 0.1, -0.2, 0.3], dtype=float)

    def f(x: np.ndarray) -> float:
        a = x[:3]
        v = x[3:]
        return float(model.log_posterior(a, v))

    g_fd = finite_difference_gradient(f, x0, step=1e-6)
    g, H = model.gradient_hessian(x0[:3], x0[3:], consider_w2=True)
    assert H.shape == (6, 6)
    assert np.allclose(H, H.T, atol=1e-8)
    assert np.allclose(g, g_fd, atol=5e-4, rtol=5e-3)


def test_gpt_cis_av_general_cross_blocks_are_transpose_after_postprocess():
    model = _make_model(seed=3)
    a = np.array([-0.5, -0.5, -0.5], dtype=float)
    v = np.zeros(3, dtype=float)
    for _ in range(3):
        da, dv, _ = model.update(a, v)
        a += 0.3 * da
        v += 0.3 * dv
    model.postprocess(a, v)
    assert np.allclose(model.K_av_pos, model.K_va_pos.T, atol=1e-8, rtol=1e-6)

