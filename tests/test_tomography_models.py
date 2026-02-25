from __future__ import annotations

import numpy as np

from gpoloidal.tomography import GPT_lin_general, GPT_log_general


def test_gpt_lin_general_matches_closed_form_reference():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(9, 5))
    K = rng.normal(size=(5, 5))
    K = K @ K.T + 0.3 * np.eye(5)
    mu = rng.normal(size=5)
    obs_noise_std = np.full(H.shape[0], 0.25)
    g_obs = rng.normal(size=H.shape[0])

    # Reference closed-form solution
    K_inv = np.linalg.inv(K)
    w = 1.0 / obs_noise_std
    sigiH = w[:, None] * H
    A = sigiH.T @ sigiH + K_inv
    rhs = sigiH.T @ (w * g_obs) + K_inv @ mu
    f_ref = np.linalg.solve(A, rhs)
    K_ref = np.linalg.inv(A)

    model = GPT_lin_general(H=H, Kf_pri=K, muf_pri=mu)
    model.set_obs(
        g_obs=g_obs,
        obs_noise_profile=np.ones_like(obs_noise_std),
        normalize=False,
        obs_noise_level=float(np.mean(obs_noise_std)),
    )
    f_est = model.solve()

    assert np.allclose(f_est, f_ref, atol=1e-6, rtol=1e-6)
    assert np.allclose(model.Kf_pos, K_ref, atol=1e-8, rtol=1e-6)
    assert np.isfinite(model.mll)


def test_gpt_log_general_smoke_runs_and_returns_positive_posterior_mean():
    rng = np.random.default_rng(1)
    H = np.abs(rng.normal(size=(8, 4))) + 0.05
    K = np.eye(4) * 0.8
    mu_log = np.full(4, -1.5)
    f_true = np.exp(np.array([-1.2, -1.0, -1.8, -1.3]))

    obs_noise_level = 0.05
    g_true = H @ f_true
    g_obs = g_true + obs_noise_level * rng.normal(size=H.shape[0])

    model = GPT_log_general(H=H, Kf_pri=K, muf_pri=mu_log)
    model.set_obs(
        g_obs=g_obs,
        obs_noise_profile=np.ones(H.shape[0]),
        normalize=False,
        obs_noise_level=obs_noise_level,
    )

    f_latent = mu_log.copy()
    for _ in range(5):
        delta_f, loss = model.update(f_latent)
        f_latent = np.clip(f_latent + delta_f, -12.0, 8.0)
        assert np.isfinite(loss)

    model.postprocess(f_latent)

    assert np.all(np.isfinite(model.f_mean))
    assert np.all(np.isfinite(model.sigf_pos))
    assert np.all(model.expf_mean > 0)
    assert np.isfinite(model.mll)


def test_gpt_lin_general_normalize_true_smoke():
    rng = np.random.default_rng(2)
    H = np.abs(rng.normal(size=(7, 3))) + 0.1
    K = np.eye(3)
    mu = np.zeros(3)
    g_obs = np.abs(rng.normal(size=7)) + 0.2

    model = GPT_lin_general(H=H, Kf_pri=K, muf_pri=mu)
    model.set_obs(
        g_obs=g_obs,
        obs_noise_profile=np.ones(7),
        normalize=True,
        obs_noise_level=0.1,
    )
    f_mean = model.solve()

    assert f_mean.shape == (3,)
    assert np.all(np.isfinite(f_mean))
    assert np.all(np.isfinite(model.f_std))
    assert np.isfinite(model.mll)

