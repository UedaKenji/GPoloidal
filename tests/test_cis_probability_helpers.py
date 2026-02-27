from __future__ import annotations

import numpy as np

from gpoloidal.tomography import build_a_prior_from_emit_posterior, recover_tv_posterior_from_emit_and_av


def test_build_a_prior_from_emit_posterior_matches_sum_rule():
    mu_e = np.array([1.0, 2.0])
    mu_T = np.array([0.3, -0.1])
    K_e = np.array([[0.4, 0.1], [0.1, 0.3]])
    K_T = np.array([[0.2, 0.0], [0.0, 0.5]])
    mu_a, K_a = build_a_prior_from_emit_posterior(mu_e=mu_e, K_e=K_e, mu_T_pri=mu_T, K_T_pri=K_T)
    assert np.allclose(mu_a, mu_e - mu_T)
    assert np.allclose(K_a, K_e + K_T)


def test_recover_tv_posterior_shapes_and_symmetry():
    n = 3
    mu_e = np.zeros(n)
    mu_T = np.zeros(n)
    mu_a = np.zeros(n)
    mu_v = np.zeros(n)
    K_e = np.eye(n) * 0.3
    K_T = np.eye(n) * 0.4
    K_aa = np.eye(n) * 0.2
    K_av = np.eye(n) * 0.05
    K_vv = np.eye(n) * 0.6

    out = recover_tv_posterior_from_emit_and_av(
        mu_e=mu_e,
        K_e=K_e,
        mu_T_pri=mu_T,
        K_T_pri=K_T,
        mu_a=mu_a,
        mu_v=mu_v,
        K_aa=K_aa,
        K_av=K_av,
        K_vv=K_vv,
    )
    assert out.mu_T.shape == (n,)
    assert out.mu_v.shape == (n,)
    assert out.K_TT.shape == (n, n)
    assert out.K_Tv.shape == (n, n)
    assert out.K_vT.shape == (n, n)
    assert out.K_vv.shape == (n, n)
    assert np.allclose(out.K_TT, out.K_TT.T, atol=1e-10)
    assert np.allclose(out.K_vv, out.K_vv.T, atol=1e-10)
    assert np.allclose(out.K_Tv, out.K_vT.T, atol=1e-10)
    assert np.all(np.isfinite(out.sig_T))
    assert np.all(np.isfinite(out.sig_v))

