from __future__ import annotations

import numpy as np

from gpoloidal.cis.forward import forward_cis_from_av, forward_cis_from_etv, forward_emit


def test_forward_emit_matches_dense_reference():
    H = np.array([[1.0, 2.0], [0.5, 0.0]])
    e_hat = np.log(np.array([2.0, 3.0]))
    y = forward_emit(H, e_hat)
    ref = H @ np.exp(e_hat)
    assert np.allclose(y, ref)


def test_forward_cis_from_av_zero_velocity_gives_zero_imag():
    H = np.array([[1.0, 2.0], [0.5, 0.1]])
    Dcos = np.array([[0.2, -0.3], [0.0, 0.9]])
    a_hat = np.array([0.1, -0.2])
    v_hat = np.zeros(2)
    IRe, IIm = forward_cis_from_av(H, Dcos, a_hat, v_hat)
    assert np.allclose(IIm, 0.0, atol=1e-12)
    assert np.allclose(IRe, H @ np.exp(a_hat), atol=1e-12)


def test_forward_cis_from_av_zero_dcos_reduces_to_weighted_amplitude():
    H = np.array([[1.0, 2.0], [0.5, 0.1]])
    Dcos = np.zeros_like(H)
    a_hat = np.array([0.3, -0.4])
    v_hat = np.array([2.0, -1.0])
    IRe, IIm = forward_cis_from_av(H, Dcos, a_hat, v_hat)
    assert np.allclose(IIm, 0.0, atol=1e-12)
    assert np.allclose(IRe, H @ np.exp(a_hat), atol=1e-12)


def test_forward_cis_from_etv_consistency():
    H = np.array([[1.0, 0.0], [0.5, 1.0]])
    Dcos = np.array([[0.1, 0.0], [0.2, -0.4]])
    e_hat = np.array([-1.0, -0.5])
    T_hat = np.array([0.2, 0.3])
    v_hat = np.array([1.2, -0.8])
    I0, IRe, IIm = forward_cis_from_etv(H, Dcos, e_hat, T_hat, v_hat)
    assert I0.shape == (2,)
    assert IRe.shape == (2,)
    assert IIm.shape == (2,)
    assert np.all(np.isfinite(I0))
    assert np.all(np.isfinite(IRe))
    assert np.all(np.isfinite(IIm))

