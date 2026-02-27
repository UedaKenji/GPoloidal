from __future__ import annotations

import numpy as np

from gpoloidal.rt1 import phantom


def test_new_cis_phantom_names_are_registered():
    names = set(phantom.list_phantom_names())
    assert {"ring_emissivity", "ring_velocity", "simple_temperature"} <= names


def test_get_cis_phantom_bundle_returns_three_component_functions():
    bundle = phantom.get_cis_phantom_bundle()
    r = np.array([0.6, 0.8, 0.9], dtype=float)
    z = np.array([0.0, 0.1, -0.2], dtype=float)
    e = np.asarray(bundle.emissivity_fn(r, z), dtype=float)
    T = np.asarray(bundle.temperature_fn(r, z), dtype=float)
    v = np.asarray(bundle.velocity_fn(r, z), dtype=float)
    assert e.shape == r.shape
    assert T.shape == r.shape
    assert v.shape == r.shape
    assert np.all(np.isfinite(e))
    assert np.all(np.isfinite(T))
    assert np.all(np.isfinite(v))


def test_get_cis_phantom_bundle_accepts_component_overrides():
    bundle = phantom.get_cis_phantom_bundle(
        emissivity_name="hollow",
        emissivity_params={"n0": 2.0},
        temperature_params={"a": 8.0},
        velocity_params={"ring_scale": 10.0},
    )
    r = np.array([0.7], dtype=float)
    z = np.array([0.0], dtype=float)
    e = float(np.asarray(bundle.emissivity_fn(r, z))[0])
    T = float(np.asarray(bundle.temperature_fn(r, z))[0])
    v = float(np.asarray(bundle.velocity_fn(r, z))[0])
    assert np.isfinite(e)
    assert np.isfinite(T)
    assert np.isfinite(v)

