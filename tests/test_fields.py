"""Checks of the pressure families, the plug-in estimate and the population study."""

import numpy as np
import pytest

import pfb

PATCH = pfb.Patch(1.0, 1.0, N=32)
FIELDS = pfb.population(PATCH, n_per_family=1, seed=0)
TOL = 1e-6


@pytest.mark.parametrize("field", FIELDS, ids=[f.family for f in FIELDS])
def test_population_fields_are_valid_measures(field):
    nu = field.measure
    assert field.family in pfb.FAMILIES
    assert np.all(nu.w >= 0) and np.isclose(nu.fN, 1.0)
    assert np.all(nu.cop >= 0) and np.all(nu.cop <= [PATCH.Lx, PATCH.Ly])


def test_xydas_kao_limits():
    c, a = (0.5, 0.5), 0.3
    assert np.allclose(pfb.xydas_kao(PATCH, c, a, 2).w, pfb.hertz(PATCH, c, a).w)
    flat = pfb.xydas_kao(PATCH, c, a, 200)
    rho = flat.w / PATCH.quad
    inner = np.linalg.norm(PATCH.points - c, axis=1) < 0.8 * a
    assert rho[inner].max() / rho[inner].min() < 1.01  # k -> inf: uniform disk


def test_partial_contact_is_clipped():
    nu = pfb.hertz(PATCH, (1.0, 0.5), 0.3)  # centred on the right edge: half the disk is off the pad
    assert nu.cop[0] < 1.0 - 0.05


def test_plugin_reproduces_taxel_forces():
    tx = pfb.TaxelGrid(PATCH, 4, 4)
    for field in FIELDS:
        F = tx.downsample(field.measure)
        assert np.allclose(tx.downsample(pfb.plugin_uniform(tx, F)), F)


def test_resolution_study_orderings_and_cache(tmp_path):
    fields, ks, Q = FIELDS[:2], [1, 2], np.array([[0.0, 0.0, 1.0]])
    S = pfb.resolution_study(fields, ks, Q, mf_kwargs=dict(n_c=3, polish=False), cache=tmp_path / "s.npz", verbose=False)
    h = S["truth"][:, None, :]
    for key in ("MT", "MF", "ref"):
        assert np.all(S[f"{key}_lo"] <= h + TOL) and np.all(h <= S[f"{key}_up"] + TOL)
    assert np.all(S["ref_lo"] <= S["MF_lo"] + TOL) and np.all(S["MF_up"] <= S["ref_up"] + TOL)
    assert np.all(S["MF_lo"] <= S["MT_lo"] + TOL)
    again = pfb.resolution_study(fields, ks, Q, mf_kwargs=dict(n_c=3, polish=False), cache=tmp_path / "s.npz", verbose=False)
    assert np.array_equal(again["MT_lo"], S["MT_lo"])
    other = pfb.resolution_study(fields, [1], Q, mf_kwargs=dict(n_c=3, polish=False), cache=tmp_path / "s.npz", verbose=False)
    assert other["MT_lo"].shape[1] == 1  # different parameters: recomputed, not reloaded
