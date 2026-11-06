"""Thin-film optics against closed-form results and an independent field integral."""
import numpy as np
import pytest

from pqcsim.optics.tmm import absorption_by_field_integral, tmm
from pqcsim._compat import trapezoid

LAM = 1550.0


@pytest.mark.parametrize("n2", [1.45, 3.48, 2.0 + 0.5j, 0.55 + 11.5j])
def test_single_interface_fresnel(n2):
    res = tmm(LAM, 1.0, [], [], n2)
    assert res.R == pytest.approx(abs((1 - n2) / (1 + n2)) ** 2, rel=1e-12)
    assert res.T == pytest.approx(1 - res.R, rel=1e-12)   # all non-reflected power enters the substrate


def test_lossless_stack_conserves_energy():
    res = tmm(LAM, 1.0, [2.1, 1.45, 3.48, 1.45], [113.0, 250.0, 90.0, 37.0], 1.45)
    assert res.R + res.T == pytest.approx(1.0, abs=1e-13)
    assert np.allclose(res.A_layers, 0.0, atol=1e-13)


def test_quarter_wave_antireflection():
    ns = 3.48
    nc = np.sqrt(ns)
    assert tmm(LAM, 1.0, [nc], [LAM / (4 * nc)], ns).R == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("pairs", [1, 3, 6])
def test_bragg_mirror_peak_reflectance(pairs):
    nH, nL, ns = 2.3, 1.45, 1.52
    layers = [nH, nL] * pairs
    d = [LAM / (4 * n) for n in layers]
    Y = (nH / nL) ** (2 * pairs) * ns
    assert tmm(LAM, 1.0, layers, d, ns).R == pytest.approx(((1 - Y) / (1 + Y)) ** 2, rel=1e-12)


def test_half_wave_layer_is_absent():
    with_layer = tmm(LAM, 1.0, [2.0], [LAM / 4.0], 3.48)   # d = lambda / (2 n)
    assert with_layer.R == pytest.approx(tmm(LAM, 1.0, [], [], 3.48).R, rel=1e-12)


def test_beer_lambert_in_thick_absorber():
    n = 1.5 + 0.01j
    T1 = tmm(LAM, 1.5, [n], [20000.0], 1.5).T
    T2 = tmm(LAM, 1.5, [n], [30000.0], 1.5).T
    alpha = 4 * np.pi * n.imag / (LAM * 1e-9)
    # Residual interface reflections |r|^2 ~ (k / 2n)^2 ~ 1e-5 perturb the pure exponential slightly.
    assert T2 / T1 == pytest.approx(np.exp(-alpha * 10000e-9), rel=1e-4)


def test_transmission_is_reciprocal():
    layers = [1.45, 4.48 + 1.45j, 2.0, 0.3 + 3j]
    d = [40.0, 25.0, 120.0, 8.0]
    fwd = tmm(LAM, 1.45, layers, d, 1.45)
    bwd = tmm(LAM, 1.45, layers[::-1], d[::-1], 1.45)
    assert fwd.T == pytest.approx(bwd.T, rel=1e-12)
    assert fwd.R != pytest.approx(bwd.R, rel=1e-3)          # reflection is not reciprocal with loss


@pytest.mark.parametrize("seed", range(5))
def test_layer_absorption_matches_field_integral_and_closes(seed):
    rng = np.random.default_rng(seed)
    N = 5
    layers = [complex(rng.uniform(1.3, 4.5), rng.choice([0, rng.uniform(0, 3)])) for _ in range(N)]
    d = list(rng.uniform(5, 200, N))
    res = tmm(LAM, 1.0, layers, d, 1.45)
    assert res.R + res.T + res.A_total == pytest.approx(1.0, abs=1e-12)
    assert np.all(res.A_layers > -1e-12)
    assert np.allclose(res.A_layers, absorption_by_field_integral(LAM, 1.0, layers, d, res), atol=1e-7)


def test_rejects_gain_and_lossy_incidence():
    with pytest.raises(ValueError):
        tmm(LAM, 1.0, [2.0 - 0.1j], [10.0], 1.45)
    with pytest.raises(ValueError):
        tmm(LAM, 1.0 + 0.1j, [], [], 1.45)


@pytest.mark.parametrize("seed", range(4))
def test_absorption_profile_sums_to_layer_absorption(seed):
    from pqcsim.optics.tmm import absorption_profile, field_profile
    rng = np.random.default_rng(10 + seed)
    layers = [1.45, complex(rng.uniform(3, 5), rng.uniform(0.2, 2)), 1.45, 0.55 + 11.5j]
    d = [20.0, float(rng.uniform(10, 60)), 270.0, 100.0]
    res = tmm(LAM, 1.0, layers, d, 1.45)
    prof = absorption_profile(LAM, 1.0, layers, d, res, [2, 7, 3, 5])
    for j, p in enumerate(prof):
        assert p.sum() == pytest.approx(res.A_layers[j], abs=1e-13)
        assert np.all(p >= -1e-15)
    # each slab against direct numerical integration of k0 Im(eps) |E|^2
    k0 = 2 * np.pi / (LAM * 1e-9)
    z, E = field_profile(LAM, layers, d, res, points_per_layer=70001)[1]
    z = z - d[0]                                   # field_profile returns absolute depth
    for s in range(7):                             # 70001 points: slab edges fall on samples 10000 s
        m = slice(10000 * s, 10000 * (s + 1) + 1)
        num = k0 * (layers[1] ** 2).imag * trapezoid(abs(E[m]) ** 2, z[m] * 1e-9)
        assert prof[1][s] == pytest.approx(num, rel=1e-6)
