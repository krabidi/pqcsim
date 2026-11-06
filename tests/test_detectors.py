import itertools
import math

import numpy as np
import pytest

from pqcsim.circuits.network import BeamSplitter, LinearNetwork, Loss
from pqcsim.quantum.detectors import (Detector, accidental_rate, coincidence_window_fraction, detect,
                                      gaussian_clicks)
from pqcsim.quantum.gaussian import GaussianState, squeezing_for_mean_pairs
from pqcsim.quantum.lossy import simulate_fock


def test_efficiency_equals_loss_before_ideal_detector():
    net = LinearNetwork(2, [BeamSplitter(0, 1, 0.4)])
    eff = (0.8, 0.55)
    a = detect(simulate_fock(net, [2, 1]), [Detector(eff[0]), Detector(eff[1])])
    b = simulate_fock(LinearNetwork(2, [BeamSplitter(0, 1, 0.4), Loss(0, eff[0]), Loss(1, eff[1])]), [2, 1])
    for k in set(a) | set(b):
        assert a.get(k, 0) == pytest.approx(b.get(k, 0), abs=1e-12)


def test_dark_counts_on_vacuum_and_normalisation():
    mu = 0.03
    d = detect({(0,): 1.0}, [Detector(dark_mean=mu, threshold=True, max_dark=12)])
    assert d[(1,)] == pytest.approx(1 - math.exp(-mu), rel=1e-12)
    p = detect(simulate_fock(LinearNetwork(2, [BeamSplitter(0, 1)]), [1, 1]),
               [Detector(0.9, 0.01, max_dark=10)] * 2)
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-12)


def test_threshold_hides_bunching():
    p = detect(simulate_fock(LinearNetwork(2, [BeamSplitter(0, 1)]), [1, 1]), [Detector(threshold=True)] * 2)
    assert p == pytest.approx({(1, 0): 0.5, (0, 1): 0.5})


def test_gaussian_clicks_match_fock_route_for_thermal_light():
    """Reduced TMSV mode is thermal: P(no click) = e^{-mu} / (1 + eta nbar)."""
    nbar, eta, mu = 0.8, 0.6, 0.02
    s = GaussianState(2).two_mode_squeeze(0, 1, squeezing_for_mean_pairs(nbar))
    p_none = gaussian_clicks(s, [0, 0], [eta, 1.0], [mu, 0.0]) + gaussian_clicks(s, [0, 1], [eta, 1.0], [mu, 0.0])
    assert p_none == pytest.approx(math.exp(-mu) / (1 + eta * nbar), rel=1e-12)
    total = sum(gaussian_clicks(s, c, [eta, 0.7], [mu, 0.05]) for c in itertools.product([0, 1], repeat=2))
    assert total == pytest.approx(1.0, abs=1e-12)


def test_gaussian_clicks_agree_with_fock_distribution_route():
    """Same physics two ways: PNR distribution (hafnians) -> Detector model, vs. Gaussian click formula."""
    s = GaussianState(2).two_mode_squeeze(0, 1, 0.3)
    dist = {(a, b): s.fock_probability([a, b]) for a in range(9) for b in range(9) if a == b}
    det = [Detector(0.7, 0.01, threshold=True, max_dark=8), Detector(0.5, 0.03, threshold=True, max_dark=8)]
    viaf = detect(dist, det)
    for c in itertools.product([0, 1], repeat=2):
        assert viaf.get(c, 0) == pytest.approx(gaussian_clicks(s, c, [0.7, 0.5], [0.01, 0.03]), abs=1e-8)


def test_window_fraction_and_accidentals_monte_carlo():
    rng = np.random.default_rng(0)
    j1, j2, w = 40e-12, 30e-12, 80e-12
    dt = rng.normal(0, j1, 400000) - rng.normal(0, j2, 400000)
    assert np.mean(np.abs(dt) <= w / 2) == pytest.approx(coincidence_window_fraction(w, j1, j2), abs=3e-3)
    R1, R2, T = 2e5, 3e5, 0.5
    t1 = np.sort(rng.uniform(0, T, int(R1 * T)))
    t2 = np.sort(rng.uniform(0, T, int(R2 * T)))
    w = 1e-7
    lo, hi = np.searchsorted(t2, t1 - w / 2), np.searchsorted(t2, t1 + w / 2)
    assert (hi - lo).sum() / T == pytest.approx(accidental_rate(R1, R2, w), rel=0.05)
