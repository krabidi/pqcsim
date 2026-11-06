import math

import numpy as np
import pytest

from pqcsim.analysis.calibration import fit, fit_crosstalk_matrix
from pqcsim.analysis.variation import monte_carlo, perturb_network, yield_fraction
from pqcsim.circuits.network import BeamSplitter, LinearNetwork, PhaseShift
from pqcsim.electronics.control import (DAC, ThermoOpticHeater, crosstalk_phases, feedforward_delay_transmission,
                                        heater_powers_for_phases)


def test_heater_phase_and_P_pi():
    h = ThermoOpticHeater(length_um=100.0)
    expected = 2 * math.pi / 1550e-9 * 1.86e-4 * 100e-6 * 1e4
    assert h.phase_per_watt == pytest.approx(expected)
    V = h.voltage_for_phase(math.pi)
    assert h.phase(V) == pytest.approx(math.pi) and V ** 2 / h.R_heater == pytest.approx(h.P_pi)


def test_phase_noise_from_dac_noise_monte_carlo():
    h = ThermoOpticHeater(length_um=200.0)
    V = h.voltage_for_phase(1.2)
    dac = DAC(bits=16, v_fullscale=10.0, noise_rms=1e-3)
    phases = h.phase(dac.output(np.full(200000, V), rng=np.random.default_rng(1)))
    assert phases.std() == pytest.approx(h.phase_noise_std(V, 1e-3), rel=0.02)
    assert np.max(np.abs(DAC(bits=8, v_fullscale=2.0).output(np.linspace(0, 2, 999)) - np.linspace(0, 2, 999))) \
        <= DAC(bits=8, v_fullscale=2.0).lsb / 2 + 1e-15


def test_crosstalk_compensation():
    D = [3.0, 2.5, 3.2]
    K = np.array([[1.0, 0.15, 0.03], [0.15, 1.0, 0.15], [0.03, 0.15, 1.0]])
    target = [0.4, 2.9, 5.5]
    P = heater_powers_for_phases(D, K, target)
    assert np.all(P >= 0)
    assert np.mod(crosstalk_phases(D, K, P), 2 * np.pi) == pytest.approx(target, abs=1e-10)


def test_feedforward_delay_loss():
    # 10 ns latency, n_g = 4, 1 dB/cm -> L = 75 cm -> 75 dB
    eta = feedforward_delay_transmission(10e-9, 4.0, 1.0)
    assert -10 * math.log10(eta) == pytest.approx(299792458 * 10e-9 / 4 * 100, rel=1e-12)


def _mzi(theta1, theta2, phi):
    return LinearNetwork(2, [BeamSplitter(0, 1, theta1), PhaseShift(0, phi), BeamSplitter(0, 1, theta2)])


def test_mzi_extinction_range_closed_form():
    t1, t2 = math.pi / 4 + 0.05, math.pi / 4 - 0.02
    P = [abs(_mzi(t1, t2, phi).matrix()[1, 0]) ** 2 for phi in np.linspace(0, 2 * math.pi, 2001)]
    # cross-port power spans [sin^2(t1 - t2), sin^2(t1 + t2)]
    assert min(P) == pytest.approx(math.sin(t1 - t2) ** 2, abs=1e-6)
    assert max(P) == pytest.approx(math.sin(t1 + t2) ** 2, abs=1e-6)


def test_monte_carlo_coupler_errors_match_analytic_mean():
    """Bar-port leakage of a tuned MZI = cos^2(t1 + t2); with t_i = pi/4 + N(0, s^2):
    E[sin^2(d1 + d2)] = (1 - exp(-4 s^2)) / 2."""
    s = 0.05
    base = _mzi(math.pi / 4, math.pi / 4, 0.0)

    def sample(rng):
        net = perturb_network(base, rng, sigma_theta=s)
        t1, t2 = net.elements[0].theta, net.elements[2].theta
        return abs(_mzi(t1, t2, 0.0).matrix()[0, 0]) ** 2       # bar port at phi = 0

    vals = monte_carlo(sample, 20000, seed=3)
    assert vals.mean() == pytest.approx((1 - math.exp(-4 * s ** 2)) / 2, rel=0.03)
    p, err = yield_fraction(vals, 1e-2, higher_is_better=False)
    assert 0 < p < 1 and err < 0.01


def test_fit_recovers_heater_mzi_parameters():
    rng = np.random.default_rng(7)
    V = np.linspace(0, 4, 120)
    true = np.array([0.9, 0.05, 2.7, 0.4])        # amplitude, offset, V_pi, phase offset

    def model(V, p):
        return p[0] * np.cos(np.pi * V ** 2 / (2 * p[2] ** 2) + p[3] / 2) ** 2 + p[1]

    y = model(V, true) + rng.normal(0, 0.01, V.size)
    res = fit(model, V, y, p0=[1.0, 0.0, 2.5, 0.0], sigma=np.full(V.size, 0.01))
    assert np.all(np.abs(res.params - true) < 4 * res.stderr)
    assert res.chi2_reduced == pytest.approx(1.0, abs=0.3)


def test_fit_crosstalk_matrix():
    rng = np.random.default_rng(2)
    D = np.array([3.0, 2.5, 3.2])
    K = np.array([[1.0, 0.12, 0.02], [0.1, 1.0, 0.14], [0.03, 0.11, 1.0]])
    P = rng.uniform(0, 0.05, (40, 3))
    phi = P @ K.T * D[None, :]
    assert fit_crosstalk_matrix(P, phi, D) == pytest.approx(K, abs=1e-12)
