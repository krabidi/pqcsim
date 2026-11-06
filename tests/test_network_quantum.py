"""Classical lossy networks and exact lossy quantum simulation, cross-checked against each other."""
import math

import numpy as np
import pytest

from pqcsim.circuits.network import (BeamSplitter, LinearNetwork, Loss, PhaseShift, pcm_segment,
                                     single_mode_elements, waveguide)
from pqcsim.quantum.lossy import simulate_fock
from pqcsim.quantum.reference import lossy_probabilities


def random_network(n, depth, rng, lossy=True):
    net = LinearNetwork(n)
    for _ in range(depth):
        i, j = rng.choice(n, 2, replace=False)
        net.add(BeamSplitter(int(i), int(j), rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)))
        m = int(rng.integers(n))
        net.add(PhaseShift(m, rng.uniform(0, 2 * math.pi)))
        if lossy:
            net.add(Loss(m, rng.uniform(0.3, 1.0)))
    return net


def test_lossless_network_is_unitary_and_lossy_is_contraction():
    rng = np.random.default_rng(0)
    assert random_network(4, 8, rng, lossy=False).is_lossless()
    T = random_network(4, 8, rng).matrix()
    assert np.linalg.svd(T, compute_uv=False).max() <= 1 + 1e-12


def test_waveguide_loss_conversion():
    (_, loss), = [waveguide(0, 10000.0, 2.4, 1550.0, loss_dB_per_cm=3.0)[i:i + 2] for i in (0,)]
    assert loss.eta == pytest.approx(10 ** -0.3)


def test_mach_zehnder_classical_fringe():
    for phi in np.linspace(0, 2 * math.pi, 7):
        net = LinearNetwork(2, [BeamSplitter(0, 1), PhaseShift(0, phi), BeamSplitter(0, 1)])
        P = np.abs(net.propagate_fields([1, 0])) ** 2
        assert P == pytest.approx([math.sin(phi / 2) ** 2, math.cos(phi / 2) ** 2], abs=1e-12)


def test_incoherent_powers_add():
    net = LinearNetwork(2, [BeamSplitter(0, 1, 0.3), Loss(1, 0.5)])
    assert net.propagate_powers([1.0, 2.0]) == pytest.approx(
        np.abs(net.propagate_fields([1, 0])) ** 2 + 2 * np.abs(net.propagate_fields([0, 1])) ** 2)


@pytest.mark.parametrize("n,eta", [(1, 0.3), (3, 0.6), (5, 0.9)])
def test_fock_loss_is_binomial(n, eta):
    p = simulate_fock(LinearNetwork(1, [Loss(0, eta)]), [n])
    for k in range(n + 1):
        assert p.get((k,), 0.0) == pytest.approx(math.comb(n, k) * eta ** k * (1 - eta) ** (n - k), abs=1e-12)


@pytest.mark.parametrize("eta0,eta1", [(1.0, 0.5), (0.2, 0.9), (0.7, 0.7)])
def test_hom_survives_loss_before_beamsplitter(eta0, eta1):
    p = simulate_fock(LinearNetwork(2, [Loss(0, eta0), Loss(1, eta1), BeamSplitter(0, 1)]), [1, 1])
    assert p.get((1, 1), 0.0) == pytest.approx(0.0, abs=1e-12)
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-12)
    assert p.get((0, 0), 0.0) == pytest.approx((1 - eta0) * (1 - eta1), abs=1e-12)


@pytest.mark.parametrize("seed,n_in", [(1, (1, 1, 0)), (2, (2, 0, 1)), (3, (1, 1, 1))])
def test_matches_dilation_permanent_reference(seed, n_in):
    net = random_network(3, 5, np.random.default_rng(seed))
    fast = simulate_fock(net, n_in)
    ref = lossy_probabilities(net.matrix(), n_in)
    for k in set(fast) | set(ref):
        assert fast.get(k, 0.0) == pytest.approx(ref.get(k, 0.0), abs=1e-10)


@pytest.mark.parametrize("seed", range(3))
def test_quantum_mean_photon_number_equals_classical_power(seed):
    """For Fock inputs <n_i> = sum_j |T_ij|^2 n_j: quantum and classical layers must agree."""
    net = random_network(3, 6, np.random.default_rng(seed))
    n_in = [2, 1, 0]
    p = simulate_fock(net, n_in)
    mean = np.array([sum(q * occ[i] for occ, q in p.items()) for i in range(3)])
    assert mean == pytest.approx(net.propagate_powers(n_in), abs=1e-10)


def test_pcm_segment_transmission(gete):
    L, lam, gamma = 2.0, 1550.0, 0.1
    for x in (0.0, 0.5, 1.0):
        net = LinearNetwork(1, pcm_segment(0, gete, x, L, lam, gamma, n_ref=1.45))
        dn = gamma * (gete.nk(x, lam) - 1.45)
        expected = np.exp(1j * 2 * math.pi / (lam * 1e-9) * dn * L * 1e-6)
        assert net.matrix()[0, 0] == pytest.approx(expected, abs=1e-12)
    with pytest.raises(ValueError):
        single_mode_elements(0, 1.01)
