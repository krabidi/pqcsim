"""Partial distinguishability: limits, normalisation, HOM dip, and an independent internal-mode route."""
import math

import numpy as np
import pytest
from scipy.stats import unitary_group

from pqcsim._core import core
from pqcsim.circuits.decompose import unitary_elements
from pqcsim.circuits.network import BeamSplitter, LinearNetwork, Loss, PhaseShift
from pqcsim.quantum.distinguishable import gaussian_overlap, simulate_partial_distinguishability as spd
from pqcsim.quantum.lossy import simulate_fock
from pqcsim._compat import trapezoid


def random_gram(N, rng, K=None):
    K = K or N
    V = rng.normal(size=(K, N)) + 1j * rng.normal(size=(K, N))
    V /= np.linalg.norm(V, axis=0)
    return V.conj().T @ V, V


def unitary_net(N, seed):
    return LinearNetwork(N, unitary_elements(unitary_group.rvs(N, random_state=seed)))


@pytest.mark.parametrize("s", [0.0, 0.3, 0.7, 1.0])
def test_hom_dip_formula(s):
    net = LinearNetwork(2, [BeamSplitter(0, 1)])
    p = spd(net, [0, 1], np.array([[1, s], [s, 1]]))
    assert p.get((1, 1), 0.0) == pytest.approx((1 - s ** 2) / 2, abs=1e-12)
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-12)


def test_indistinguishable_limit_matches_fock_engine():
    net = unitary_net(3, 4)
    for modes in ([0, 1, 2], [0, 0, 1], [2, 2, 2]):
        occ = [modes.count(m) for m in range(3)]
        p, q = spd(net, modes, np.ones((3, 3))), simulate_fock(net, occ)
        for key in set(p) | set(q):
            assert p.get(key, 0) == pytest.approx(q.get(key, 0), abs=1e-10)


def test_distinguishable_limit_is_classical_permanent():
    from pqcsim.quantum.reference import permanent
    net = unitary_net(3, 7)
    U = net.matrix()
    p = spd(net, [0, 1, 2], np.eye(3))
    for out, val in p.items():
        r = [m for m, n in enumerate(out) for _ in range(n)]
        assert val == pytest.approx(permanent(np.abs(U[np.ix_(r, [0, 1, 2])]) ** 2).real
                                    / np.prod([math.factorial(n) for n in out]), abs=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_probabilities_normalised_for_random_gram(seed):
    rng = np.random.default_rng(seed)
    S, _ = random_gram(3, rng)
    p = spd(unitary_net(3, seed), [0, 1, 2], S)
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-12)
    assert min(p.values()) >= -1e-12


def _internal_mode_reference(U, input_modes, V):
    """Independent route: spatial mode m, internal level q -> engine mode m*K + q.
    Each photon starts in internal level 0 and a preparation unitary on its spatial mode's internal
    levels maps e_0 -> v_j; the network acts identically on every internal level."""
    n, K = U.shape[0], V.shape[0]
    circuit = core.OpticalCircuit(n * K)
    occ = [0] * (n * K)
    for j, m in enumerate(input_modes):
        occ[m * K] += 1
        Q, _ = np.linalg.qr(np.column_stack([V[:, j], np.eye(K)[:, 1:]]))
        Q[:, 0] *= np.vdot(Q[:, 0], V[:, j]) / abs(np.vdot(Q[:, 0], V[:, j]))   # first column exactly v_j
        assert Q[:, 0] == pytest.approx(V[:, j], abs=1e-12)
        for el in unitary_elements(Q, [m * K + q for q in range(K)]):
            _apply(circuit, el)
    for q in range(K):
        for el in unitary_elements(U, [m * K + q for m in range(n)]):
            _apply(circuit, el)
    out = circuit.run(core.FockState(occ)).probabilities()
    probs = {}
    for o, p in out.items():
        key = tuple(sum(o[m * K:(m + 1) * K]) for m in range(n))
        probs[key] = probs.get(key, 0.0) + p
    return probs


def _apply(circuit, el):
    if isinstance(el, BeamSplitter):
        circuit.beamsplitter(el.i, el.j, el.theta, el.phi)
    else:
        circuit.phase_shift(el.mode, el.phi)


@pytest.mark.parametrize("seed", range(3))
def test_matches_internal_mode_simulation(seed):
    rng = np.random.default_rng(100 + seed)
    S, V = random_gram(3, rng)
    net = unitary_net(3, 20 + seed)
    p = spd(net, [0, 1, 2], S)
    q = _internal_mode_reference(net.matrix(), [0, 1, 2], V)
    for key in set(p) | set(q):
        assert p.get(key, 0) == pytest.approx(q.get(key, 0), abs=1e-10)


def test_loss_with_distinguishability():
    S = np.array([[1, 0.6], [0.6, 1]])
    net = LinearNetwork(2, [Loss(0, 0.7), Loss(1, 0.4), BeamSplitter(0, 1)])
    p = spd(net, [0, 1], S)
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-12)
    # both photons survive with prob 0.28, then the HOM formula applies
    assert p[(1, 1)] == pytest.approx(0.7 * 0.4 * (1 - 0.36) / 2, abs=1e-12)
    assert p.get((0, 0), 0.0) == pytest.approx(0.3 * 0.6, abs=1e-12)


def test_gaussian_overlap_against_numerical_integral():
    sig, tau, det, w0 = 2.0, 0.37, 1.1, 5.0
    w = np.linspace(-40, 50, 400001)
    phi1 = np.exp(-(w - (w0 - det / 2)) ** 2 / (4 * sig ** 2))
    phi2 = np.exp(-(w - (w0 + det / 2)) ** 2 / (4 * sig ** 2)) * np.exp(1j * w * tau)
    norm = np.sqrt(trapezoid(abs(phi1) ** 2, w) * trapezoid(abs(phi2) ** 2, w))
    num = trapezoid(np.conj(phi1) * phi2, w) / norm
    assert gaussian_overlap(tau, sig, det, w0) == pytest.approx(num, abs=1e-9)


def test_rejects_invalid_gram():
    net = LinearNetwork(2, [BeamSplitter(0, 1)])
    with pytest.raises(ValueError):
        spd(net, [0, 1], np.array([[1, 1.2], [1.2, 1]]))
