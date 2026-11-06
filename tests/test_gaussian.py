"""Gaussian backend against closed-form photon statistics of squeezed and thermal light."""
import itertools
import math

import numpy as np
import pytest

from pqcsim.circuits.network import BeamSplitter, LinearNetwork, Loss
from pqcsim.quantum.gaussian import GaussianState, hafnian, squeezing_for_mean_pairs
from pqcsim.quantum.reference import permanent


def test_hafnian_known_values():
    assert hafnian(np.ones((4, 4))) == pytest.approx(3)          # (2m-1)!! perfect matchings
    assert hafnian(np.ones((6, 6))) == pytest.approx(15)
    B = np.random.default_rng(0).normal(size=(3, 3))
    # haf([[0, B], [B^T, 0]]) = perm(B)
    A = np.block([[np.zeros((3, 3)), B], [B.T, np.zeros((3, 3))]])
    assert hafnian(A) == pytest.approx(permanent(B))


@pytest.mark.parametrize("r", [0.1, 0.5, 1.0])
def test_two_mode_squeezed_vacuum_is_thermal_pairs(r):
    s = GaussianState(2).two_mode_squeeze(0, 1, r)
    lam = math.tanh(r)
    for n in range(5):
        assert s.fock_probability([n, n]) == pytest.approx((1 - lam ** 2) * lam ** (2 * n), abs=1e-12)
    assert s.fock_probability([1, 0]) == pytest.approx(0.0, abs=1e-14)
    assert s.mean_photon_number(0) == pytest.approx(math.sinh(r) ** 2, rel=1e-12)


@pytest.mark.parametrize("r", [0.3, 0.9])
def test_single_mode_squeezed_vacuum(r):
    s = GaussianState(1).squeeze(0, r)
    for n in range(5):
        exact = math.factorial(2 * n) / (2 ** n * math.factorial(n)) ** 2 * math.tanh(r) ** (2 * n) / math.cosh(r)
        assert s.fock_probability([2 * n]) == pytest.approx(exact, abs=1e-12)
        assert s.fock_probability([2 * n + 1]) == pytest.approx(0.0, abs=1e-14)
    assert s.cov[0, 0] == pytest.approx(0.5 * math.exp(-2 * r))        # x is squeezed for phi = 0


def test_symplectic_check_rejects_non_symplectic():
    with pytest.raises(ValueError):
        GaussianState(1).apply_symplectic(np.diag([2.0, 2.0]))


@pytest.mark.parametrize("nbar", [0.2, 1.5])
def test_reduced_mode_is_thermal_mixed_state(nbar):
    s = GaussianState(2).two_mode_squeeze(0, 1, squeezing_for_mean_pairs(nbar))
    single = GaussianState(1)
    single.cov, single.mean = s.reduced([0])
    for n in range(5):
        assert single.fock_probability([n]) == pytest.approx(nbar ** n / (1 + nbar) ** (n + 1), abs=1e-12)
    assert single.click_probability([1]) == pytest.approx(nbar / (1 + nbar), abs=1e-12)


def test_tmsv_on_balanced_splitter_gives_two_single_mode_squeezers():
    """50:50 mixing of a TMSV gives a product of single-mode squeezed vacua with r and -r."""
    r = 0.6
    s = GaussianState(2).two_mode_squeeze(0, 1, r, phi=math.pi / 2)
    s.apply_network(LinearNetwork(2, [BeamSplitter(0, 1)]))
    single = lambda k: (math.factorial(k) / (2 ** (k // 2) * math.factorial(k // 2)) ** 2
                        * math.tanh(r) ** k / math.cosh(r)) if k % 2 == 0 else 0.0
    for a, b in itertools.product(range(5), repeat=2):
        assert s.fock_probability([a, b]) == pytest.approx(single(a) * single(b), abs=1e-12)


def test_loss_scales_mean_photon_number_and_keeps_normalisation():
    r, eta = 0.4, 0.35
    lam2 = math.tanh(r) ** 2
    s = GaussianState(2).two_mode_squeeze(0, 1, r).apply_network(LinearNetwork(2, [Loss(0, eta)]))
    assert s.mean_photon_number(0) == pytest.approx(eta * math.sinh(r) ** 2, rel=1e-12)
    # loss only on the signal: idler count = pair number, so outcomes with b <= 4 carry 1 - lam^10 exactly
    total = sum(s.fock_probability([a, b]) for b in range(5) for a in range(b + 1))
    assert total == pytest.approx(1 - lam2 ** 5, abs=1e-12)
    # binomial thinning of the signal given b pairs
    for b in range(4):
        for a in range(b + 1):
            exact = (1 - lam2) * lam2 ** b * math.comb(b, a) * eta ** a * (1 - eta) ** (b - a)
            assert s.fock_probability([a, b]) == pytest.approx(exact, abs=1e-12)
    assert sum(s.click_probability(c) for c in itertools.product([0, 1], repeat=2)) == pytest.approx(1.0, abs=1e-12)


def test_heralding():
    r = 0.4
    lam2 = math.tanh(r) ** 2
    s = GaussianState(2).two_mode_squeeze(0, 1, r)
    # PNR herald n = 1 on the idler leaves exactly one signal photon
    p1 = s.fock_probability([1, 1])
    assert sum(s.fock_probability([n, 1]) for n in range(8)) == pytest.approx(p1, abs=1e-12)
    # threshold herald: P(signal = n | click) = (1 - lam^2) lam^(2n) / lam^2 for n >= 1
    p_click = s.click_probability([0, 1]) + s.click_probability([1, 1])
    assert p_click == pytest.approx(lam2, abs=1e-12)
    assert s.click_probability([1, 1]) / p_click == pytest.approx(1.0, abs=1e-12)


def test_coherent_state_click_statistics():
    alpha = 0.8 + 0.3j
    s = GaussianState(1).displace(0, alpha)
    assert s.mean_photon_number(0) == pytest.approx(abs(alpha) ** 2)
    assert s.click_probability([0]) == pytest.approx(math.exp(-abs(alpha) ** 2))


def test_cpp_kernels_match_python_references():
    from pqcsim.quantum import kernels
    from pqcsim.quantum.reference import permanent as py_perm
    assert kernels._HAVE_CORE
    rng = np.random.default_rng(5)
    for n in (1, 2, 5, 9):
        A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        assert kernels.permanent(A) == pytest.approx(py_perm(A), rel=1e-11)
    for n in (2, 4, 8, 12):
        B = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        B = B + B.T
        assert kernels.hafnian(B) == pytest.approx(hafnian(B), rel=1e-11)
    assert kernels.hafnian(np.ones((3, 3))) == 0          # odd size
