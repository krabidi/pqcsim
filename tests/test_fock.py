"""Check the Fock-state engine against textbook results and an independent permanent calculation."""
import itertools
import math

import numpy as np
import pytest

import photonics_core as pc


def bs_matrix(n_modes, i, j, theta=math.pi / 4, phi=0.0):
    """Same convention as the C++ code: U[out][in]."""
    u = np.eye(n_modes, dtype=complex)
    u[i, i] = u[j, j] = math.cos(theta)
    u[i, j] = 1j * np.exp(-1j * phi) * math.sin(theta)
    u[j, i] = 1j * np.exp(1j * phi) * math.sin(theta)
    return u


def phase_matrix(n_modes, mode, phi):
    u = np.eye(n_modes, dtype=complex)
    u[mode, mode] = np.exp(1j * phi)
    return u


def permanent(a):
    n = a.shape[0]
    if n == 0:
        return 1.0
    return sum(np.prod([a[k, p[k]] for k in range(n)]) for p in itertools.permutations(range(n)))


def expected_amplitude(u, n_in, n_out):
    """<n_out| U |n_in> = Perm(U_sub) / sqrt(prod n_in! n_out!)."""
    rows = [m for m, n in enumerate(n_out) for _ in range(n)]
    cols = [m for m, n in enumerate(n_in) for _ in range(n)]
    norm = math.sqrt(np.prod([math.factorial(n) for n in (*n_in, *n_out)]))
    return permanent(u[np.ix_(rows, cols)]) / norm


def outputs(n_modes, n_photons):
    return [o for o in itertools.product(range(n_photons + 1), repeat=n_modes) if sum(o) == n_photons]


def test_hong_ou_mandel_dip():
    out = pc.OpticalCircuit(2).beamsplitter(0, 1).run(pc.FockState([1, 1]))

    assert out.probability([1, 1]) == pytest.approx(0.0, abs=1e-12)
    assert out.probability([2, 0]) == pytest.approx(0.5)
    assert out.probability([0, 2]) == pytest.approx(0.5)


@pytest.mark.parametrize("theta", np.linspace(0, math.pi / 2, 7))
def test_coincidence_probability_vs_reflectivity(theta):
    # P(1,1) = (T - R)^2 = cos^2(2 theta)
    out = pc.OpticalCircuit(2).beamsplitter(0, 1, theta).run(pc.FockState([1, 1]))
    assert out.probability([1, 1]) == pytest.approx(math.cos(2 * theta) ** 2, abs=1e-12)


def test_single_photon_splits_evenly():
    out = pc.OpticalCircuit(2).beamsplitter(0, 1).run(pc.FockState([1, 0]))

    assert out.probabilities() == pytest.approx({(1, 0): 0.5, (0, 1): 0.5})
    assert out.mean_photon_number(0) == pytest.approx(0.5)


@pytest.mark.parametrize("phi", np.linspace(0, 2 * math.pi, 9))
def test_mach_zehnder_fringe(phi):
    circuit = pc.OpticalCircuit(2).beamsplitter(0, 1).phase_shift(0, phi).beamsplitter(0, 1)
    out = circuit.run(pc.FockState([1, 0]))

    assert out.probability([0, 1]) == pytest.approx(math.cos(phi / 2) ** 2, abs=1e-12)
    assert out.probability([1, 0]) == pytest.approx(math.sin(phi / 2) ** 2, abs=1e-12)


@pytest.mark.parametrize("n_in", [(1, 1, 1), (2, 1, 0), (3, 0, 1), (1, 2, 2)])
def test_matches_permanent_for_random_circuit(n_in):
    rng = np.random.default_rng(42)
    n_modes = len(n_in)
    circuit = pc.OpticalCircuit(n_modes)
    u = np.eye(n_modes, dtype=complex)
    for _ in range(6):
        i, j = rng.choice(n_modes, size=2, replace=False)
        theta, phi = rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)
        circuit.beamsplitter(int(i), int(j), theta, phi)
        u = bs_matrix(n_modes, i, j, theta, phi) @ u
        mode, alpha = int(rng.integers(n_modes)), rng.uniform(0, 2 * math.pi)
        circuit.phase_shift(mode, alpha)
        u = phase_matrix(n_modes, mode, alpha) @ u

    out = circuit.run(pc.FockState(list(n_in)))

    assert out.norm() == pytest.approx(1.0, abs=1e-12)
    for n_out in outputs(n_modes, sum(n_in)):
        assert out.amplitude(list(n_out)) == pytest.approx(expected_amplitude(u, n_in, n_out), abs=1e-10)


def test_sampling_matches_probabilities():
    out = pc.OpticalCircuit(2).beamsplitter(0, 1, math.pi / 6).run(pc.FockState([1, 0]))

    samples = out.sample(20000, seed=1)

    frac = sum(s == (1, 0) for s in samples) / len(samples)
    assert frac == pytest.approx(out.probability([1, 0]), abs=0.02)
    assert out.sample(10, seed=7) == out.sample(10, seed=7)


def test_invalid_inputs_raise():
    with pytest.raises(IndexError):
        pc.OpticalCircuit(2).beamsplitter(0, 2)
    with pytest.raises(ValueError):
        pc.OpticalCircuit(2).beamsplitter(1, 1)
    with pytest.raises(ValueError):
        pc.OpticalCircuit(3).run(pc.FockState([1, 0]))
    with pytest.raises(ValueError):
        pc.FockState([-1, 0])
