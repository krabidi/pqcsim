"""Decompositions and physical components against closed forms and reconstruction."""
import math

import numpy as np
import pytest
from scipy.stats import unitary_group

from pqcsim.circuits.components import (coupler_length_for_split, directional_coupler, load_touchstone,
                                        matrix_elements, ring_add_drop, ring_allpass, transfer_from_sparams)
from pqcsim.circuits.decompose import passive_elements, rebuild, two_mode_elements, unitary_elements
from pqcsim.circuits.network import BeamSplitter, LinearNetwork
from pqcsim.quantum.lossy import simulate_fock
from pqcsim.quantum.reference import lossy_probabilities


@pytest.mark.parametrize("seed", range(6))
def test_two_mode_decomposition(seed):
    U = unitary_group.rvs(2, random_state=seed)
    assert rebuild(two_mode_elements(0, 1, U), 2) == pytest.approx(U, abs=1e-12)


@pytest.mark.parametrize("U", [np.eye(2), np.array([[0, 1j], [1j, 0]]), np.diag([1j, -1]),
                               np.array([[0, np.exp(0.3j)], [np.exp(1.1j), 0]])])
def test_two_mode_edge_cases(U):
    assert rebuild(two_mode_elements(0, 1, U), 2) == pytest.approx(U, abs=1e-12)


@pytest.mark.parametrize("N", [2, 3, 4, 6, 9])
def test_unitary_decomposition_is_exact_with_minimal_couplers(N):
    U = unitary_group.rvs(N, random_state=N)
    els = unitary_elements(U)
    assert rebuild(els, N) == pytest.approx(U, abs=1e-11)
    bs = [e for e in els if isinstance(e, BeamSplitter)]
    assert len(bs) == N * (N - 1) // 2
    assert all(abs(e.i - e.j) == 1 for e in bs)          # nearest-neighbour couplers only


def test_passive_decomposition_and_rejects_gain():
    T = unitary_group.rvs(4, random_state=1) @ np.diag([1, .8, .3, .05]) @ unitary_group.rvs(4, random_state=2)
    assert rebuild(passive_elements(T), 4) == pytest.approx(T, abs=1e-12)
    with pytest.raises(ValueError):
        passive_elements(1.01 * np.eye(2))


def test_quantum_through_decomposed_passive_matrix():
    T = unitary_group.rvs(3, random_state=5) @ np.diag([0.9, 0.6, 0.95]) @ unitary_group.rvs(3, random_state=6)
    net = LinearNetwork(3, matrix_elements(T, [0, 1, 2]))
    p, q = simulate_fock(net, [1, 1, 0]), lossy_probabilities(T, [1, 1, 0])
    for k in set(p) | set(q):
        assert p.get(k, 0) == pytest.approx(q.get(k, 0), abs=1e-10)


def test_directional_coupler_split_and_dispersion():
    L = coupler_length_for_split(0.5, 1550.0, 0.02)
    bs = directional_coupler(0, 1, L, 1550.0, 0.02)
    assert math.sin(bs.theta) ** 2 == pytest.approx(0.5, abs=1e-12)
    # full transfer at twice the 50:50 length; dispersion detunes it away from lam0
    assert math.sin(directional_coupler(0, 1, 2 * L, 1550.0, 0.02).theta) ** 2 == pytest.approx(1.0, abs=1e-12)
    off = directional_coupler(0, 1, L, 1500.0, 0.02, ddn_dlam_per_nm=1e-5)
    assert math.sin(off.theta) ** 2 != pytest.approx(0.5, abs=1e-3)


RING = dict(length_um=100.0, n_eff=2.4, n_g=4.2, lam0_nm=1550.0)


def test_lossless_allpass_ring_is_pure_phase():
    for lam in np.linspace(1549, 1551, 41):
        assert abs(ring_allpass(lam, theta=0.3, **RING)) == pytest.approx(1.0, abs=1e-12)


def test_critical_coupling_and_closed_form_transmission():
    loss = 20.0                                                  # dB/cm
    a = 10 ** (-loss * 100e-4 / 20)
    theta_c = math.acos(a)                                       # t = a: critical coupling
    for lam in np.linspace(1549.5, 1550.5, 11):
        t = ring_allpass(lam, theta=0.4, loss_dB_per_cm=loss, **RING)
        phi = 2 * math.pi * (2.4 - (4.2 - 2.4) * (lam - 1550) / 1550) * 100e3 / lam
        tc = math.cos(0.4)
        exact = (tc ** 2 + a ** 2 - 2 * a * tc * math.cos(phi)) / (1 + (a * tc) ** 2 - 2 * a * tc * math.cos(phi))
        assert abs(t) ** 2 == pytest.approx(exact, rel=1e-12)
    # at resonance (phi = 2 pi m) critical coupling extinguishes the through port
    lam_res = _exact_resonance(round(2.4 * 100e3 / 1550))
    assert abs(ring_allpass(lam_res, theta=theta_c, loss_dB_per_cm=loss, **RING)) ** 2 < 1e-24


def _exact_resonance(m, n0=2.4, ng=4.2, L_nm=100e3, lam0=1550.0):
    """n(lam) = ng - (ng - n0) lam / lam0 and n(lam) L = m lam  =>  lam_m = ng L / (m + (ng - n0) L / lam0)."""
    return ng * L_nm / (m + (ng - n0) * L_nm / lam0)


def test_free_spectral_range_uses_group_index():
    m = round(2.4 * 100e3 / 1550)
    lm, lm1 = _exact_resonance(m), _exact_resonance(m + 1)
    # resonances really are transmission zeros of the phase condition
    for lam in (lm, lm1):
        phi = 2 * math.pi * (4.2 - 1.8 * lam / 1550) * 100e3 / lam
        assert math.cos(phi) == pytest.approx(1.0, abs=1e-12)
    assert lm - lm1 == pytest.approx(lm * lm1 / (4.2 * 100e3), rel=1e-12)   # exact: FSR = lam_m lam_m+1 / (n_g L)


def test_add_drop_energy_and_passivity():
    for lam in np.linspace(1549, 1551, 21):
        T = ring_add_drop(lam, theta1=0.3, theta2=0.3, **RING)
        assert np.allclose(T.conj().T @ T, np.eye(2), atol=1e-12)     # lossless -> unitary
        Tl = ring_add_drop(lam, theta1=0.3, theta2=0.5, loss_dB_per_cm=10, **RING)
        assert np.linalg.svd(Tl, compute_uv=False).max() <= 1 + 1e-12
        assert Tl[0, 1] == pytest.approx(Tl[1, 0])                   # reciprocity


@pytest.mark.parametrize("fmt", ["RI", "MA", "DB"])
@pytest.mark.parametrize("N", [2, 3])
def test_touchstone_roundtrip(tmp_path, fmt, N):
    rng = np.random.default_rng(N)
    f = np.array([193.0, 194.0])
    S = 0.5 * (rng.normal(size=(2, N, N)) + 1j * rng.normal(size=(2, N, N)))
    lines = [f"! test\n# THZ S {fmt} R 50"]
    for k in range(2):
        order = S[k].T.ravel() if N == 2 else S[k].ravel()
        vals = []
        for z in order:
            if fmt == "RI":
                vals += [z.real, z.imag]
            elif fmt == "MA":
                vals += [abs(z), np.degrees(np.angle(z))]
            else:
                vals += [20 * np.log10(abs(z)), np.degrees(np.angle(z))]
        lines.append(" ".join([repr(float(f[k]))] + [repr(float(v)) for v in vals]))
    path = tmp_path / f"dev.s{N}p"
    path.write_text("\n".join(lines))
    freq, S2 = load_touchstone(str(path))
    assert freq == pytest.approx(f * 1e12)
    assert S2 == pytest.approx(S, abs=1e-12)
    T = transfer_from_sparams(S2[0], [0], [N - 1])
    assert T[0, 0] == pytest.approx(S[0, N - 1, 0])
