"""Multiphoton interference with partially distinguishable photons.

Photon j enters spatial mode k_j with internal (spectral/temporal/polarization) state |phi_j>;
S_ij = <phi_i|phi_j> is their Gram matrix. Detectors resolve photon number per spatial mode but not
the internal state. For an output pattern s with row list r and M = U[r, k] (N x N):

    P(s) = 1 / (||psi_in||^2 prod_l s_l!)  sum_{sigma in S_N} ( prod_j S_{sigma(j), j} ) perm(M o conj(M[:, sigma]))

with ||psi_in||^2 = sum over permutations sigma that keep every photon in its own input mode of
prod_j S_{j, sigma(j)}  (= prod n_k! for identical photons). Limits: S = all ones gives
|perm M|^2 / (prod n_k! prod s_l!); S = identity gives perm(|M|^2) / prod s_l!.
(Tichy, J. Phys. B 47, 103001 (2014); Shchesnovich, PRA 91, 013844 (2015).)

Losses: the network's transfer matrix is dilated to a unitary (Halmos) and the lost-photon
patterns are summed over. Cost grows as N! 2^N N per output pattern: meant for N <= ~6.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Dict, Sequence, Tuple

import numpy as np

from ..circuits.network import LinearNetwork
from .kernels import permanent
from .reference import halmos_dilation


def _modes_list(occupation: Sequence[int]):
    return [m for m, n in enumerate(occupation) for _ in range(n)]


def _norm2(S: np.ndarray, k: Sequence[int]) -> float:
    N = len(k)
    total = 0.0
    for sig in itertools.permutations(range(N)):
        if all(k[sig[j]] == k[j] for j in range(N)):
            total += np.prod([S[j, sig[j]] for j in range(N)]).real
    return total


def _pattern_probability(U: np.ndarray, k: Sequence[int], S: np.ndarray, out: Sequence[int],
                         perms) -> float:
    r = _modes_list(out)
    M = U[np.ix_(r, k)]
    total = 0.0 + 0.0j
    for sig in perms:
        w = np.prod([S[sig[j], j] for j in range(len(k))])
        if abs(w) < 1e-15:
            continue
        total += w * permanent(M * np.conj(M[:, list(sig)]))
    return total.real / np.prod([math.factorial(n) for n in out])


def simulate_partial_distinguishability(network: LinearNetwork, input_modes: Sequence[int],
                                        gram: np.ndarray, tol: float = 1e-14) -> Dict[Tuple[int, ...], float]:
    """Detection probabilities on the network modes.

    input_modes[j] is the spatial input mode of photon j; gram[i, j] = <phi_i|phi_j> (Hermitian,
    positive semidefinite, unit diagonal).
    """
    S = np.asarray(gram, complex)
    N = len(input_modes)
    if S.shape != (N, N):
        raise ValueError("gram matrix must be N x N for N photons")
    if not np.allclose(S, S.conj().T, atol=1e-12) or not np.allclose(np.diag(S), 1.0, atol=1e-12):
        raise ValueError("gram matrix must be Hermitian with unit diagonal")
    if np.linalg.eigvalsh(S).min() < -1e-10:
        raise ValueError("gram matrix must be positive semidefinite")
    n = network.n_modes
    if any(not 0 <= m < n for m in input_modes):
        raise IndexError("input mode out of range")
    T = network.matrix()
    lossless = network.is_lossless()
    U = T if lossless else halmos_dilation(T)
    n_tot = U.shape[0]
    k = list(input_modes)
    perms = list(itertools.permutations(range(N)))
    norm2 = _norm2(S, k)
    probs: Dict[Tuple[int, ...], float] = defaultdict(float)
    for out in itertools.product(range(N + 1), repeat=n_tot):
        if sum(out) != N:
            continue
        p = _pattern_probability(U, k, S, out, perms) / norm2
        probs[tuple(out[:n])] += p
    return {key: v for key, v in probs.items() if v > tol}


def gaussian_overlap(delay_s: float = 0.0, sigma_omega: float = 1.0, detuning: float = 0.0,
                     omega_mean: float = 0.0) -> complex:
    """<phi_1|phi_2> for Gaussian spectral amplitudes phi(w) ∝ exp(-(w - w_c)^2 / (4 sigma^2))
    (|phi|^2 has standard deviation sigma_omega, rad/s), centres w_c = omega_mean -/+ detuning/2,
    photon 2 delayed by delay_s:
        S = exp(-detuning^2 / (8 sigma^2)) exp(-sigma^2 tau^2 / 2) exp(i omega_mean tau).
    HOM coincidence probability on a 50:50 splitter is (1 - |S|^2) / 2."""
    return (math.exp(-detuning ** 2 / (8 * sigma_omega ** 2)) * math.exp(-0.5 * (sigma_omega * delay_s) ** 2)
            * complex(math.cos(omega_mean * delay_s), math.sin(omega_mean * delay_s)))
