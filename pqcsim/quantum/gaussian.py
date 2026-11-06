"""Gaussian states: squeezed-light sources (SPDC / SFWM), lossy linear networks, PNR and click detection.

Conventions: hbar = 1, quadratures x = (a + a^dag)/sqrt 2, p = (a - a^dag)/(i sqrt 2); ordering
R = (x_1..x_n, p_1..p_n); vacuum covariance sigma = I/2; mean vector d = <R>.

Passive network with transfer matrix T (a_out = T a_in, any contraction):
    X = [[Re T, -Im T], [Im T, Re T]],   sigma -> X sigma X^T + (I - X X^T)/2,   d -> X d
(the second term is the vacuum injected by loss).
Bogoliubov map a -> A a + B a^dag is the symplectic matrix [[Re(A+B), -Im(A-B)], [Im(A+B), Re(A-B)]].

Photon-number probabilities (zero mean), Hamilton et al., PRL 119, 170501 (2017):
    sigma_A = W sigma W^dag  (W maps R to (a, a^dag)),  Q = sigma_A + I/2,  A = X2 (I - Q^{-1}),
    P(n) = haf(A_n) / (sqrt(det Q) prod n_i!),   A_n = A with mode i's row/column repeated n_i times.
Vacuum probability of a set of modes: P0 = exp(-d^T (sigma + I/2)^{-1} d / 2) / sqrt(det(sigma + I/2)),
which gives threshold-detector (click) probabilities by inclusion-exclusion.
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import Sequence

import numpy as np

from ..circuits.network import LinearNetwork


def hafnian(A: np.ndarray) -> complex:
    """Hafnian by recursion over perfect matchings with memoisation on the remaining index set."""
    A = np.asarray(A, complex)
    n = A.shape[0]
    if n % 2:
        return 0.0
    if n == 0:
        return 1.0

    @lru_cache(maxsize=None)
    def rec(mask: int) -> complex:
        if mask == 0:
            return 1.0
        i = (mask & -mask).bit_length() - 1
        rest = mask & ~(1 << i)
        total = 0.0
        m = rest
        while m:
            j = (m & -m).bit_length() - 1
            m &= m - 1
            if A[i, j] != 0:
                total += A[i, j] * rec(rest & ~(1 << j))
        return total

    return complex(rec((1 << n) - 1))


def symplectic_from_bogoliubov(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A, B = np.asarray(A, complex), np.asarray(B, complex)
    return np.block([[np.real(A + B), -np.imag(A - B)], [np.imag(A + B), np.real(A - B)]])


class GaussianState:
    def __init__(self, n_modes: int):
        if n_modes < 1:
            raise ValueError("need at least one mode")
        self.n = n_modes
        self.cov = 0.5 * np.eye(2 * n_modes)
        self.mean = np.zeros(2 * n_modes)

    # ----------------------------------------------------------------- operations
    def _embed(self, modes: Sequence[int], A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = self.n
        AA, BB = np.eye(n, dtype=complex), np.zeros((n, n), complex)
        idx = np.ix_(list(modes), list(modes))
        AA[idx], BB[idx] = A, B
        return symplectic_from_bogoliubov(AA, BB)

    def apply_symplectic(self, S: np.ndarray) -> "GaussianState":
        Om = np.block([[np.zeros((self.n, self.n)), np.eye(self.n)], [-np.eye(self.n), np.zeros((self.n, self.n))]])
        if not np.allclose(S @ Om @ S.T, Om, atol=1e-9):
            raise ValueError("matrix is not symplectic")
        self.cov = S @ self.cov @ S.T
        self.mean = S @ self.mean
        return self

    def squeeze(self, mode: int, r: float, phi: float = 0.0) -> "GaussianState":
        """Single-mode squeezer S(z) = exp((z* a^2 - z a^dag^2)/2), z = r e^{i phi}:
        a -> cosh r a - e^{i phi} sinh r a^dag (phi = 0 squeezes x)."""
        return self.apply_symplectic(self._embed([mode], np.array([[math.cosh(r)]]),
                                                 np.array([[-np.exp(1j * phi) * math.sinh(r)]])))

    def two_mode_squeeze(self, i: int, j: int, r: float, phi: float = 0.0) -> "GaussianState":
        """Pair source (SPDC / SFWM, single-mode approximation): a_i -> cosh r a_i + e^{i phi} sinh r a_j^dag."""
        if i == j:
            raise ValueError("two-mode squeezing needs two distinct modes")
        A = math.cosh(r) * np.eye(2)
        B = np.exp(1j * phi) * math.sinh(r) * np.array([[0, 1], [1, 0]])
        return self.apply_symplectic(self._embed([i, j], A, B))

    def displace(self, mode: int, alpha: complex) -> "GaussianState":
        self.mean[mode] += math.sqrt(2) * alpha.real
        self.mean[self.n + mode] += math.sqrt(2) * alpha.imag
        return self

    def apply_network(self, network: LinearNetwork) -> "GaussianState":
        if network.n_modes != self.n:
            raise ValueError("network and state have different mode counts")
        T = network.matrix()
        X = np.block([[T.real, -T.imag], [T.imag, T.real]])
        self.cov = X @ self.cov @ X.T + 0.5 * (np.eye(2 * self.n) - X @ X.T)
        self.mean = X @ self.mean
        return self

    # ----------------------------------------------------------------- statistics
    def reduced(self, modes: Sequence[int]):
        idx = list(modes) + [self.n + m for m in modes]
        return self.cov[np.ix_(idx, idx)], self.mean[idx]

    def mean_photon_number(self, mode: int) -> float:
        cov, d = self.reduced([mode])
        return float(0.5 * (cov[0, 0] + cov[1, 1]) - 0.5 + 0.5 * (d[0] ** 2 + d[1] ** 2))

    def vacuum_probability(self, modes: Sequence[int]) -> float:
        if len(modes) == 0:
            return 1.0
        cov, d = self.reduced(modes)
        M = cov + 0.5 * np.eye(cov.shape[0])
        return float(np.exp(-0.5 * d @ np.linalg.solve(M, d)) / math.sqrt(np.linalg.det(M)))

    def click_probability(self, clicks: Sequence[int]) -> float:
        """Threshold detectors on every mode: clicks[i] = 1 (click) or 0 (no click)."""
        if len(clicks) != self.n or any(c not in (0, 1) for c in clicks):
            raise ValueError("clicks must be a 0/1 pattern over all modes")
        dark = [m for m, c in enumerate(clicks) if c == 0]
        lit = [m for m, c in enumerate(clicks) if c == 1]
        total = 0.0
        for k in range(len(lit) + 1):
            for E in itertools.combinations(lit, k):
                total += (-1) ** k * self.vacuum_probability(dark + list(E))
        return total

    def fock_probability(self, pattern: Sequence[int]) -> float:
        """Photon-number-resolving detection probability on every mode (zero-mean states)."""
        if len(pattern) != self.n or any(k < 0 for k in pattern):
            raise ValueError("pattern must give a non-negative photon number for every mode")
        if np.any(np.abs(self.mean) > 1e-12):
            raise NotImplementedError("displaced states need the loop hafnian (not implemented)")
        n = self.n
        W = np.block([[np.eye(n), 1j * np.eye(n)], [np.eye(n), -1j * np.eye(n)]]) / math.sqrt(2)
        Q = W @ self.cov @ W.conj().T + 0.5 * np.eye(2 * n)
        X2 = np.block([[np.zeros((n, n)), np.eye(n)], [np.eye(n), np.zeros((n, n))]])
        A = X2 @ (np.eye(2 * n) - np.linalg.inv(Q))
        rep = [m for m, k in enumerate(pattern) for _ in range(k)]
        idx = rep + [n + m for m in rep]
        from .kernels import hafnian as fast_hafnian
        h = fast_hafnian(A[np.ix_(idx, idx)])
        p = h / (np.sqrt(np.linalg.det(Q)) * np.prod([math.factorial(k) for k in pattern]))
        if abs(p.imag) > 1e-9 * max(1.0, abs(p.real)):
            raise RuntimeError("complex probability: numerical problem")
        return float(p.real)


def squeezing_for_mean_pairs(mu: float) -> float:
    """Two-mode squeezing r with mean pair number sinh^2 r = mu per mode pair."""
    if mu < 0:
        raise ValueError("mean pair number must be >= 0")
    return math.asinh(math.sqrt(mu))


def sfwm_squeezing(gamma_per_W_m: float, pump_peak_W: float, length_m: float) -> float:
    """Single-temporal-mode estimate r = gamma P L for spontaneous four-wave mixing in a waveguide
    (undepleted pump, perfect phase matching, no pump self-phase-modulation)."""
    if min(gamma_per_W_m, pump_peak_W, length_m) < 0:
        raise ValueError("gamma, pump power and length must be >= 0")
    return gamma_per_W_m * pump_peak_W * length_m
