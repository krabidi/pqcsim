"""Slow, independent reference calculations used to validate the fast engines (tests only)."""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.linalg import sqrtm


def permanent(a: np.ndarray) -> complex:
    """Ryser's formula, O(2^n n)."""
    n = a.shape[0]
    if n == 0:
        return 1.0
    total = 0.0
    for subset in range(1, 1 << n):
        cols = [k for k in range(n) if subset >> k & 1]
        total += (-1) ** len(cols) * np.prod(a[:, cols].sum(axis=1))
    return (-1) ** n * total


def fock_amplitude(U: np.ndarray, n_in: Sequence[int], n_out: Sequence[int]) -> complex:
    """<n_out| U |n_in> = Perm(U[rows, cols]) / sqrt(prod n_in! n_out!),  U[out][in]."""
    if sum(n_in) != sum(n_out):
        return 0.0
    rows = [m for m, k in enumerate(n_out) for _ in range(k)]
    cols = [m for m, k in enumerate(n_in) for _ in range(k)]
    norm = math.sqrt(np.prod([math.factorial(k) for k in (*n_in, *n_out)]))
    return permanent(U[np.ix_(rows, cols)]) / norm


def halmos_dilation(T: np.ndarray) -> np.ndarray:
    """Unitary [[T, sqrt(I - T T^H)], [sqrt(I - T^H T), -T^H]] for a contraction T."""
    n = T.shape[0]
    I = np.eye(n)
    D1 = sqrtm(I - T @ T.conj().T)
    D2 = sqrtm(I - T.conj().T @ T)
    U = np.block([[T, D1], [D2, -T.conj().T]])
    if not np.allclose(U.conj().T @ U, np.eye(2 * n), atol=1e-9):
        raise ValueError("T is not a contraction (network not passive)")
    return U


def lossy_probabilities(T: np.ndarray, n_in: Sequence[int]) -> Dict[Tuple[int, ...], float]:
    """Detection probabilities on the system modes, via Halmos dilation and permanents."""
    n, N = T.shape[0], sum(n_in)
    U = halmos_dilation(T)
    probs: Dict[Tuple[int, ...], float] = defaultdict(float)
    full_in = list(n_in) + [0] * n
    for out in itertools.product(range(N + 1), repeat=2 * n):
        if sum(out) == N:
            probs[out[:n]] += abs(fock_amplitude(U, full_in, out)) ** 2
    return dict(probs)
