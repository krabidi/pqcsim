"""Exact decomposition of passive linear optics into BeamSplitter / PhaseShift / Loss elements.

- 2x2 unitary:  U = diag(e^{i a1}, e^{i a2}) B(theta, 0) diag(e^{i b1}, e^{i b2}),  B = [[c, i s], [i s, c]]
- N x N unitary: nulling with 2x2 Givens rotations on adjacent modes in the Clements order
  (Clements et al., Optica 3, 1460 (2016)); N(N-1)/2 couplers, depth ~N.
- passive N x N (a contraction): singular value decomposition T = U diag(sigma) V^H, realised as
  V^H, then losses sigma_k^2 per mode, then U (sigma_k <= 1 is exactly passivity).
Every routine is verified by rebuilding the matrix from the returned elements.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from .network import BeamSplitter, Element, LinearNetwork, Loss, PhaseShift

_TOL = 1e-12


def two_mode_elements(i: int, j: int, U: np.ndarray) -> List[Element]:
    """Elements realising the 2x2 unitary U (U[out][in], ordering (i, j)) on modes i and j."""
    U = np.asarray(U, complex)
    if U.shape != (2, 2) or not np.allclose(U.conj().T @ U, np.eye(2), atol=1e-9):
        raise ValueError("U must be a 2x2 unitary")
    c, s = abs(U[0, 0]), abs(U[0, 1])
    theta = math.atan2(s, c)
    if s < _TOL:                       # diagonal: phases only
        a1, a2, b1, b2 = np.angle(U[0, 0]), np.angle(U[1, 1]), 0.0, 0.0
    elif c < _TOL:                     # anti-diagonal: full swap with phases
        a1, a2, b1 = np.angle(U[0, 1]) - math.pi / 2, np.angle(U[1, 0]) - math.pi / 2, 0.0
        b2 = 0.0
    else:
        b1 = 0.0
        a1 = np.angle(U[0, 0])
        a2 = np.angle(U[1, 0]) - math.pi / 2
        b2 = np.angle(U[0, 1]) - math.pi / 2 - a1
    els: List[Element] = [PhaseShift(i, float(b1)), PhaseShift(j, float(b2))]
    if theta > _TOL:
        els.append(BeamSplitter(i, j, theta, 0.0))
    els += [PhaseShift(i, float(a1)), PhaseShift(j, float(a2))]
    return [e for e in els if not (isinstance(e, PhaseShift) and abs(e.phi) < 1e-15)]


def _null_right(U, row, a, b):
    """V acting on columns (a, b) with (U[row, a], U[row, b]) V = (0, n)."""
    p, q = U[row, a], U[row, b]
    n = math.hypot(abs(p), abs(q))
    return np.array([[q, np.conj(p)], [-p, np.conj(q)]]) / n


def _null_left(U, col, a, b):
    """W acting on rows (a, b) with W (U[a, col], U[b, col])^T = (n, 0): zeroes the entry in row b."""
    p, q = U[a, col], U[b, col]
    n = math.hypot(abs(p), abs(q))
    return np.array([[np.conj(p), np.conj(q)], [-q, p]]) / n


def unitary_elements(U: np.ndarray, modes: Sequence[int] = None) -> List[Element]:
    """Elements realising an N x N unitary on `modes` (default 0..N-1), adjacent-mode couplers only."""
    U = np.array(U, complex)
    N = U.shape[0]
    if U.shape != (N, N) or not np.allclose(U.conj().T @ U, np.eye(N), atol=1e-9):
        raise ValueError("U must be square and unitary")
    modes = list(range(N)) if modes is None else list(modes)
    right, left = [], []          # (a, 2x2) in application order
    for i in range(N - 1):
        if i % 2 == 0:
            for j in range(i + 1):
                r, c = N - 1 - j, i - j
                if abs(U[r, c]) > _TOL:
                    V = _null_right(U, r, c, c + 1)
                    U[:, [c, c + 1]] = U[:, [c, c + 1]] @ V
                    right.append((c, V.conj().T))           # U_orig = ... (V^H on these columns) ...
        else:
            for j in range(1, i + 2):
                r, c = N + j - i - 2, j - 1
                if abs(U[r, c]) > _TOL:
                    W = _null_left(U, c, r - 1, r)
                    U[[r - 1, r], :] = W @ U[[r - 1, r], :]
                    left.append((r - 1, W.conj().T))
    D = np.diag(U)
    if not np.allclose(U, np.diag(D), atol=1e-9):
        raise RuntimeError("nulling did not diagonalise the unitary")
    # L_k..L_1 U_orig R_1..R_m = D   =>   U_orig = L_1^H .. L_k^H  D  R_m^H .. R_1^H.
    # Rightmost acts first: R_1^H, ..., R_m^H, then D, then L_k^H, ..., L_1^H.
    els: List[Element] = []
    for a, Vh in right:
        els += two_mode_elements(modes[a], modes[a + 1], Vh)
    els += [PhaseShift(modes[k], float(np.angle(D[k]))) for k in range(N) if abs(np.angle(D[k])) > 1e-15]
    for a, Wh in reversed(left):
        els += two_mode_elements(modes[a], modes[a + 1], Wh)
    return els


def passive_elements(T: np.ndarray, modes: Sequence[int] = None) -> List[Element]:
    """Elements realising any passive (contractive) N x N transfer matrix, via SVD."""
    T = np.asarray(T, complex)
    N = T.shape[0]
    if T.shape != (N, N):
        raise ValueError("T must be square")
    U, sig, Vh = np.linalg.svd(T)
    if sig.max() > 1 + 1e-9:
        raise ValueError(f"matrix is not passive: largest singular value {sig.max():.6f} > 1")
    modes = list(range(N)) if modes is None else list(modes)
    els = unitary_elements(Vh, modes)
    els += [Loss(modes[k], float(min(sig[k], 1.0) ** 2)) for k in range(N) if sig[k] < 1 - 1e-15]
    els += unitary_elements(U, modes)
    return els


def rebuild(elements: Sequence[Element], n_modes: int) -> np.ndarray:
    return LinearNetwork(n_modes, elements).matrix()
