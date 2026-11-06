"""Normal-incidence transfer-matrix method for planar multilayers.

Fields in layer j (local coordinate z from its left edge):
    E(z) = A e^{+i k0 n z} + B e^{-i k0 n z},      H(z) ∝ n [A e^{+i k0 n z} - B e^{-i k0 n z}]
Time-averaged Poynting flux (in units of |E|^2 / (2 Z0)), with a = A e^{ik0nz}, b = B e^{-ik0nz}:
    S = Re(E H*) = n' (|a|^2 - |b|^2) + 2 k Im(b a*),        n~ = n' + i k
The interference term 2k Im(b a*) matters inside absorbing layers; dropping it misallocates
absorption between layers and breaks R + T + sum(A_j) = 1.

Layer absorption A_j = (S_left - S_right) / S_inc, so R + T + sum_j A_j = 1 exactly (energy conservation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from .._compat import trapezoid


@dataclass
class TMMResult:
    r: complex           # reflection amplitude
    t: complex           # transmission amplitude (field in substrate / incident field)
    R: float             # reflected power fraction
    T: float             # transmitted power fraction (into the substrate)
    A_layers: np.ndarray  # absorbed power fraction in each finite layer
    amplitudes: List[np.ndarray]  # (A, B) at the LEFT edge of each finite layer (incident field = 1)

    @property
    def A_total(self) -> float:
        return float(self.A_layers.sum())


def _interface(n_left: complex, n_right: complex) -> np.ndarray:
    """Map (A, B) just right of an interface to (A, B) just left of it (E and H continuous)."""
    rho = n_right / n_left
    return 0.5 * np.array([[1 + rho, 1 - rho], [1 - rho, 1 + rho]], dtype=complex)


def _flux(n: complex, ab: np.ndarray) -> float:
    """Exact time-averaged Poynting flux Re(E H*) of the amplitude pair ab = (A, B) at one point."""
    A, B = ab
    return float(np.real((A + B) * np.conj(n * (A - B))))


def _check_passive(n: complex, what: str) -> None:
    if n.imag < -1e-12:
        raise ValueError(f"{what} has Im(n) = {n.imag} < 0 (gain); pqcsim uses n + ik with k >= 0")
    if n.real <= 0:
        raise ValueError(f"{what} has Re(n) = {n.real} <= 0")


def tmm(lam_nm: float, n_inc: complex, n_layers: Sequence[complex], d_nm: Sequence[float],
        n_sub: complex) -> TMMResult:
    """Solve a stack  incidence | layer_1 ... layer_N | substrate  at normal incidence.

    The incidence medium must be lossless (otherwise the incident flux is ill-defined).
    """
    if len(n_layers) != len(d_nm):
        raise ValueError("n_layers and d_nm must have the same length")
    if lam_nm <= 0 or any(d < 0 for d in d_nm):
        raise ValueError("wavelength must be positive and thicknesses non-negative")
    n_inc, n_sub = complex(n_inc), complex(n_sub)
    if abs(n_inc.imag) > 1e-12:
        raise ValueError("incidence medium must be lossless")
    for k, n in enumerate([n_inc, *n_layers, n_sub]):
        _check_passive(complex(n), f"medium {k}")

    k0 = 2 * np.pi / (lam_nm * 1e-9)
    n_all = [n_inc] + [complex(n) for n in n_layers] + [n_sub]
    N = len(n_layers)

    # Start in the substrate with a purely outgoing wave of unit amplitude, propagate leftwards.
    ab = np.array([1.0, 0.0], dtype=complex)
    left_edges: List[np.ndarray] = [None] * N
    for j in range(N, 0, -1):
        ab = _interface(n_all[j], n_all[j + 1]) @ ab            # right edge of layer j
        delta = k0 * n_all[j] * d_nm[j - 1] * 1e-9
        ab = np.array([ab[0] * np.exp(-1j * delta), ab[1] * np.exp(1j * delta)])  # left edge
        left_edges[j - 1] = ab
    ab_inc = _interface(n_all[0], n_all[1]) @ ab

    scale = 1.0 / ab_inc[0]                     # normalise to unit incident amplitude
    r = complex(ab_inc[1] * scale)
    t = complex(scale)
    left_edges = [v * scale for v in left_edges]

    S_inc = n_inc.real
    R = abs(r) ** 2
    T = n_sub.real * abs(t) ** 2 / S_inc        # outgoing wave only: S = Re(n_sub) |t|^2

    A_layers = np.zeros(N)
    for j in range(N):
        n = n_all[j + 1]
        delta = k0 * n * d_nm[j] * 1e-9
        a, b = left_edges[j]
        ab_right = np.array([a * np.exp(1j * delta), b * np.exp(-1j * delta)])
        A_layers[j] = (_flux(n, left_edges[j]) - _flux(n, ab_right)) / S_inc
    return TMMResult(r=r, t=t, R=float(R), T=float(T), A_layers=A_layers, amplitudes=left_edges)


def field_profile(lam_nm: float, n_layers: Sequence[complex], d_nm: Sequence[float],
                  result: TMMResult, points_per_layer: int = 200):
    """|E(z)|^2 inside every finite layer (incident amplitude 1). Returns list of (z_nm, E) per layer."""
    k0 = 2 * np.pi / (lam_nm * 1e-9)
    out, z0 = [], 0.0
    for n, d, (a, b) in zip(n_layers, d_nm, result.amplitudes):
        z = np.linspace(0.0, d, points_per_layer)
        phase = k0 * complex(n) * z * 1e-9
        out.append((z0 + z, a * np.exp(1j * phase) + b * np.exp(-1j * phase)))
        z0 += d
    return out


def absorption_by_field_integral(lam_nm: float, n_inc: complex, n_layers: Sequence[complex],
                                 d_nm: Sequence[float], result: TMMResult, points: int = 20001) -> np.ndarray:
    """Independent check of A_layers:  A_j = k0 Im(eps_j) ∫ |E|^2 dz / Re(n_inc)."""
    k0 = 2 * np.pi / (lam_nm * 1e-9)
    prof = field_profile(lam_nm, n_layers, d_nm, result, points)
    return np.array([k0 * (complex(n) ** 2).imag * trapezoid(abs(E) ** 2, z * 1e-9) / complex(n_inc).real
                     for n, (z, E) in zip(n_layers, prof)])


def _int_exp(c: complex, z1: float, z2: float) -> complex:
    """∫_{z1}^{z2} e^{c z} dz, stable for c -> 0."""
    if abs(c) * (z2 - z1) < 1e-8:
        return (z2 - z1) * np.exp(c * 0.5 * (z1 + z2))
    return (np.exp(c * z2) - np.exp(c * z1)) / c


def absorption_profile(lam_nm: float, n_inc: complex, n_layers: Sequence[complex], d_nm: Sequence[float],
                       result: TMMResult, splits: Sequence[int]) -> List[np.ndarray]:
    """Absorbed power fraction in equal sub-slabs of every layer (layer j split into splits[j] slabs).

    Exact closed-form integral of k0 Im(eps) |E|^2 / Re(n_inc), with
        |E|^2 = |A|^2 e^{-2 b'' z} + |B|^2 e^{2 b'' z} + 2 Re(A B* e^{2 i b' z}),   b = k0 n.
    The slabs of layer j sum to result.A_layers[j].
    """
    if len(splits) != len(n_layers):
        raise ValueError("need one split count per layer")
    k0 = 2 * np.pi / (lam_nm * 1e-9)
    out = []
    for n, d, (A, B), m in zip(n_layers, d_nm, result.amplitudes, splits):
        n = complex(n)
        if m < 1:
            raise ValueError("split counts must be >= 1")
        b = k0 * n
        pref = k0 * (n * n).imag / complex(n_inc).real
        edges = np.linspace(0.0, d * 1e-9, m + 1)
        vals = []
        for z1, z2 in zip(edges[:-1], edges[1:]):
            I = (abs(A) ** 2 * _int_exp(-2 * b.imag, z1, z2) + abs(B) ** 2 * _int_exp(2 * b.imag, z1, z2)
                 + 2 * (A * np.conj(B) * _int_exp(2j * b.real, z1, z2)).real)
            vals.append(pref * float(np.real(I)))
        out.append(np.array(vals))
    return out
