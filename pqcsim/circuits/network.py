"""Classical linear optical networks of single-mode waveguides.

A network on M modes is a complex transfer matrix T (fields):  E_out = T E_in,  T[out][in].
Every element is passive, so T is a contraction (largest singular value <= 1); it is unitary
iff the network is lossless. The same element list drives the quantum simulation
(pqcsim.quantum.lossy), so classical and quantum results come from one circuit description.

Conventions match the C++ Fock engine:
  beamsplitter  [[cos t, i e^{-i phi} sin t], [i e^{i phi} sin t, cos t]]   (theta = pi/4 -> 50:50)
  phase shift   E -> e^{i phi} E
  loss          E -> sqrt(eta) E
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Union

import numpy as np

from ..materials import PCMMaterial


@dataclass(frozen=True)
class BeamSplitter:
    i: int
    j: int
    theta: float = math.pi / 4
    phi: float = 0.0


@dataclass(frozen=True)
class PhaseShift:
    mode: int
    phi: float


@dataclass(frozen=True)
class Loss:
    """Power transmission eta in [0, 1] on one mode (amplitude sqrt(eta))."""
    mode: int
    eta: float

    def __post_init__(self):
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError(f"loss transmission eta={self.eta} outside [0, 1]")


Element = Union[BeamSplitter, PhaseShift, Loss]


def single_mode_elements(mode: int, t: complex) -> List[Element]:
    """Decompose a passive single-mode transmission t (|t| <= 1) into PhaseShift + Loss."""
    if abs(t) > 1 + 1e-12:
        raise ValueError(f"|t| = {abs(t)} > 1 is not passive")
    return [PhaseShift(mode, float(np.angle(t))), Loss(mode, min(abs(t) ** 2, 1.0))]


def waveguide(mode: int, length_um: float, n_eff: float, lam_nm: float,
              loss_dB_per_cm: float = 0.0) -> List[Element]:
    """Straight waveguide: phase 2 pi n_eff L / lambda, power loss 10^(-alpha L / 10)."""
    if length_um < 0 or loss_dB_per_cm < 0:
        raise ValueError("length and loss must be non-negative")
    phi = 2 * math.pi * n_eff * length_um * 1e3 / lam_nm
    eta = 10 ** (-loss_dB_per_cm * length_um * 1e-4 / 10)
    return [PhaseShift(mode, phi), Loss(mode, eta)]


def pcm_segment(mode: int, material: PCMMaterial, x: float, length_um: float, lam_nm: float,
                confinement: float, n_ref: complex, channel: str = "probe") -> List[Element]:
    """PCM patch on a waveguide, relative to the same waveguide with a reference cladding n_ref.

    First-order perturbation theory:  delta n_eff = Gamma (n_pcm(x) - n_ref),  with the confinement
    factor Gamma taken from a mode solver. Valid for weak perturbations (small Gamma |delta n|);
    use a full mode solve for strongly confined PCM. Only the change relative to n_ref is returned.
    """
    if not 0.0 <= confinement <= 1.0:
        raise ValueError("confinement factor must be in [0, 1]")
    dn = confinement * (material.nk(x, lam_nm, channel) - complex(n_ref))
    if dn.imag < -1e-15:
        raise ValueError("PCM segment would have gain; n_ref must not be more absorbing than the PCM")
    k0 = 2 * math.pi / (lam_nm * 1e-9)
    t = np.exp(1j * k0 * dn * length_um * 1e-6)
    return single_mode_elements(mode, complex(t))


def element_matrix(el: Element, n_modes: int) -> np.ndarray:
    U = np.eye(n_modes, dtype=complex)
    if isinstance(el, BeamSplitter):
        c, s = math.cos(el.theta), math.sin(el.theta)
        U[el.i, el.i] = U[el.j, el.j] = c
        U[el.i, el.j] = 1j * np.exp(-1j * el.phi) * s
        U[el.j, el.i] = 1j * np.exp(1j * el.phi) * s
    elif isinstance(el, PhaseShift):
        U[el.mode, el.mode] = np.exp(1j * el.phi)
    elif isinstance(el, Loss):
        U[el.mode, el.mode] = math.sqrt(el.eta)
    else:
        raise TypeError(f"unknown element {el!r}")
    return U


class LinearNetwork:
    def __init__(self, n_modes: int, elements: Sequence[Element] = ()):
        if n_modes < 1:
            raise ValueError("need at least one mode")
        self.n_modes = n_modes
        self.elements: List[Element] = []
        self.extend(elements)

    def _check(self, el: Element) -> None:
        modes = (el.i, el.j) if isinstance(el, BeamSplitter) else (el.mode,)
        for m in modes:
            if not 0 <= m < self.n_modes:
                raise IndexError(f"mode {m} out of range for {self.n_modes}-mode network")
        if isinstance(el, BeamSplitter) and el.i == el.j:
            raise ValueError("beamsplitter needs two distinct modes")

    def add(self, el: Element) -> "LinearNetwork":
        self._check(el)
        self.elements.append(el)
        return self

    def extend(self, els: Sequence[Element]) -> "LinearNetwork":
        for el in els:
            self.add(el)
        return self

    def matrix(self) -> np.ndarray:
        T = np.eye(self.n_modes, dtype=complex)
        for el in self.elements:
            T = element_matrix(el, self.n_modes) @ T
        return T

    def propagate_fields(self, E_in: Sequence[complex]) -> np.ndarray:
        """Coherent fields (all inputs from one laser)."""
        return self.matrix() @ np.asarray(E_in, dtype=complex)

    def propagate_powers(self, P_in: Sequence[float]) -> np.ndarray:
        """Mutually incoherent inputs: powers add, P_out = |T|^2 P_in (element-wise |.|^2)."""
        return np.abs(self.matrix()) ** 2 @ np.asarray(P_in, dtype=float)

    def is_lossless(self, tol: float = 1e-10) -> bool:
        T = self.matrix()
        return bool(np.allclose(T.conj().T @ T, np.eye(self.n_modes), atol=tol))
