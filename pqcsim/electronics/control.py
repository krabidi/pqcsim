"""Control electronics for reconfigurable photonics: heaters, DACs, thermal crosstalk, feed-forward.

Thermo-optic phase shifter of length L heated by power P through thermal resistance R_th:
    dphi = (2 pi / lambda) (dn/dT) L R_th P        (linear in power),   P = V^2 / R_heater
Thermal crosstalk between N heaters: dT_i = sum_j K_ij P_j  (K from measurement or a heat solver),
so phases phi = D K P with D = diag(2 pi L_i dn/dT / lambda). Compensation solves D K P = phi_target
(mod 2 pi) with P >= 0.
Feed-forward: a photon waiting for the electronic latency tau in a delay line of group index n_g
travels L = c tau / n_g and keeps the power fraction 10^(-alpha L / 10).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

C_LIGHT = 299792458.0


@dataclass(frozen=True)
class ThermoOpticHeater:
    length_um: float
    dn_dT: float = 1.86e-4          # silicon at 1550 nm, 1/K
    R_th: float = 1.0e4             # K/W, heater to waveguide
    R_heater: float = 1.0e3         # Ohm
    lam_nm: float = 1550.0

    @property
    def phase_per_watt(self) -> float:
        return 2 * math.pi / (self.lam_nm * 1e-9) * self.dn_dT * self.length_um * 1e-6 * self.R_th

    @property
    def P_pi(self) -> float:
        return math.pi / self.phase_per_watt

    def phase(self, V: np.ndarray) -> np.ndarray:
        return self.phase_per_watt * np.asarray(V, float) ** 2 / self.R_heater

    def voltage_for_phase(self, phi: float) -> float:
        phi = phi % (2 * math.pi)
        return math.sqrt(phi / self.phase_per_watt * self.R_heater)

    def phase_noise_std(self, V: float, sigma_V: float) -> float:
        """First-order phase noise from voltage noise: dphi/dV = 2 (dphi/dP) V / R."""
        return 2 * self.phase_per_watt * V / self.R_heater * sigma_V


@dataclass(frozen=True)
class DAC:
    bits: int = 12
    v_fullscale: float = 5.0
    noise_rms: float = 0.0           # V

    def __post_init__(self):
        if self.bits < 1 or self.v_fullscale <= 0 or self.noise_rms < 0:
            raise ValueError("bits >= 1, v_fullscale > 0, noise_rms >= 0 required")

    @property
    def lsb(self) -> float:
        return self.v_fullscale / (2 ** self.bits - 1)

    def output(self, V: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        q = np.clip(np.rint(np.asarray(V, float) / self.lsb), 0, 2 ** self.bits - 1) * self.lsb
        if rng is not None and self.noise_rms > 0:
            q = q + rng.normal(0.0, self.noise_rms, np.shape(q))
        return q


def crosstalk_phases(D: Sequence[float], K: np.ndarray, P: Sequence[float]) -> np.ndarray:
    return np.asarray(D, float) * (np.asarray(K, float) @ np.asarray(P, float))


def heater_powers_for_phases(D: Sequence[float], K: np.ndarray, phi_target: Sequence[float],
                             max_wraps: int = 3) -> np.ndarray:
    """Non-negative heater powers giving phi_target (mod 2 pi) despite thermal crosstalk.
    Searches 2 pi offsets of the targets and returns the solution with the lowest total power."""
    A = np.asarray(D, float)[:, None] * np.asarray(K, float)
    target = np.mod(np.asarray(phi_target, float), 2 * np.pi)
    N = target.size
    best = None
    for wraps in np.ndindex(*([max_wraps + 1] * N)):
        P = np.linalg.solve(A, target + 2 * np.pi * np.array(wraps))
        if np.all(P >= -1e-12) and (best is None or P.sum() < best.sum()):
            best = np.clip(P, 0.0, None)
    if best is None:
        raise ValueError("no non-negative solution within the searched phase wraps")
    return best


def feedforward_delay_transmission(latency_s: float, n_group: float, loss_dB_per_cm: float) -> float:
    if latency_s < 0 or n_group <= 0 or loss_dB_per_cm < 0:
        raise ValueError("latency >= 0, n_group > 0, loss >= 0 required")
    L_cm = C_LIGHT * latency_s / n_group * 100
    return 10 ** (-loss_dB_per_cm * L_cm / 10)
