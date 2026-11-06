"""Photodetector + transimpedance amplifier + ADC, with shot, dark and thermal noise.

Photocurrent I = R_resp P + I_dark. Noise (one-sided bandwidth B, Gaussian approximation valid
for many photoelectrons per integration time):
    sigma_I^2 = 2 q (R_resp P + I_dark) B  +  4 kB T B / R_f
Output voltage V = I R_f, then an N-bit ADC over [0, V_fs] with clipping.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

Q_E = 1.602176634e-19
K_B = 1.380649e-23


@dataclass
class Photodetector:
    responsivity: float = 1.0      # A/W
    dark_current: float = 1e-9     # A
    bandwidth: float = 1e9         # Hz
    R_f: float = 1e3               # TIA feedback resistance, Ohm
    temperature: float = 300.0     # K

    def mean_current(self, P: np.ndarray) -> np.ndarray:
        return self.responsivity * np.asarray(P, float) + self.dark_current

    def current_noise_std(self, P: np.ndarray) -> np.ndarray:
        shot = 2 * Q_E * self.mean_current(P) * self.bandwidth
        thermal = 4 * K_B * self.temperature * self.bandwidth / self.R_f
        return np.sqrt(shot + thermal)

    def voltage(self, P: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        I = self.mean_current(P)
        if rng is not None:
            I = I + rng.normal(0.0, 1.0, np.shape(I)) * self.current_noise_std(P)
        return I * self.R_f


@dataclass
class ADC:
    bits: int = 8
    v_fullscale: float = 1.0

    def __post_init__(self):
        if self.bits < 1 or self.v_fullscale <= 0:
            raise ValueError("bits >= 1 and v_fullscale > 0 required")

    @property
    def lsb(self) -> float:
        return self.v_fullscale / (2 ** self.bits - 1)

    def codes(self, V: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(np.asarray(V, float) / self.lsb), 0, 2 ** self.bits - 1).astype(int)

    def quantize(self, V: np.ndarray) -> np.ndarray:
        return self.codes(V) * self.lsb
