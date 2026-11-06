"""Physical integrated-photonic components (monochromatic, single polarization).

Coupler convention (same as BeamSplitter): [[t, i k], [i k, t]], t = cos(theta), k = sin(theta).
All components return passive transfer matrices; passive_elements() turns them into network
elements, so they work unchanged in classical and quantum simulations.
"""
from __future__ import annotations

import cmath
import math
import re
from typing import List, Sequence, Tuple

import numpy as np

from .decompose import passive_elements
from .network import BeamSplitter, Element


def directional_coupler(i: int, j: int, length_um: float, lam_nm: float, dn_super: float,
                        ddn_dlam_per_nm: float = 0.0, lam0_nm: float = 1550.0) -> BeamSplitter:
    """Coupled-mode directional coupler. kappa = pi dn(lambda) / lambda, theta = kappa L, with the
    supermode index difference dn = n_even - n_odd linearised around lam0."""
    dn = dn_super + ddn_dlam_per_nm * (lam_nm - lam0_nm)
    if dn <= 0 or length_um < 0:
        raise ValueError("supermode index difference must be > 0 and length >= 0")
    theta = math.pi * dn / (lam_nm * 1e-9) * length_um * 1e-6
    return BeamSplitter(i, j, theta, 0.0)


def coupler_length_for_split(power_cross: float, lam_nm: float, dn_super: float) -> float:
    """Shortest length (um) giving cross-power fraction sin^2(theta) = power_cross."""
    if not 0 <= power_cross <= 1:
        raise ValueError("power_cross must be in [0, 1]")
    theta = math.asin(math.sqrt(power_cross))
    return theta * (lam_nm * 1e-9) / (math.pi * dn_super) * 1e6


def _ring_phase(lam_nm, length_um, n_eff, n_g, lam0_nm):
    # first-order dispersion: n_eff(lam) = n_eff0 - (n_g - n_eff0) (lam - lam0) / lam0
    n = n_eff - (n_g - n_eff) * (lam_nm - lam0_nm) / lam0_nm
    return 2 * math.pi * n * length_um * 1e3 / lam_nm


def _round_trip_amplitude(length_um, loss_dB_per_cm):
    return 10 ** (-loss_dB_per_cm * length_um * 1e-4 / 20)


def ring_allpass(lam_nm: float, length_um: float, n_eff: float, n_g: float, theta: float,
                 loss_dB_per_cm: float = 0.0, lam0_nm: float = 1550.0) -> complex:
    """Through amplitude of an all-pass ring:  (t - a e^{i phi}) / (1 - t a e^{i phi})."""
    t = math.cos(theta)
    a = _round_trip_amplitude(length_um, loss_dB_per_cm)
    e = a * cmath.exp(1j * _ring_phase(lam_nm, length_um, n_eff, n_g, lam0_nm))
    return (t - e) / (1 - t * e)


def ring_add_drop(lam_nm: float, length_um: float, n_eff: float, n_g: float, theta1: float, theta2: float,
                  loss_dB_per_cm: float = 0.0, lam0_nm: float = 1550.0) -> np.ndarray:
    """2x2 transfer matrix (inputs: in, add; outputs: through, drop) of an add-drop ring.

    Derived self-consistently with the coupler convention above; the drop port is half-way round:
      through = (t1 - t2 a e^{i phi}) / D,   drop = -k1 k2 sqrt(a) e^{i phi/2} / D,   D = 1 - t1 t2 a e^{i phi}
      add->through (= drop by reciprocity), add->drop = (t2 - t1 a e^{i phi}) / D
    """
    t1, k1, t2, k2 = math.cos(theta1), math.sin(theta1), math.cos(theta2), math.sin(theta2)
    a = _round_trip_amplitude(length_um, loss_dB_per_cm)
    phi = _ring_phase(lam_nm, length_um, n_eff, n_g, lam0_nm)
    e = a * cmath.exp(1j * phi)
    D = 1 - t1 * t2 * e
    drop = -k1 * k2 * math.sqrt(a) * cmath.exp(0.5j * phi) / D
    return np.array([[(t1 - t2 * e) / D, drop], [drop, (t2 - t1 * e) / D]])


def matrix_elements(T: np.ndarray, modes: Sequence[int]) -> List[Element]:
    """Network elements for an arbitrary passive matrix acting on `modes`."""
    return passive_elements(T, modes)


# ------------------------------------------------------------------ Touchstone S-parameters
_FREQ = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9, "THZ": 1e12}


def load_touchstone(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a Touchstone v1 .sNp file. Returns (frequency_Hz, S[n_freq, N, N]), S[out][in].

    Supports RI, MA (degrees) and DB (dB, degrees) formats. For N = 2 the file order is
    S11 S21 S12 S22 (column-major); for N >= 3 it is row-major (S11 S12 ... S1N, S21 ...).
    """
    m = re.search(r"\.s(\d+)p$", path.lower())
    if not m:
        raise ValueError("file name must end in .sNp")
    N = int(m.group(1))
    unit, fmt = "GHZ", "MA"
    numbers: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("!")[0].strip()
            if not line:
                continue
            if line.startswith("#"):
                tok = line[1:].upper().split()
                for tkn in tok:
                    if tkn in _FREQ:
                        unit = tkn
                    elif tkn in ("RI", "MA", "DB"):
                        fmt = tkn
                if "S" not in tok:
                    raise ValueError("only S-parameter files are supported")
                continue
            numbers += [float(v) for v in line.split()]
    per = 1 + 2 * N * N
    if len(numbers) % per:
        raise ValueError("data length does not match the port count")
    data = np.array(numbers).reshape(-1, per)
    freq = data[:, 0] * _FREQ[unit]
    a, b = data[:, 1::2], data[:, 2::2]
    if fmt == "RI":
        vals = a + 1j * b
    elif fmt == "MA":
        vals = a * np.exp(1j * np.deg2rad(b))
    else:
        vals = 10 ** (a / 20) * np.exp(1j * np.deg2rad(b))
    S = vals.reshape(-1, N, N)
    if N == 2:
        S = np.transpose(S, (0, 2, 1))     # file order S11 S21 S12 S22 -> [out][in]
    return freq, S


def transfer_from_sparams(S: np.ndarray, in_ports: Sequence[int], out_ports: Sequence[int]) -> np.ndarray:
    """Forward transfer matrix T[out][in] for a unidirectional circuit (reflections ignored)."""
    if len(in_ports) != len(out_ports):
        raise ValueError("need the same number of input and output ports")
    return np.asarray(S)[np.ix_(list(out_ports), list(in_ports))]
