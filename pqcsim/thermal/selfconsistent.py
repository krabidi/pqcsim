"""Self-consistent heating with temperature-dependent absorption.

Heat flow is linear in the source, so for piecewise-constant heating power q_j on [t_j, t_{j+1}) the probe
temperature is exactly
    dT(t_i) = Σ_{j<i} q_j [S(t_i - t_j) - S(t_i - t_{j+1})]
with S(t) the step response (temperature rise for unit power switched on at t = 0), taken from the full
solver. With temperature-dependent absorption, q_j = q0 A(T(t_j)) / A(T0): marching in time gives the
self-consistent history (explicit in q, exact convolution). The spatial heating pattern is held fixed; only its
amplitude follows A(T).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def solve_selfconsistent(t, S, power: Callable[[float], float], t_pulse: float):
    """t: uniform grid starting at 0; S: step response on that grid (S[0] = 0); power(dT) -> heating power
    (same units as the S normalisation) while t < t_pulse, zero afterwards. Returns dT(t) and q(t)."""
    t = np.asarray(t, float)
    S = np.asarray(S, float)
    n = t.size
    K = np.diff(S)                                   # K[m-1] = S(m dt) - S((m-1) dt), m = 1..n-1
    dT = np.zeros(n)
    q = np.zeros(n)
    for i in range(1, n):
        q[i - 1] = power(dT[i - 1]) if t[i - 1] < t_pulse - 1e-15 else 0.0
        m = i - np.arange(i)                         # lags (in steps) of the past intervals j = 0..i-1
        dT[i] = np.dot(q[:i], K[m - 1])
    return dT, q
