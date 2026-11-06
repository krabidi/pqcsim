"""Two-parameter analytic model for a thin heated film on a vertical heat sink (Gaussian spot).

Lateral conduction in the film stack (diffusivity D_lat) plus vertical leakage to the sink (time constant
tau_v) give, for each lateral wavenumber k, a relaxation rate (1 + k^2 L_h^2)/tau_v with healing length
L_h = sqrt(D_lat tau_v). For a Gaussian source of 1/e^2 radius w switched on for tau (Hankel transform):
    dT(r, t) = A ∫_0^inf k dk J0(k r) e^{-k^2 w^2 / 8} F(k, t) / (1 + k^2 L_h^2),
    F = 1 - exp(-s t)                          for t <= tau,
    F = (1 - exp(-s tau)) exp(-s (t - tau))    for t >  tau,      s = (1 + k^2 L_h^2) / tau_v.
Limits: tau << tau_v gives Gaussian spreading w_eff^2 = w^2 + 8 D_lat tau (diffusion); tau >> tau_v gives the
steady healing-length profile (Gaussian convolved with a K0(r / L_h) kernel).
"""
from __future__ import annotations

import numpy as np
from scipy.special import j0


def peak_profile(r_um, w_um, tau, L_h_um, tau_v, n_k=4000, n_t=200):
    """Peak-in-time temperature rise at radii r_um, normalised to the centre value."""
    r = np.concatenate([[0.0], np.atleast_1d(np.asarray(r_um, float))])     # r = 0 first, for normalisation
    kmax = 12.0 / min(w_um, L_h_um) + 12.0 / w_um
    k = np.linspace(0, kmax, n_k)[1:]
    dk = k[1] - k[0]
    s = (1 + (k * L_h_um) ** 2) / tau_v
    base = k * np.exp(-(k * w_um) ** 2 / 8) / (1 + (k * L_h_um) ** 2)
    J = j0(np.outer(r, k))                                          # (nr, nk)
    t = np.concatenate([np.linspace(tau / n_t, tau, n_t), tau + np.geomspace(1e-3 * tau, 20 * max(tau, tau_v), n_t)])
    peak = np.zeros(r.size)
    for ti in t:
        F = -np.expm1(-s * ti) if ti <= tau else -np.expm1(-s * tau) * np.exp(-s * (ti - tau))
        peak = np.maximum(peak, (J * (base * F)[None, :]).sum(axis=1) * dk)
    return peak[1:] / peak[0]


def safe_temperature(Ea_eV, v_inf, t_exposure, n_writes, delta_m):
    """Highest neighbour temperature keeping glass-branch growth below delta_m after n_writes exposures:
    v_inf exp(-Ea/kT) t_exposure n_writes = delta_m."""
    kB = 8.617333262e-5
    return Ea_eV / (kB * np.log(v_inf * t_exposure * n_writes / delta_m))
