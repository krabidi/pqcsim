"""Focused Gaussian beam on a periodic pixel array: exact Bloch (array-scanning) superposition of RCWA solutions.

Incident tangential field at the top of the stack (x-polarised): E_x(r) = exp(-r^2 / w^2), i.e. 1/e^2 intensity
radius w. Its angular spectrum a(kappa) = (w^2 / 4 pi) exp(-kappa^2 w^2 / 4) gives
    E(r) = ∫_BZ dk Σ_G a(k + G) e^{i (k + G) . r} ,
so each Brillouin-zone point k is one oblique RCWA problem in which every propagating order G is incident
with amplitude a(k + G). The total field is the k-integral (midpoint rule over n_bz x n_bz points; the sampled
field is periodic over n_bz periods, which must exceed the beam size).
Normalisation (all in units of 2 Z0 x power):
    P_in   = (2 pi)^2 Σ_k dkappa_k  F_inc(k)       F_inc: unit-cell-averaged incident flux of the Bloch wave
    P_abs  = k0 Im(eps) ∫_pixel |E|^2 dV
Absorbed fraction in the pixel at lattice site R = P_abs(R) / P_in.

Accuracy: the field reconstruction inside high-contrast pixels converges slowly with the number of orders
(the field-integrated total is 9-22 % low at 3-7 orders), while the *distribution* among pixels converges
fast (shares stable to < 1 %). The returned values are therefore the field-based shares, computed over the
full supercell of n_bz x n_bz pixels, multiplied by the exact flux-based total absorbed in the PCM layer
(orthogonality of different k over the supercell makes that total exact).
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from .rcwa import RCWA, RCWALayer


def focused_pixel_absorption(lam_nm: float, period_um: float, layers: Sequence[RCWALayer], pcm_layer: int,
                             n_pixel: complex, side_um: float, w_um: float, sites: Sequence[Tuple[int, int]],
                             n_bz: int = 8, orders: int = 4, n_inc: complex = 1.0, n_sub: complex = 1.45,
                             n_xy: int = 10, n_z: int = 5, return_total: bool = False):
    lam = lam_nm * 1e-9
    k0 = 2 * np.pi / lam
    P = period_um
    G = lam_nm * 1e-3 / P                                   # reciprocal lattice step in k0 units
    kb = (np.arange(n_bz) + 0.5) / n_bz * G - G / 2         # BZ midpoints (k0 units)
    dk = (G / n_bz) ** 2                                    # k0-normalised area element
    dkappa = dk * k0 ** 2                                   # rad^2 / m^2
    w = w_um * 1e-6
    # sample points inside each requested pixel (cell-centred), and depths inside the PCM layer
    u = (np.arange(n_xy) + 0.5) / n_xy * side_um - side_um / 2
    UX, UY = np.meshgrid(u, u, indexing="ij")
    half = n_bz // 2
    cells = [(i, j) for i in range(-half, n_bz - half) for j in range(-half, n_bz - half)]   # one supercell
    for s_ in sites:
        if tuple(s_) not in cells:
            raise ValueError(f"site {s_} lies outside the supercell of {n_bz} x {n_bz} pixels")
    pts_x = np.concatenate([UX.ravel() + sx * P for sx, sy in cells])
    pts_y = np.concatenate([UY.ravel() + sy * P for sx, sy in cells])
    d_pcm = layers[pcm_layer].thickness_nm
    z = (np.arange(n_z) + 0.5) / n_z * d_pcm
    E_tot = np.zeros((n_z, 3, pts_x.size), complex)
    P_in = 0.0
    P_abs_flux = 0.0                                        # exact total absorbed in the PCM layer (all pixels)
    for kx0 in kb:
        for ky0 in kb:
            r = RCWA(lam_nm, (P, P), orders=orders, n_inc=n_inc, n_sub=n_sub, k_parallel=(kx0, ky0))
            kap = np.hypot(r.kx, r.ky)                       # k0 units, per order
            prop = kap < np.real(n_inc) * 0.999
            a = np.where(prop, (w ** 2 / (4 * np.pi)) * np.exp(-(kap * k0) ** 2 * w ** 2 / 4), 0.0)
            e_src = np.concatenate([a, np.zeros_like(a)]).astype(complex)          # x-polarised tangential field
            if not np.any(a > 1e-12 * w ** 2):
                continue
            E, inc = r.layer_fields(layers, pcm_layer, e_src, z, pts_x, pts_y)
            E_tot += dkappa * E
            P_in += (2 * np.pi) ** 2 * dkappa * inc
            P_abs_flux += (2 * np.pi) ** 2 * dkappa * r.last_layer_absorption[pcm_layer]
    im_eps = (complex(n_pixel) ** 2).imag
    dV = (side_um * 1e-6 / n_xy) ** 2 * (d_pcm * 1e-9 / n_z)
    I = (np.abs(E_tot) ** 2).sum(axis=1)                    # (n_z, npts)
    per = n_xy * n_xy
    raw = {c: float(k0 * im_eps * I[:, i * per:(i + 1) * per].sum() * dV / P_in) for i, c in enumerate(cells)}
    total_field = sum(raw.values())
    total_exact = float(P_abs_flux / P_in)
    out = {tuple(s_): raw[tuple(s_)] / total_field * total_exact for s_ in sites}
    if return_total:
        return out, total_exact, total_field
    return out


def focused_reflectance(lam_nm: float, period_um: float, layers: Sequence[RCWALayer], w_um: float,
                        n_bz: int = 4, orders: int = 4, n_inc: complex = 1.0, n_sub: complex = 1.45,
                        na_collect: float = None, na_illum: float = None) -> float:
    """Reflected fraction of the power of a focused Gaussian beam (x-polarised, 1/e^2 intensity radius w) on a
    periodic structure, by the same Bloch superposition as focused_pixel_absorption. Different Bloch waves are
    orthogonal over the plane, so powers add:
        R = sum_k F_inc(k) sum_{orders in NA} R_order(k) / sum_k F_inc(k).
    na_collect limits the reflected orders to |k_par| < na_collect * n_inc (the collecting objective); None takes
    every propagating order. For a meaningful readout of one feature per unit cell the period must exceed the
    beam size by a few w (the neighbouring copies then see only the beam's tail).
    na_illum truncates the incident angular spectrum at |k_par| < na_illum * n_inc (the illuminating objective;
    use na_illum = na_collect for one objective in reflection). None keeps every propagating order.
    With one objective the hard pupil edge is shared by the incident and the specular light, so the midpoint-rule
    edge error largely cancels: a uniform film matches the angular-spectrum TMM integral to < 0.3 % at a k-step
    of 0.12 k0, whereas without na_illum the error is up to 3 % and erratic in n_bz."""
    lam = lam_nm * 1e-9
    k0 = 2 * np.pi / lam
    G = lam_nm * 1e-3 / period_um
    kb = (np.arange(n_bz) + 0.5) / n_bz * G - G / 2
    n_r = np.real(n_inc)
    w = w_um * 1e-6
    P_in = P_r = 0.0
    for kx0 in kb:
        for ky0 in kb:
            r = RCWA(lam_nm, (period_um, period_um), orders=orders, n_inc=n_inc, n_sub=n_sub, k_parallel=(kx0, ky0))
            kap = np.hypot(r.kx, r.ky)
            prop = kap < np.real(n_inc) * 0.999
            a = np.where(prop, (w ** 2 / (4 * np.pi)) * np.exp(-(kap * k0) ** 2 * w ** 2 / 4), 0.0)
            if na_illum is not None:
                a = a * (kap < na_illum * n_r)
            if not np.any(a > 1e-12 * w ** 2):
                continue
            res = r.solve(layers, e_src=np.concatenate([a, np.zeros_like(a)]).astype(complex))
            inc = r.last_incident_flux
            Ro = res.R_orders.ravel()
            keep = prop if na_collect is None else prop & (kap < na_collect * n_r)
            P_in += inc
            P_r += inc * Ro[keep].sum()
    return float(P_r / P_in)


def naive_pixel_absorption(A_cell: float, fill_fraction: float, side_um: float, period_um: float, w_um: float,
                           sites: Sequence[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
    """Plane-wave approximation: per-pixel-area absorption A_cell / ff times the fraction of the
    beam power that falls on the pixel footprint."""
    from scipy.special import erf
    a = side_um / 2
    out = {}
    for sx, sy in sites:
        cx, cy = sx * period_um, sy * period_um
        fx = 0.5 * (erf(np.sqrt(2) * (cx + a) / w_um) - erf(np.sqrt(2) * (cx - a) / w_um))
        fy = 0.5 * (erf(np.sqrt(2) * (cy + a) / w_um) - erf(np.sqrt(2) * (cy - a) / w_um))
        out[(sx, sy)] = float(A_cell / fill_fraction * fx * fy)
    return out
