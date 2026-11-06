"""Waveguide modes: exact 1D slab TE modes and a 2D finite-difference semi-vectorial quasi-TE solver.

Convention e^{i(beta z - w t)}, n + ik: absorbing media give Im(n_eff) > 0 and power ~ e^{-2 k0 Im(n_eff) z}.

1D slab (exact): TE field E_y(x) in a stack of layers; the guided-mode condition is that the field
decays into both claddings, found by complex Newton iteration on the transfer-matrix determinant.

2D quasi-TE (dominant E_x, x horizontal, y vertical), semi-vectorial equation
    d/dx[ (1/eps) d(eps E)/dx ] + d^2 E/dy^2 + k0^2 eps E = beta^2 E
on a non-uniform node grid (fine where thin films are), E = 0 on the outer boundary. E_x is
tangential to horizontal films and continuous across them, which makes the quasi-TE approximation
accurate for thin horizontal absorbers (e.g. a PCM film on top of a waveguide). Eigenpairs by
shift-invert Arnoldi around (k0 n_guess)^2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs


# ----------------------------------------------------------------------------- 1D exact slab TE
def slab_te_neff(lam_nm: float, n_layers: Sequence[complex], d_nm: Sequence[float], n_top: complex,
                 n_bottom: complex, n_guess: complex, tol: float = 1e-13) -> complex:
    """Exact TE guided mode of top | layers | bottom (layers listed top to bottom)."""
    k0 = 2 * np.pi / (lam_nm * 1e-9)

    def char(neff):
        # transfer (E, dE/dx) down through the layers and require pure decay e^{-g x} below
        def kx(n):
            g = np.sqrt(complex(neff) ** 2 - complex(n) ** 2 + 0j) * k0
            return g if g.real >= 0 else -g
        g_top = kx(n_top)
        # x points downward; above the stack E ~ e^{+g x} decays upward, so dE/dx = +g E at the top
        E, dE = 1.0 + 0j, g_top * (1.0 + 0j)
        for n, d in zip(n_layers, d_nm):
            q = np.sqrt(complex(n) ** 2 - complex(neff) ** 2 + 0j) * k0
            L = d * 1e-9
            if abs(q) < 1e-30:
                E, dE = E + dE * L, dE
            else:
                c, s = np.cos(q * L), np.sin(q * L)
                E, dE = E * c + dE * s / q, -E * q * s + dE * c
        g_bot = kx(n_bottom)
        return dE + g_bot * E                               # decaying below: dE/dx = -g E

    x = complex(n_guess)
    for _ in range(100):
        f = char(x)
        h = 1e-7 * max(1.0, abs(x))
        df = (char(x + h) - char(x - h)) / (2 * h)
        step = f / df
        x -= step
        if abs(step) < tol * abs(x):
            return x
    raise RuntimeError("slab mode Newton iteration did not converge")


# ----------------------------------------------------------------------------- 2D quasi-TE FD
def graded_axis(breakpoints_nm: Sequence[float], fine_nm: Sequence[float], max_ratio: float = 1.15) -> np.ndarray:
    """Node coordinates (nm) with target spacing fine_nm[i] in [breakpoints[i], breakpoints[i+1]].

    Breakpoints (material interfaces) are always nodes. Inside each region the spacing grows
    geometrically (ratio <= max_ratio) away from neighbouring finer regions, so adjacent cells never
    differ by more than max_ratio: abrupt spacing jumps make the 3-point second difference only
    first-order accurate."""
    bp = np.asarray(breakpoints_nm, float)
    h = np.asarray(fine_nm, float)
    pts = [bp[0]]
    for i, (a, b) in enumerate(zip(bp[:-1], bp[1:])):
        h_left = min(h[i], h[i - 1]) if i > 0 else h[i]
        h_right = min(h[i], h[i + 1]) if i + 1 < h.size else h[i]
        # spacing profile: start at h_left, grow towards h[i], shrink to h_right near b
        steps, x = [], a
        while True:
            d_left = x - a
            d_right = b - x
            grow_l = h_left * max_ratio ** (np.log1p(d_left * (max_ratio - 1) / h_left) / np.log(max_ratio))
            grow_r = h_right * max_ratio ** (np.log1p(d_right * (max_ratio - 1) / h_right) / np.log(max_ratio))
            dx = min(h[i], grow_l, grow_r)
            if x + dx >= b - 0.5 * dx:
                break
            steps.append(dx)
            x += dx
        n = len(steps) + 1
        # rescale the profile so it ends exactly on b
        prof = np.array(steps + [max(b - x, 1e-12)])
        prof *= (b - a) / prof.sum()
        pts += list(a + np.cumsum(prof))
    pts = np.asarray(pts)
    pts[-1] = bp[-1]
    return pts


@dataclass
class Mode:
    neff: complex
    field: np.ndarray                    # E_x on the (nx, ny) node grid
    x_nm: np.ndarray
    y_nm: np.ndarray

    def loss_dB_per_um(self, lam_nm: float) -> float:
        k0 = 2 * np.pi / (lam_nm * 1e-9)
        return float(10 * np.log10(np.e) * 2 * k0 * self.neff.imag * 1e-6)


def quasi_te_modes(lam_nm: float, x_nm: np.ndarray, y_nm: np.ndarray,
                   index: Callable[[np.ndarray, np.ndarray], np.ndarray], n_guess: float,
                   n_modes: int = 1) -> List[Mode]:
    """Quasi-TE modes. index(X, Y) returns the complex refractive index at node coordinates (nm).

    Node permittivity is the average over the node's dual cell (sampled 4x4), so a thin film between
    grid lines is represented by its area fraction."""
    k0 = 2 * np.pi / (lam_nm * 1e-9)
    x, y = np.asarray(x_nm, float) * 1e-9, np.asarray(y_nm, float) * 1e-9
    nx, ny = x.size, y.size
    # dual-cell averaged permittivity
    xe = np.concatenate([[x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]]])
    ye = np.concatenate([[y[0]], 0.5 * (y[1:] + y[:-1]), [y[-1]]])
    s = (np.arange(4) + 0.5) / 4
    eps = np.zeros((nx, ny), complex)
    for a in s:
        for b in s:
            X = (xe[:-1] + a * np.diff(xe))[:, None] * np.ones((1, ny))
            Y = np.ones((nx, 1)) * (ye[:-1] + b * np.diff(ye))[None, :]
            eps += np.asarray(index(X * 1e9, Y * 1e9), complex) ** 2
    eps /= 16
    # interior unknowns (E = 0 on the boundary)
    ii, jj = np.meshgrid(np.arange(1, nx - 1), np.arange(1, ny - 1), indexing="ij")
    idx = -np.ones((nx, ny), int)
    idx[1:-1, 1:-1] = np.arange(ii.size).reshape(ii.shape)
    hxm, hxp = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    hym, hyp = y[1:-1] - y[:-2], y[2:] - y[1:-1]
    rows, cols, vals = [], [], []

    def add(r, c, v):
        m = c >= 0
        rows.append(r[m]); cols.append(c[m]); vals.append(v[m])

    I = idx[1:-1, 1:-1]
    e = eps
    em, ep = 0.5 * (e[:-2, 1:-1] + e[1:-1, 1:-1]), 0.5 * (e[2:, 1:-1] + e[1:-1, 1:-1])
    ax = 2.0 / (hxm + hxp)[:, None]
    cxp = ax / (hxp[:, None] * ep) * e[2:, 1:-1]          # coefficient of E_{i+1}
    cxm = ax / (hxm[:, None] * em) * e[:-2, 1:-1]          # coefficient of E_{i-1}
    cx0 = -ax * e[1:-1, 1:-1] * (1 / (hxp[:, None] * ep) + 1 / (hxm[:, None] * em))
    ay = 2.0 / (hym + hyp)[None, :]
    cyp, cym = ay / hyp[None, :], ay / hym[None, :]
    cy0 = -(cyp + cym)
    diag = cx0 + cy0 + k0 ** 2 * e[1:-1, 1:-1]
    add(I.ravel(), I.ravel(), diag.ravel())
    add(I.ravel(), idx[2:, 1:-1].ravel(), cxp.ravel())
    add(I.ravel(), idx[:-2, 1:-1].ravel(), cxm.ravel())
    add(I.ravel(), idx[1:-1, 2:].ravel(), np.broadcast_to(cyp, I.shape).ravel())
    add(I.ravel(), idx[1:-1, :-2].ravel(), np.broadcast_to(cym, I.shape).ravel())
    N = ii.size
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsc()
    w, v = eigs(A, k=n_modes, sigma=(k0 * n_guess) ** 2, which="LM")
    out = []
    for k in np.argsort(-w.real):
        neff = np.sqrt(w[k] + 0j) / k0
        if neff.real < 0:
            neff = -neff
        F = np.zeros((nx, ny), complex)
        F[1:-1, 1:-1] = v[:, k].reshape(ii.shape)
        out.append(Mode(complex(neff), F, np.asarray(x_nm), np.asarray(y_nm)))
    return out
