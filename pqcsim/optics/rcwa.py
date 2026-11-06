"""Rigorous coupled-wave analysis (RCWA) for 2D-periodic multilayers at normal incidence.

Convention e^{-i w t}, n + ik. With the normalised field H~ = i Z0 H and z' = k0 z the curl equations
are curl E = H~ and curl H~ = eps E; for Fourier orders with Kx = kx/k0, Ky = ky/k0:
    d/dz [Ex, Ey] = P [H~x, H~y],   P = [[Kx E^-1 Ky, I - Kx E^-1 Kx], [Ky E^-1 Ky - I, -Ky E^-1 Kx]]
    d/dz [H~x, H~y] = Q [Ex, Ey],   Q = [[Kx Ky, E - Kx^2], [Ky^2 - E, -Ky Kx]]
(E: Fourier convolution matrix of eps). With factorization="li" (default) the eps acting on E_x / E_y in Q
uses Li's rules for rectangular inclusions (inverse rule across the walls the component is normal to,
Laurent along the other direction; Li, JOSA A 14, 2758 (1997)); E^-1 in P keeps the Laurent rule. Modes: E = W (e^{-lam z} c+ + e^{lam z} c-),
H~ = V (-e^{-lam z} c+ + e^{lam z} c-), lam^2 = eig(P Q), V = Q W lam^-1, Re(lam) >= 0 and, for
propagating modes, Im(lam) < 0 so that e^{-lam z} = e^{+i kz z} travels towards +z.
Layers are combined with Redheffer star products of scattering matrices referenced to a zero-thickness
gap medium (results are independent of its permittivity; checked in tests). Absorption per layer is
the difference of the z Poynting flux, S_z ∝ Re[i (Ex H~y* - Ey H~x*)] summed over orders, between
the layer's two interfaces, evaluated in the gap with the interior amplitudes from partial S-matrices.

Approximations: plane-wave illumination of an infinite periodic array; truncated Fourier orders (check
convergence with the number of orders).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

Pattern = Union[complex, float, np.ndarray]   # uniform index, or eps map on an (Nx, Ny) grid over one cell


@dataclass
class RCWALayer:
    thickness_nm: float
    eps: Pattern                               # relative permittivity: scalar or 2D map over the unit cell

    @staticmethod
    def uniform(thickness_nm: float, n: complex) -> "RCWALayer":
        return RCWALayer(thickness_nm, complex(n) ** 2)


def pixel_eps(n_pixel: complex, n_fill: complex, period_um: float, side_um: float, grid: int = 256) -> np.ndarray:
    """Permittivity map of one unit cell with a square pixel of side side_um centred on the origin.
    Grid index i is at x = i P / grid (the FFT convention used by the solver), so the pixel occupies the
    cell corners of the array index space; in real space it is centred at x = y = 0."""
    x = np.arange(grid) / grid * period_um
    x = np.abs((x + period_um / 2) % period_um - period_um / 2)        # distance to the origin (periodic)
    inside = (x[:, None] < side_um / 2) & (x[None, :] < side_um / 2)
    return np.where(inside, complex(n_pixel) ** 2, complex(n_fill) ** 2)


@dataclass
class RCWAResult:
    R: float
    T: float
    A_layers: np.ndarray
    R_orders: np.ndarray
    T_orders: np.ndarray


class RCWA:
    def __init__(self, lam_nm: float, period_um: Tuple[float, float], orders: int = 5,
                 n_inc: complex = 1.0, n_sub: complex = 1.45, eps_gap: Optional[float] = None,
                 factorization: str = "li", k_parallel: Tuple[float, float] = (0.0, 0.0)):
        """k_parallel: in-plane wavevector of the zeroth order, normalised to k0 (oblique / Bloch incidence)."""
        if factorization not in ("li", "laurent"):
            raise ValueError("factorization must be 'li' or 'laurent'")
        self.factorization = factorization
        self.lam = lam_nm * 1e-9
        self.Lx, self.Ly = period_um[0] * 1e-6, period_um[1] * 1e-6
        self.M = orders
        m = np.arange(-orders, orders + 1)
        P, Qq = np.meshgrid(m, m, indexing="ij")
        self.p, self.q = P.ravel(), Qq.ravel()
        self.N = self.p.size
        self.k_parallel = (float(k_parallel[0]), float(k_parallel[1]))
        kx = self.k_parallel[0] + self.p * self.lam / self.Lx     # normalised (k0 units) wavevector of each order
        ky = self.k_parallel[1] + self.q * self.lam / self.Ly
        self.Kx, self.Ky = np.diag(kx).astype(complex), np.diag(ky).astype(complex)
        self.kx, self.ky = kx, ky
        self.eps_inc, self.eps_sub = complex(n_inc) ** 2, complex(n_sub) ** 2
        if abs(self.eps_inc.imag) > 1e-12:
            raise ValueError("incidence medium must be lossless")
        # gap medium: real permittivity large enough that every retained order propagates (no singular V0)
        self.eps_gap = float(1.0 + np.max(kx ** 2 + ky ** 2) + 1.0) if eps_gap is None else float(eps_gap)
        self.W0, self.V0 = self._homogeneous_modes(self.eps_gap)[:2]

    # ------------------------------------------------------------------ building blocks
    def _conv(self, eps: Pattern) -> np.ndarray:
        if np.isscalar(eps) or np.ndim(eps) == 0:
            return complex(eps) * np.eye(self.N, dtype=complex)
        F = np.fft.fft2(np.asarray(eps, complex)) / np.size(eps)
        dp = self.p[:, None] - self.p[None, :]
        dq = self.q[:, None] - self.q[None, :]
        if np.max(np.abs(dp)) >= eps.shape[0] // 2 or np.max(np.abs(dq)) >= eps.shape[1] // 2:
            raise ValueError("pattern grid too coarse for the requested number of orders")
        return F[dp % eps.shape[0], dq % eps.shape[1]]

    def _li_matrix(self, eps: np.ndarray, axis: int) -> np.ndarray:
        """Li's rule for the field component normal to walls perpendicular to `axis` (0: x, 1: y):
        inverse rule along `axis` (invert the Toeplitz matrix of 1/eps), Laurent rule along the other."""
        M = self.M
        e = np.asarray(eps, complex) if axis == 0 else np.asarray(eps, complex).T
        n_a, n_b = e.shape
        orders = np.arange(-M, M + 1)
        d = orders[:, None] - orders[None, :]
        if 2 * M >= n_a // 2 or 2 * M >= n_b // 2:
            raise ValueError("pattern grid too coarse for the requested number of orders")
        inv_f = np.fft.fft(1.0 / e, axis=0) / n_a                       # along `axis`, for every line
        A = np.linalg.inv(inv_f[d % n_a, :].transpose(2, 0, 1))          # (n_b, 2M+1, 2M+1)
        Ahat = np.fft.fft(A, axis=0) / n_b                               # along the other axis
        # assemble [(p, q), (p', q')] with p along `axis`, q along the other
        P, Q = np.meshgrid(orders, orders, indexing="ij")
        P, Q = P.ravel(), Q.ravel()
        ip = P + M
        dq = (Q[:, None] - Q[None, :]) % n_b
        out = Ahat[dq, ip[:, None], ip[None, :]]
        if axis == 1:
            # reorder from (q-major along y) back to the solver's (p over x, q over y) ordering
            perm = np.array([np.where((P == qq) & (Q == pp))[0][0] for pp, qq in zip(self.p, self.q)])
            out = out[np.ix_(perm, perm)]
        return out

    def _homogeneous_modes(self, eps: complex):
        kz = np.sqrt(complex(eps) - self.kx ** 2 - self.ky ** 2 + 0j)
        kz = np.where(kz.imag < 0, -kz, kz)
        kz = np.where((np.abs(kz.imag) < 1e-14) & (kz.real < 0), -kz, kz)
        lam = np.concatenate([-1j * kz, -1j * kz])                       # e^{-lam z} = e^{i kz z}
        I = np.eye(self.N)
        Kx, Ky = self.Kx, self.Ky
        Q = np.block([[Kx @ Ky, eps * I - Kx @ Kx], [Ky @ Ky - eps * I, -Ky @ Kx]])
        W = np.eye(2 * self.N, dtype=complex)
        V = Q / lam[None, :]
        return W, V, lam, kz

    def _layer_modes(self, eps: Pattern):
        if np.isscalar(eps) or np.ndim(eps) == 0:
            W, V, lam, _ = self._homogeneous_modes(complex(eps))
            return W, V, lam
        E = self._conv(eps)
        Ei = np.linalg.inv(E)                                   # E_z: tangential to both walls -> Laurent
        if self.factorization == "li":
            Exx, Eyy = self._li_matrix(eps, 0), self._li_matrix(eps, 1)
        else:
            Exx = Eyy = E
        I = np.eye(self.N)
        Kx, Ky = self.Kx, self.Ky
        P = np.block([[Kx @ Ei @ Ky, I - Kx @ Ei @ Kx], [Ky @ Ei @ Ky - I, -Ky @ Ei @ Kx]])
        # d/dz H~x = Eyy Ey - Kx^2 Ey + Kx Ky Ex ;  d/dz H~y = Ky^2 Ex - Ky Kx Ey - Exx Ex
        Q = np.block([[Kx @ Ky, Eyy - Kx @ Kx], [Ky @ Ky - Exx, -Ky @ Kx]])
        lam2, W = np.linalg.eig(P @ Q)
        lam = np.sqrt(lam2.astype(complex))
        flip = (lam.real < 0) | ((np.abs(lam.real) < 1e-12) & (lam.imag > 0))
        lam = np.where(flip, -lam, lam)
        V = Q @ W / lam[None, :]
        return W, V, lam

    def _S_layer(self, eps: Pattern, d_m: float):
        W, V, lam = self._layer_modes(eps)
        Wi, Vi = np.linalg.inv(W), np.linalg.inv(V)
        A = Wi @ self.W0 + Vi @ self.V0
        B = Wi @ self.W0 - Vi @ self.V0
        X = np.diag(np.exp(-lam * 2 * np.pi / self.lam * d_m))
        Ai = np.linalg.inv(A)
        D = np.linalg.inv(A - X @ B @ Ai @ X @ B)
        S11 = D @ (X @ B @ Ai @ X @ A - B)
        S12 = D @ X @ (A - B @ Ai @ B)
        return S11, S12, S12, S11

    def _S_ref(self):
        W, V, _, _ = self._homogeneous_modes(self.eps_inc)
        A = np.linalg.solve(self.W0, W) + np.linalg.solve(self.V0, V)
        B = np.linalg.solve(self.W0, W) - np.linalg.solve(self.V0, V)
        Ai = np.linalg.inv(A)
        return -Ai @ B, 2 * Ai, 0.5 * (A - B @ Ai @ B), B @ Ai

    def _S_trn(self):
        W, V, _, _ = self._homogeneous_modes(self.eps_sub)
        A = np.linalg.solve(self.W0, W) + np.linalg.solve(self.V0, V)
        B = np.linalg.solve(self.W0, W) - np.linalg.solve(self.V0, V)
        Ai = np.linalg.inv(A)
        return B @ Ai, 0.5 * (A - B @ Ai @ B), 2 * Ai, -Ai @ B

    @staticmethod
    def _star(SA, SB):
        A11, A12, A21, A22 = SA
        B11, B12, B21, B22 = SB
        I = np.eye(A11.shape[0])
        D = A12 @ np.linalg.inv(I - B11 @ A22)
        F = B21 @ np.linalg.inv(I - A22 @ B11)
        return A11 + D @ B11 @ A21, D @ B12, F @ A21, B22 + F @ A22 @ B12

    def _identity_S(self):
        Z, I = np.zeros((2 * self.N, 2 * self.N), complex), np.eye(2 * self.N, dtype=complex)
        return Z, I, I, Z

    # ------------------------------------------------------------------ solve
    def _flux(self, a_plus, a_minus, W=None, V=None):
        """Normalised z flux for forward / backward mode amplitudes (gap medium unless W, V are given)."""
        W = self.W0 if W is None else W
        V = self.V0 if V is None else V
        E = W @ (a_plus + a_minus)
        H = V @ (-a_plus + a_minus)
        n = self.N
        Ex, Ey, Hx, Hy = E[:n], E[n:], H[:n], H[n:]
        return float(np.real(1j * (Ex @ np.conj(Hy) - Ey @ np.conj(Hx))))

    def solve(self, layers: Sequence[RCWALayer], pol: Tuple[complex, complex] = (1.0, 0.0),
              e_src: Optional[np.ndarray] = None) -> RCWAResult:
        """Incidence from the incidence medium. Either a single zeroth-order plane wave with tangential
        polarisation pol = (Ex, Ey), or a general set of incident orders given by their tangential fields
        e_src (length 2N: Ex of all orders, then Ey). Efficiencies are normalised to the incident flux."""
        core = self._solve_core(layers, pol, e_src)
        n = self.N
        W_ref, V_ref, _, kz_ref = self._homogeneous_modes(self.eps_inc)
        W_trn, V_trn, _, kz_trn = self._homogeneous_modes(self.eps_sub)
        S, c_src = core["S"], core["c_src"]
        e_ref = W_ref @ (S[0] @ c_src)
        e_trn = W_trn @ (S[2] @ c_src)

        def efficiencies(e, kz):
            Ex, Ey = e[:n], e[n:]
            Ez = -(self.kx * Ex + self.ky * Ey) / kz          # transverse field: k . E = 0
            return (np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2) * np.real(kz)

        inc_eff = efficiencies(W_ref @ c_src, kz_ref).sum()
        self.last_incident_flux = float(inc_eff)              # unnormalised, comparable across k_parallel
        R_orders = efficiencies(e_ref, kz_ref) / inc_eff
        T_orders = efficiencies(e_trn, kz_trn) / inc_eff
        flux = np.array(core["flux"]) / core["inc"]
        A_layers = flux[:-1] - flux[1:]
        return RCWAResult(R=float(R_orders.sum()), T=float(T_orders.sum()), A_layers=A_layers,
                          R_orders=R_orders.reshape(2 * self.M + 1, -1), T_orders=T_orders.reshape(2 * self.M + 1, -1))

    def _solve_core(self, layers, pol=(1.0, 0.0), e_src=None):
        n = self.N
        if e_src is None:
            i0 = int(np.where((self.p == 0) & (self.q == 0))[0][0])
            e_src = np.zeros(2 * n, complex)
            e_src[i0], e_src[n + i0] = pol
            e_src /= np.linalg.norm([pol[0], pol[1]])
        W_ref, V_ref, _, _ = self._homogeneous_modes(self.eps_inc)
        c_src = np.linalg.solve(W_ref, np.asarray(e_src, complex))
        S_layers = [self._S_layer(L.eps, L.thickness_nm * 1e-9) for L in layers]
        S_ref, S_trn = self._S_ref(), self._S_trn()
        S = S_ref
        for SL in S_layers:
            S = self._star(S, SL)
        S = self._star(S, S_trn)
        # interior gap amplitudes at every interface (above layer j, j = 0..len(layers))
        S_left = [S_ref]
        for SL in S_layers:
            S_left.append(self._star(S_left[-1], SL))
        S_right = [S_trn]
        for SL in reversed(S_layers):
            S_right.append(self._star(SL, S_right[-1]))
        S_right = S_right[::-1]                                 # S_right[j]: layers j.. plus substrate
        I = np.eye(2 * n)
        flux, gaps = [], []
        for j in range(len(layers) + 1):
            L11, L12, L21, L22 = S_left[j]
            R11 = S_right[j][0]
            a_plus = np.linalg.solve(I - L22 @ R11, L21 @ c_src)
            a_minus = R11 @ a_plus
            gaps.append((a_plus, a_minus))
            flux.append(self._flux(a_plus, a_minus))
        inc = self._flux(c_src, np.zeros_like(c_src), W_ref, V_ref)     # incident wave in the incidence medium
        return {"S": S, "c_src": c_src, "flux": flux, "gaps": gaps, "inc": inc}

    def layer_fields(self, layers: Sequence[RCWALayer], j: int, e_src: np.ndarray, z_nm: Sequence[float],
                     x_um: np.ndarray, y_um: np.ndarray):
        """Complex vector field (Ex, Ey, Ez) inside layer j at depths z_nm (from the layer top) and in-plane
        points (x_um, y_um), for incident tangential fields e_src. Also returns the incident flux (2 Z0 S_z,
        unit-cell average) for normalisation."""
        core = self._solve_core(layers, e_src=e_src)
        a_plus, a_minus = core["gaps"][j]
        L = layers[j]
        W, V, lam = self._layer_modes(L.eps)
        k0 = 2 * np.pi / self.lam
        d = L.thickness_nm * 1e-9 * k0
        X = np.exp(-lam * d)
        # continuity at the layer top: W(c+ + X c-) = W0(a+ + a-),  V(-c+ + X c-) = V0(-a+ + a-)
        n2 = 2 * self.N
        A = np.block([[W, W * X[None, :]], [-V, V * X[None, :]]])
        rhs = np.concatenate([self.W0 @ (a_plus + a_minus), self.V0 @ (-a_plus + a_minus)])
        cc = np.linalg.solve(A, rhs)
        cp, cm = cc[:n2], cc[n2:]
        uniform = np.isscalar(L.eps) or np.ndim(L.eps) == 0
        if uniform:
            Einv = np.eye(self.N) / complex(L.eps)
        else:
            Einv = np.linalg.inv(self._conv(L.eps))
            # D_x, D_y are continuous across the walls they are normal to: synthesise D in real space and
            # divide by the local eps (Li-consistent; avoids Gibbs overshoot of the discontinuous E_x, E_y)
            Exx = self._li_matrix(L.eps, 0) if self.factorization == "li" else self._conv(L.eps)
            Eyy = self._li_matrix(L.eps, 1) if self.factorization == "li" else self._conv(L.eps)
            g = np.asarray(L.eps).shape
            xi = np.rint((np.asarray(x_um, float) / (self.Lx * 1e6)) % 1.0 * g[0]).astype(int) % g[0]
            yi = np.rint((np.asarray(y_um, float) / (self.Ly * 1e6)) % 1.0 * g[1]).astype(int) % g[1]
            eps_local = np.asarray(L.eps)[xi, yi]
        x = np.asarray(x_um, float) * 1e-6 * k0
        y = np.asarray(y_um, float) * 1e-6 * k0
        phase = np.exp(1j * (np.outer(x, self.kx) + np.outer(y, self.ky)))      # (npts, N)
        out = []
        for zz in z_nm:
            z = zz * 1e-9 * k0
            ez = np.exp(-lam * z)
            ezm = np.exp(lam * (z - d))
            Et = W @ (ez * cp + ezm * cm)
            Ht = V @ (-ez * cp + ezm * cm)
            n = self.N
            Ex, Ey, Hx, Hy = Et[:n], Et[n:], Ht[:n], Ht[n:]
            Ez = Einv @ (1j * (self.kx * Hy - self.ky * Hx))
            if uniform:
                out.append(np.stack([phase @ Ex, phase @ Ey, phase @ Ez]))
                self.last_layer_absorption = np.array(core["flux"][:-1]) - np.array(core["flux"][1:])
            else:
                out.append(np.stack([(phase @ (Exx @ Ex)) / eps_local, (phase @ (Eyy @ Ey)) / eps_local, phase @ Ez]))
        self.last_layer_absorption = np.array(core["flux"][:-1]) - np.array(core["flux"][1:])   # unnormalised
        return np.array(out), core["inc"]

    def solve_unpolarised(self, layers: Sequence[RCWALayer]) -> RCWAResult:
        a, b = self.solve(layers, (1.0, 0.0)), self.solve(layers, (0.0, 1.0))
        return RCWAResult(0.5 * (a.R + b.R), 0.5 * (a.T + b.T), 0.5 * (a.A_layers + b.A_layers),
                          0.5 * (a.R_orders + b.R_orders), 0.5 * (a.T_orders + b.T_orders))
