"""Nonlinear transient heat conduction: latent heat of melting and temperature-dependent properties.

Enthalpy formulation on the axisymmetric (r, z) finite-volume grid of StackHeatSolver:
    dH_i/dt = (1/V_i) [ sum_j G_ij(T) (T_j - T_i) - G_i,b(T) (T_i - T0) ] + q_i(t)
    H(T)    = rho cp (T - T0) + rho L f(T),     f(T) = 1 / (1 + exp(-(T - T_m) / w))   (liquid fraction)
    G_ij(T) = A_ij / (d_i / (2 k_i(T_i)) + d_j / (2 k_j(T_j)))
Integrating H (not T) makes energy conservation exact through the melting front (Radau time
stepping, analytic sparse Jacobian); T is recovered
from H by safeguarded Newton (H is strictly increasing in T). The melting range is smoothed over
a width w (K); results converge as w -> 0 (checked against the Stefan solution in tests).
Layer properties may depend on temperature and phase: k = k_solid + f (k_liquid - k_solid),
optionally times (1 + beta (T - T0)).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import spsolve

from .diffusion import StackHeatSolver, ThermalLayer, ThermalResult


@dataclass(frozen=True)
class PhaseChangeLayer:
    name: str
    thickness_nm: float
    k: float                        # solid thermal conductivity, W/(m K)
    rho: float
    cp: float
    cells: int = 4
    k_liquid: Optional[float] = None
    latent_J_per_kg: float = 0.0
    T_melt: Optional[float] = None
    melt_width_K: float = 5.0
    k_beta_per_K: float = 0.0       # linear temperature coefficient of k
    tbr_below: float = 0.0          # thermal boundary resistance to the next layer down, m^2 K / W

    def __post_init__(self):
        if self.latent_J_per_kg < 0 or self.melt_width_K <= 0:
            raise ValueError("latent heat must be >= 0 and melt width > 0")
        if self.latent_J_per_kg > 0 and self.T_melt is None:
            raise ValueError(f"layer {self.name}: latent heat needs T_melt")

    def linear(self) -> ThermalLayer:
        return ThermalLayer(self.name, self.thickness_nm, self.k, self.rho, self.cp, self.cells, self.tbr_below)


class NonlinearStackHeatSolver:
    def __init__(self, layers: Sequence[PhaseChangeLayer], r_faces_m: np.ndarray, T0: float = 300.0,
                 top: str = "insulated", bottom: str = "fixed", outer: str = "fixed"):
        self.layers = list(layers)
        self.T0 = T0
        self.geo = StackHeatSolver([L.linear() for L in self.layers], r_faces_m, top, bottom, outer)
        g = self.geo
        self.nz, self.nr = g.nz, g.nr
        n = self.nz * self.nr
        lay = np.repeat(g.layer_of_row, self.nr)
        P = lambda attr, default=0.0: np.array([getattr(self.layers[l], attr) if getattr(self.layers[l], attr) is not None
                                                else default for l in lay], float)
        self.rho, self.cp = P("rho"), P("cp")
        self.k_s = P("k")
        self.k_l = np.array([self.layers[l].k_liquid if self.layers[l].k_liquid is not None else self.layers[l].k
                             for l in lay], float)
        self.L = P("latent_J_per_kg")
        self.Tm = P("T_melt", 1e30)
        self.wm = P("melt_width_K", 1.0)
        self.beta = P("k_beta_per_K")
        self.V = g.V.ravel()
        self._faces(top, bottom, outer)
        self.n = n

    # ------------------------------------------------------------------ geometry of faces
    def _faces(self, top, bottom, outer):
        g = self.geo
        idx = np.arange(self.nz * self.nr).reshape(self.nz, self.nr)
        dr = np.diff(g.rf)
        ring = np.pi * (g.rf[1:] ** 2 - g.rf[:-1] ** 2)
        a, b, A, da, db, rb = [], [], [], [], [], []
        for iz in range(self.nz):                               # radial faces
            for ir in range(self.nr - 1):
                a.append(idx[iz, ir]); b.append(idx[iz, ir + 1])
                A.append(2 * np.pi * g.rf[ir + 1] * g.dz[iz]); da.append(dr[ir] / 2); db.append(dr[ir + 1] / 2)
                rb.append(0.0)
        for iz in range(self.nz - 1):                           # axial faces
            for ir in range(self.nr):
                a.append(idx[iz, ir]); b.append(idx[iz + 1, ir])
                A.append(ring[ir]); da.append(g.dz[iz] / 2); db.append(g.dz[iz + 1] / 2)
                rb.append(g.tbr_row[iz])                       # boundary resistance, m^2 K / W
        self.fa, self.fb = np.array(a), np.array(b)
        self.fA, self.fda, self.fdb = np.array(A), np.array(da), np.array(db)
        self.fRb = np.array(rb)                               # in series with d/(2k) on each face
        bc, bA, bd = [], [], []                                 # Dirichlet boundary faces (T = T0)
        if outer == "fixed":
            for iz in range(self.nz):
                bc.append(idx[iz, -1]); bA.append(2 * np.pi * g.rf[-1] * g.dz[iz]); bd.append(dr[-1] / 2)
        for iz, flag in ((0, top), (self.nz - 1, bottom)):
            if flag == "fixed":
                for ir in range(self.nr):
                    bc.append(idx[iz, ir]); bA.append(ring[ir]); bd.append(g.dz[iz] / 2)
        self.bc, self.bA, self.bd = np.array(bc, int), np.array(bA), np.array(bd)
        n = self.nz * self.nr
        rows = np.concatenate([self.fa, self.fb, self.fa, self.fb, np.arange(n)])
        cols = np.concatenate([self.fb, self.fa, self.fa, self.fb, np.arange(n)])
        self.sparsity = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()

    # ------------------------------------------------------------------ constitutive relations
    def liquid_fraction(self, T):
        return expit((T - self.Tm) / self.wm)                    # overflow-free logistic

    def enthalpy(self, T):
        return self.rho * self.cp * (T - self.T0) + self.rho * self.L * self.liquid_fraction(T)

    def _dH_dT(self, T):
        f = self.liquid_fraction(T)
        return self.rho * self.cp + self.rho * self.L * f * (1 - f) / self.wm

    def temperature(self, H, tol_K: float = 1e-11):
        """Invert H(T): Newton steps safeguarded by a bisection bracket, warm-started from the last
        result (successive calls differ little). Converged when every cell's step is below tol_K."""
        c = self.rho * self.cp
        exact = self.L == 0
        if np.all(exact):
            return self.T0 + H / c                                   # no latent heat: closed form
        lo = self.T0 + (H - self.rho * self.L) / c - 1e-9          # H(T) in [c(T-T0), c(T-T0) + rho L]
        hi = self.T0 + H / c + 1e-9
        prev = getattr(self, "_T_last", None)
        if prev is not None and prev.shape == H.shape:
            T = np.clip(prev, lo, hi)
        else:
            Ts, Tl = self.T0 + H / c, self.T0 + (H - self.rho * self.L) / c
            a, b = self.Tm - 4 * self.wm, self.Tm + 4 * self.wm
            Ha, Hb = self.enthalpy(a), self.enthalpy(b)
            Tmid = a + (H - Ha) * (b - a) / np.where(Hb > Ha, Hb - Ha, 1.0)
            T = np.clip(np.where(H <= Ha, Ts, np.where(H >= Hb, Tl, Tmid)), lo, hi)
        for _ in range(100):
            F = self.enthalpy(T) - H
            lo = np.where(F < 0, T, lo)
            hi = np.where(F > 0, T, hi)
            step = F / self._dH_dT(T)
            Tn = T - step
            bad = (Tn <= lo) | (Tn >= hi)
            Tn = np.where(bad, 0.5 * (lo + hi), Tn)
            done = np.max(np.abs(Tn - T)) < tol_K
            T = Tn
            if done:
                break
        T = np.where(exact, self.T0 + H / c, T)
        self._T_last = T
        return T

    def conductivity(self, T):
        f = self.liquid_fraction(T)
        return (self.k_s + f * (self.k_l - self.k_s)) * (1.0 + self.beta * (T - self.T0))

    def _dk_dT(self, T):
        f = self.liquid_fraction(T)
        base = self.k_s + f * (self.k_l - self.k_s)
        dbase = (self.k_l - self.k_s) * f * (1 - f) / self.wm
        return dbase * (1.0 + self.beta * (T - self.T0)) + base * self.beta

    def _jac(self, H, T=None):
        """Analytic Jacobian d(rhs)/dH = V^-1 d(net)/dT diag(dT/dH). T = T(H) may be passed to skip inversion."""
        T = self.temperature(H) if T is None else T
        k, dk = self.conductivity(T), self._dk_dT(T)
        a, b = self.fa, self.fb
        den = self.fda / k[a] + self.fdb / k[b] + self.fRb
        G = self.fA / den
        dG_da = G * G / self.fA * self.fda / k[a] ** 2 * dk[a]
        dG_db = G * G / self.fA * self.fdb / k[b] ** 2 * dk[b]
        dT = T[b] - T[a]
        dFa, dFb = -G + dT * dG_da, G + dT * dG_db             # d(flux into a)/dT_a, /dT_b
        rows = [a, a, b, b]
        cols = [a, b, a, b]
        vals = [dFa, dFb, -dFa, -dFb]
        if self.bc.size:
            c = self.bc
            Gb = self.bA * k[c] / self.bd
            rows.append(c); cols.append(c)
            vals.append(-Gb - (T[c] - self.T0) * self.bA * dk[c] / self.bd)
        J = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(self.n, self.n)).tocsr()
        return (J.multiply((1.0 / self.V)[:, None])).multiply((1.0 / self._dH_dT(T))[None, :]).tocsc()

    # ------------------------------------------------------------------ dynamics
    def _rhs(self, H, q, T=None):
        T = self.temperature(H) if T is None else T
        k = self.conductivity(T)
        if np.any(k <= 0):
            raise ValueError("conductivity became non-positive; check k_beta_per_K")
        G = self.fA / (self.fda / k[self.fa] + self.fdb / k[self.fb] + self.fRb)
        flux = G * (T[self.fb] - T[self.fa])                   # into a from b
        net = np.zeros(self.n)
        np.add.at(net, self.fa, flux)
        np.subtract.at(net, self.fb, flux)
        if self.bc.size:
            Gb = self.bA / (self.bd / k[self.bc])
            np.subtract.at(net, self.bc, Gb * (T[self.bc] - self.T0))
        return net / self.V + q

    def run(self, Q_total: np.ndarray, t_pulse: float, t_end: float, n_out: int = 300,
            T_init: Optional[np.ndarray] = None, rtol: float = 1e-6, atol_K: float = 1e-4,
            method: str = "newton_bdf2", dT_step_K: float = 2.0) -> ThermalResult:
        """Deposit heat Q_total (J per cell) uniformly over [0, t_pulse]; returns dT = T - T0 histories.

        atol_K is the absolute error tolerance expressed as a temperature (converted to enthalpy with
        rho cp); it matters wherever H passes near zero, e.g. when T0 is chosen inside the melting range."""
        if t_end <= t_pulse or t_pulse < 0:
            raise ValueError("need 0 <= t_pulse < t_end")
        T = np.full(self.n, self.T0) if T_init is None else np.asarray(T_init, float).ravel().copy()
        H = self.enthalpy(T)
        q_on = Q_total.ravel() / self.V / t_pulse if t_pulse > 0 else np.zeros(self.n)
        if method == "newton_bdf2":
            return self._run_newton(H, q_on, t_pulse, t_end, n_out, dT_step_K)
        t_pts = np.unique(np.concatenate([np.linspace(0, t_pulse, max(n_out // 3, 2)),
                                          np.geomspace(max(t_pulse, t_end * 1e-6), t_end, n_out)]))
        out_t, out_H = [], []
        segs = [(0.0, t_pulse, q_on), (t_pulse, t_end, np.zeros(self.n))] if t_pulse > 0 else [(0.0, t_end, np.zeros(self.n))]
        for a, b, q in segs:
            te = t_pts[(t_pts >= a) & (t_pts <= b)]
            # Radau (implicit Runge-Kutta, L-stable) is robust through the stiff melting nonlinearity,
            # where variable-order BDF can stall.
            sol = solve_ivp(lambda t, y, q=q: self._rhs(y, q), (a, b), H, method=method, t_eval=te,
                            jac=lambda t, y: self._jac(y), rtol=rtol, atol=atol_K * self.rho * self.cp)
            if not sol.success:
                raise RuntimeError(sol.message)
            H = sol.y[:, -1]
            keep = slice(1, None) if out_t else slice(None)
            out_t.append(sol.t[keep]); out_H.append(sol.y[:, keep])
        t = np.concatenate(out_t)
        Hs = np.concatenate(out_H, axis=1).T
        Ts = np.array([self.temperature(h) for h in Hs])
        res = ThermalResult(t=t, r_centres=self.geo.rc, z_centres=self.geo.zc,
                            dT=(Ts - self.T0).reshape(t.size, self.nz, self.nr),
                            layer_of_row=self.geo.layer_of_row, C=self.geo.C, r_faces=self.geo.rf)
        res.H = Hs.reshape(t.size, self.nz, self.nr)
        return res

    # ------------------------------------------------------------------ robust phase-change stepper
    def _newton_solve(self, H_guess, rhs_const, beta, q):
        """Solve H - beta * f(H) = rhs_const with full Newton (Jacobian every iteration), damped so that no
        cell's temperature moves more than 50 K per iteration. Returns (H, converged)."""
        H = H_guess.copy()
        I = identity(self.n, format="csc")
        for _ in range(40):
            T = self.temperature(H)                                  # one inversion per iteration
            G = H - beta * self._rhs(H, q, T) - rhs_const
            J = (I - beta * self._jac(H, T)).tocsc()
            dH = spsolve(J, -G)
            dTd = dH / self._dH_dT(T)
            scale = min(1.0, 50.0 / max(np.max(np.abs(dTd)), 1e-300))
            H = H + scale * dH
            if scale == 1.0 and np.max(np.abs(dTd)) < 1e-7:
                return H, True
        return H, False

    def _run_newton(self, H, q_on, t_pulse, t_end, n_out, dT_step_K):
        """Variable-step BDF2 in enthalpy with full Newton; BDF1 restart when the source switches.
        Linear multistep with conservative fluxes => discrete energy conservation is exact.
        Step size is controlled by the largest temperature change per step (dT_step_K)."""
        t_out = np.unique(np.concatenate([np.linspace(0, t_pulse, max(n_out // 3, 2)) if t_pulse > 0 else [0.0],
                                          np.geomspace(max(t_pulse, t_end * 1e-6), t_end, n_out)]))
        t, h = 0.0, (t_pulse if t_pulse > 0 else t_end) * 1e-4
        H_prev, h_prev = None, None
        ts, Hs = [0.0], [H.copy()]
        T_cur = self.temperature(H)
        k_out = 1
        while t < t_end * (1 - 1e-13):
            on = t < t_pulse * (1 - 1e-13)
            q = q_on if on else np.zeros(self.n)
            boundary = t_pulse if on else t_end
            nxt = t_out[k_out] if k_out < t_out.size else t_end
            h = min(h, boundary - t, nxt - t)
            if H_prev is None:                                           # BDF1
                beta, rhs_const = h, H
            else:                                                        # variable-step BDF2
                w = h / h_prev
                beta = h * (1 + w) / (1 + 2 * w)
                rhs_const = ((1 + w) ** 2 * H - w ** 2 * H_prev) / (1 + 2 * w)
            H_new, ok = self._newton_solve(H, rhs_const + beta * 0.0, beta, q)
            if not ok:
                h *= 0.25
                if h < 1e-22:
                    raise RuntimeError("phase-change stepper: step size collapsed")
                continue
            T_new = self.temperature(H_new)
            dTmax = np.max(np.abs(T_new - T_cur))
            if dTmax > 2 * dT_step_K and h > 1e-21:
                h *= 0.5
                continue
            H_prev, h_prev, H, t = H, h, H_new, t + h
            T_cur = T_new
            if abs(t - t_pulse) < 1e-13 * max(t_pulse, 1e-30) and t_pulse > 0:
                H_prev = None                                            # source switches: restart
            if k_out < t_out.size and abs(t - t_out[k_out]) <= 1e-12 * t_end:
                ts.append(t); Hs.append(H.copy()); k_out += 1
            h *= float(np.clip(0.9 * dT_step_K / max(dTmax, 1e-12), 0.3, 2.0))
        t = np.array(ts)
        Hs = np.array(Hs)
        Ts = np.array([self.temperature(x) for x in Hs])
        res = ThermalResult(t=t, r_centres=self.geo.rc, z_centres=self.geo.zc,
                            dT=(Ts - self.T0).reshape(t.size, self.nz, self.nr),
                            layer_of_row=self.geo.layer_of_row, C=self.geo.C, r_faces=self.geo.rf)
        res.H = Hs.reshape(t.size, self.nz, self.nr)
        return res

    def total_enthalpy(self, H: np.ndarray) -> float:
        return float(np.sum(H.ravel() * self.V))
