"""Transient heat conduction in a laterally uniform multilayer, axisymmetric (r, z) finite volumes.

    rho cp dT/dt = div(k grad T) + q(r, z, t)

Discretization (cell-centred finite volumes on a non-uniform grid):
    C_i d(dT_i)/dt = sum_j G_ij (dT_j - dT_i) - G_i,b dT_i + Q_i(t),       dT = T - T0
    C_i  = rho cp V_i,           V_i = pi (r_{i+1/2}^2 - r_{i-1/2}^2) dz
    radial G  = 2 pi r_face dz / (dr_l/(2 k) + dr_r/(2 k))
    axial  G  = pi (r_{i+1/2}^2 - r_{i-1/2}^2) / (dz_a/(2 k_a) + dz_b/(2 k_b) + R_b)   (series; R_b = thermal
                boundary resistance at a layer interface, ThermalLayer.tbr_below)
Boundaries: top insulated or fixed T0, bottom fixed T0 (heat sink) or insulated, outer radius fixed T0
or insulated. Integrated in time with an implicit BDF method (stiff, sparse Jacobian).

The equation is linear (constant k, rho, cp), so temperature rises superpose and scale with the
absorbed energy. Temperature-dependent properties and latent heat are not included.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix, csr_matrix, diags


@dataclass(frozen=True)
class ThermalLayer:
    name: str
    thickness_nm: float
    k: float       # W/(m K)
    rho: float     # kg/m^3
    cp: float      # J/(kg K)
    cells: int = 4
    tbr_below: float = 0.0   # thermal boundary resistance to the next layer down, m^2 K / W

    def __post_init__(self):
        if self.thickness_nm <= 0 or self.k <= 0 or self.rho <= 0 or self.cp <= 0 or self.cells < 1:
            raise ValueError(f"layer {self.name}: thickness, k, rho, cp must be > 0 and cells >= 1")
        if self.tbr_below < 0:
            raise ValueError(f"layer {self.name}: thermal boundary resistance must be >= 0")


def radial_grid(r_max_um: float, dr_min_um: float, growth: float = 1.08) -> np.ndarray:
    """Face positions (m): uniform dr_min out to ~4 dr_min, then geometric growth to r_max."""
    if not (0 < dr_min_um < r_max_um) or growth < 1.0:
        raise ValueError("need 0 < dr_min < r_max and growth >= 1")
    faces, dr = [0.0], dr_min_um
    while faces[-1] < r_max_um:
        faces.append(faces[-1] + dr)
        if len(faces) > 4:
            dr *= growth
    faces[-1] = r_max_um
    return np.asarray(faces) * 1e-6


@dataclass
class ThermalResult:
    t: np.ndarray              # s
    r_centres: np.ndarray      # m
    z_centres: np.ndarray      # m (0 at top surface, increasing downwards)
    dT: np.ndarray             # (n_t, n_z, n_r) temperature rise above T0, K
    layer_of_row: np.ndarray   # layer index of each z row
    C: np.ndarray              # (n_z, n_r) heat capacities, J/K
    r_faces: np.ndarray        # m

    def layer_average(self, layer: int, r_m: float) -> np.ndarray:
        """Volume-weighted mean temperature rise of one layer in the radial cell containing r_m."""
        rows = np.where(self.layer_of_row == layer)[0]
        col = int(np.clip(np.searchsorted(self.r_faces, r_m, side="right") - 1, 0, self.r_centres.size - 1))
        w = self.C[rows, col]
        return (self.dT[:, rows, col] * w).sum(axis=1) / w.sum()

    def heat_content(self) -> np.ndarray:
        """Stored heat sum C dT at every time, J."""
        return np.einsum("tzr,zr->t", self.dT, self.C)


class StackHeatSolver:
    def __init__(self, layers: Sequence[ThermalLayer], r_faces_m: np.ndarray,
                 top: str = "insulated", bottom: str = "fixed", outer: str = "fixed"):
        for name, bc in (("top", top), ("bottom", bottom), ("outer", outer)):
            if bc not in ("insulated", "fixed"):
                raise ValueError(f"{name} boundary must be 'insulated' or 'fixed'")
        self.layers = list(layers)
        self.rf = np.asarray(r_faces_m, float)
        if self.rf[0] != 0 or np.any(np.diff(self.rf) <= 0):
            raise ValueError("radial faces must start at 0 and increase")
        self.top, self.bottom, self.outer = top, bottom, outer

        dz, k, rc, lay = [], [], [], []
        self.tbr_row = np.zeros(sum(L.cells for L in self.layers))   # resistance below each row
        r = 0
        for L in self.layers:
            r += L.cells
            self.tbr_row[r - 1] = L.tbr_below
        for li, L in enumerate(self.layers):
            for _ in range(L.cells):
                dz.append(L.thickness_nm * 1e-9 / L.cells)
                k.append(L.k)
                rc.append(L.rho * L.cp)
                lay.append(li)
        self.dz, self.kz, self.rhocp = np.array(dz), np.array(k), np.array(rc)
        self.layer_of_row = np.array(lay)
        self.zc = np.cumsum(self.dz) - self.dz / 2
        self.rc = 0.5 * (self.rf[1:] + self.rf[:-1])
        self.nz, self.nr = self.dz.size, self.rc.size

        ring = np.pi * (self.rf[1:] ** 2 - self.rf[:-1] ** 2)            # annulus areas
        self.V = self.dz[:, None] * ring[None, :]
        self.C = self.rhocp[:, None] * self.V
        self._build(ring)

    def _idx(self, iz, ir):
        return iz * self.nr + ir

    def _build(self, ring):
        rows, cols, vals = [], [], []
        diag = np.zeros(self.nz * self.nr)

        def couple(a, b, G):
            rows.extend([a, b]); cols.extend([b, a]); vals.extend([G, G])
            diag[a] -= G; diag[b] -= G

        dr = np.diff(self.rf)
        for iz in range(self.nz):
            kk = self.kz[iz]
            for ir in range(self.nr - 1):     # radial neighbours (same material in a row)
                G = 2 * np.pi * self.rf[ir + 1] * self.dz[iz] / (dr[ir] / (2 * kk) + dr[ir + 1] / (2 * kk))
                couple(self._idx(iz, ir), self._idx(iz, ir + 1), G)
            if self.outer == "fixed":         # outer face held at T0: half-cell resistance
                diag[self._idx(iz, self.nr - 1)] -= 2 * np.pi * self.rf[-1] * self.dz[iz] / (dr[-1] / (2 * kk))
        for iz in range(self.nz - 1):         # axial neighbours, series resistance across interfaces
            Rz = self.dz[iz] / (2 * self.kz[iz]) + self.dz[iz + 1] / (2 * self.kz[iz + 1]) + self.tbr_row[iz]
            for ir in range(self.nr):
                couple(self._idx(iz, ir), self._idx(iz + 1, ir), ring[ir] / Rz)
        for iz, bc in ((0, self.top), (self.nz - 1, self.bottom)):
            if bc == "fixed":
                for ir in range(self.nr):
                    diag[self._idx(iz, ir)] -= ring[ir] / (self.dz[iz] / (2 * self.kz[iz]))

        n = self.nz * self.nr
        G = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr() + diags(diag)
        self.Cflat = self.C.ravel()
        self.M = csr_matrix(diags(1.0 / self.Cflat) @ G)               # d(dT)/dt = M dT + Q / C

    def ring_energies(self, E_peak: float, w_um: float) -> np.ndarray:
        """Energy (J) in each annulus for fluence E_peak exp(-2 r^2 / w^2), integrated exactly:
        ∫ 2 pi r E0 e^{-2 r^2 / w^2} dr over [r_a, r_b] = (pi w^2 / 2) E0 (e^{-a_a} - e^{-a_b})."""
        w = w_um * 1e-6
        a = 2 * self.rf ** 2 / w ** 2
        # e^{-a_i} - e^{-a_{i+1}} = e^{-a_i} (1 - e^{-(a_{i+1} - a_i)}), stable for very wide beams
        return 0.5 * np.pi * w ** 2 * E_peak * np.exp(-a[:-1]) * -np.expm1(-np.diff(a))

    def row_source(self, row_fluence: np.ndarray, w_um: float) -> np.ndarray:
        """Heat (J) per cell for absorbed peak fluence row_fluence[iz] (J/m^2) in each z row."""
        row_fluence = np.asarray(row_fluence, float)
        if row_fluence.shape != (self.nz,):
            raise ValueError(f"row_fluence must have shape ({self.nz},)")
        return row_fluence[:, None] * self.ring_energies(1.0, w_um)[None, :]

    def gaussian_source(self, layer: int, E_abs_peak: float, w_um: float) -> np.ndarray:
        """Absorbed peak fluence E_abs_peak spread uniformly over the thickness of `layer`."""
        if not 0 <= layer < len(self.layers):
            raise IndexError("layer index out of range")
        rows = self.layer_of_row == layer
        f = np.where(rows, self.dz, 0.0)
        return self.row_source(E_abs_peak * f / f.sum(), w_um)

    def rows_of_layer(self, layer: int) -> np.ndarray:
        return np.where(self.layer_of_row == layer)[0]

    def run(self, Q_total: np.ndarray, t_pulse: float, t_end: float, n_out: int = 400,
            dT0: Optional[np.ndarray] = None, rtol: float = 1e-7) -> ThermalResult:
        """Deposit heat Q_total (J per cell) uniformly in time over [0, t_pulse], simulate to t_end."""
        if t_pulse < 0 or t_end <= t_pulse:
            raise ValueError("need 0 <= t_pulse < t_end")
        y0 = np.zeros(self.nz * self.nr) if dT0 is None else np.asarray(dT0, float).ravel().copy()
        src = (Q_total.ravel() / self.Cflat / t_pulse) if t_pulse > 0 else None
        if t_pulse == 0:
            y0 += Q_total.ravel() / self.Cflat           # instantaneous deposit
        t_pts = np.unique(np.concatenate([np.linspace(0, t_pulse, max(n_out // 4, 2)) if t_pulse > 0 else [0.0],
                                          np.geomspace(max(t_pulse, t_end * 1e-6), t_end, n_out)]))
        out_t, out_y = [], []
        segments = [(0.0, t_pulse, src), (t_pulse, t_end, None)] if t_pulse > 0 else [(0.0, t_end, None)]
        y = y0
        for a, b, s in segments:
            f = (lambda t, y, s=s: self.M @ y + s) if s is not None else (lambda t, y: self.M @ y)
            te = t_pts[(t_pts >= a) & (t_pts <= b)]
            sol = solve_ivp(f, (a, b), y, method="BDF", jac=self.M, t_eval=te, rtol=rtol, atol=1e-9)
            if not sol.success:
                raise RuntimeError(sol.message)
            y = sol.y[:, -1]
            keep = slice(1, None) if out_t else slice(None)
            out_t.append(sol.t[keep]); out_y.append(sol.y[:, keep])
        t = np.concatenate(out_t)
        dT = np.concatenate(out_y, axis=1).T.reshape(t.size, self.nz, self.nr)
        return ThermalResult(t=t, r_centres=self.rc, z_centres=self.zc, dT=dT,
                             layer_of_row=self.layer_of_row, C=self.C, r_faces=self.rf)
