"""3D transient heat conduction on a Cartesian voxel grid, for patterned (pixelated) arrays.

Each layer has a background material and optional rectangular inclusions (e.g. PCM cells embedded
in SiO2, a patterned mirror). Finite volumes with face conductances
    G = A / (d_1 / (2 k_1) + d_2 / (2 k_2) + R_b)      (exact series resistance at material interfaces)
R_b is an optional thermal boundary (Kapitza) resistance per material pair, applied on every face where the two
materials meet (top, bottom and side walls of inclusions alike).
Boundaries: top insulated or fixed; bottom fixed (heat sink) or insulated; four sides fixed (T0)
or insulated. Time integration: BDF2 (second order, L-stable) at constant step within each phase,
variable-step BDF2 at phase changes (so the scheme stays second order), one cached sparse LU per
distinct step coefficient.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.linalg import splu
from scipy.special import erf


@dataclass(frozen=True)
class Material:
    name: str
    k: float
    rho: float
    cp: float

    def __post_init__(self):
        if min(self.k, self.rho, self.cp) <= 0:
            raise ValueError(f"material {self.name}: k, rho, cp must be > 0")


@dataclass
class PatternedLayer:
    name: str
    thickness_nm: float
    background: Material
    cells: int = 2
    inclusions: List[Tuple[Material, float, float, float, float]] = field(default_factory=list)
    # each inclusion: (material, x_min_um, x_max_um, y_min_um, y_max_um)

    def material_at(self, x_um: float, y_um: float) -> Material:
        for mat, x0, x1, y0, y1 in self.inclusions:
            if x0 <= x_um < x1 and y0 <= y_um < y1:
                return mat
        return self.background


def uniform_faces(half_width_um: float, step_um: float) -> np.ndarray:
    n = int(round(2 * half_width_um / step_um))
    return np.linspace(-half_width_um, half_width_um, n + 1)


@dataclass
class VoxelResult:
    t: np.ndarray
    dT: np.ndarray            # (n_t, nz, nx, ny)
    solver: "VoxelHeatSolver"

    def region_average(self, row_mask: np.ndarray, x_um: Tuple[float, float], y_um: Tuple[float, float]) -> np.ndarray:
        """Heat-capacity-weighted mean rise over rows (bool mask over z) and a lateral rectangle."""
        s = self.solver
        sel = s.region_selector(row_mask, x_um, y_um)
        w = s.C * sel
        return np.einsum("tzxy,zxy->t", self.dT, w) / w.sum()

    def heat_content(self) -> np.ndarray:
        return np.einsum("tzxy,zxy->t", self.dT, self.solver.C)


class VoxelHeatSolver:
    def __init__(self, layers: Sequence[PatternedLayer], x_faces_um: np.ndarray, y_faces_um: np.ndarray,
                 top: str = "insulated", bottom: str = "fixed", sides: str = "fixed",
                 tbr: Optional[dict] = None):
        """tbr: {(material_name_a, material_name_b): R_b in m^2 K / W}, symmetric in the pair."""
        for nm, bc in (("top", top), ("bottom", bottom), ("sides", sides)):
            if bc not in ("insulated", "fixed"):
                raise ValueError(f"{nm} boundary must be 'insulated' or 'fixed'")
        self.layers = list(layers)
        self.xf, self.yf = np.asarray(x_faces_um, float) * 1e-6, np.asarray(y_faces_um, float) * 1e-6
        if np.any(np.diff(self.xf) <= 0) or np.any(np.diff(self.yf) <= 0):
            raise ValueError("faces must increase")
        self.xc, self.yc = 0.5 * (self.xf[1:] + self.xf[:-1]), 0.5 * (self.yf[1:] + self.yf[:-1])
        self.dx, self.dy = np.diff(self.xf), np.diff(self.yf)
        dz, lay = [], []
        for li, L in enumerate(self.layers):
            dz += [L.thickness_nm * 1e-9 / L.cells] * L.cells
            lay += [li] * L.cells
        self.dz, self.layer_of_row = np.array(dz), np.array(lay)
        self.nz, self.nx, self.ny = self.dz.size, self.xc.size, self.yc.size
        k = np.zeros((self.nz, self.nx, self.ny))
        rc = np.zeros_like(k)
        names = []
        mid = np.zeros(k.shape, int)
        for iz, li in enumerate(lay):
            L = self.layers[li]
            for ix, x in enumerate(self.xc * 1e6):
                for iy, y in enumerate(self.yc * 1e6):
                    m = L.material_at(x, y)
                    k[iz, ix, iy], rc[iz, ix, iy] = m.k, m.rho * m.cp
                    if m.name not in names:
                        names.append(m.name)
                    mid[iz, ix, iy] = names.index(m.name)
        self.k = k
        self._mid, self._names = mid, names
        self._tbr = np.zeros((len(names), len(names)))
        for (a, b), R in (tbr or {}).items():
            if a in names and b in names:
                ia, ib = names.index(a), names.index(b)
                self._tbr[ia, ib] = self._tbr[ib, ia] = R
        self.V = self.dz[:, None, None] * self.dx[None, :, None] * self.dy[None, None, :]
        self.C = rc * self.V
        self._build(top, bottom, sides)

    def _build(self, top, bottom, sides):
        nz, nx, ny = self.nz, self.nx, self.ny
        idx = np.arange(nz * nx * ny).reshape(nz, nx, ny)
        k, dz, dx, dy = self.k, self.dz, self.dx, self.dy
        rows, cols, vals = [], [], []
        diag = np.zeros(nz * nx * ny)

        def couple(a, b, G):
            a, b, G = a.ravel(), b.ravel(), G.ravel()
            rows.extend([a, b]); cols.extend([b, a]); vals.extend([G, G])
            np.subtract.at(diag, a, G); np.subtract.at(diag, b, G)

        Ax = dz[:, None, None] * dy[None, None, :]
        tb = self._tbr[self._mid[:, :-1, :], self._mid[:, 1:, :]]
        R = dx[None, :-1, None] / (2 * k[:, :-1, :]) + dx[None, 1:, None] / (2 * k[:, 1:, :]) + tb
        couple(idx[:, :-1, :], idx[:, 1:, :], np.broadcast_to(Ax, (nz, nx - 1, ny)) / R)
        Ay = dz[:, None, None] * dx[None, :, None]
        tb = self._tbr[self._mid[:, :, :-1], self._mid[:, :, 1:]]
        R = dy[None, None, :-1] / (2 * k[:, :, :-1]) + dy[None, None, 1:] / (2 * k[:, :, 1:]) + tb
        couple(idx[:, :, :-1], idx[:, :, 1:], np.broadcast_to(Ay, (nz, nx, ny - 1)) / R)
        Az = dx[:, None] * dy[None, :]
        tb = self._tbr[self._mid[:-1], self._mid[1:]]
        R = dz[:-1, None, None] / (2 * k[:-1]) + dz[1:, None, None] / (2 * k[1:]) + tb
        couple(idx[:-1], idx[1:], Az[None] / R)
        if top == "fixed":
            np.subtract.at(diag, idx[0].ravel(), (Az / (dz[0] / (2 * k[0]))).ravel())
        if bottom == "fixed":
            np.subtract.at(diag, idx[-1].ravel(), (Az / (dz[-1] / (2 * k[-1]))).ravel())
        if sides == "fixed":
            for sl, d, A in ((np.s_[:, 0, :], dx[0], Ax[:, 0, :]), (np.s_[:, -1, :], dx[-1], Ax[:, 0, :]),
                             (np.s_[:, :, 0], dy[0], Ay[:, :, 0]), (np.s_[:, :, -1], dy[-1], Ay[:, :, 0])):
                np.subtract.at(diag, idx[sl].ravel(), (A / (d / (2 * k[sl]))).ravel())
        n = nz * nx * ny
        G = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsc()
        self.G = G + diags(diag)
        self.Cflat = self.C.ravel()

    def rows_of_layer(self, layer: int) -> np.ndarray:
        return np.where(self.layer_of_row == layer)[0]

    # ------------------------------------------------------------------ sources and regions
    def gaussian_lateral(self, w_um: float, x0_um: float = 0.0, y0_um: float = 0.0) -> np.ndarray:
        """Fraction of the beam's power (peak fluence 1 J/m^2, 1/e^2 radius w) falling on each (x, y) cell,
        integrated exactly: ∫∫ e^{-2((x-x0)^2+(y-y0)^2)/w^2} = (pi w^2/8) [erf]_x [erf]_y."""
        w = w_um * 1e-6
        ex = erf(math.sqrt(2) * (self.xf - x0_um * 1e-6) / w)
        ey = erf(math.sqrt(2) * (self.yf - y0_um * 1e-6) / w)
        return (math.pi * w ** 2 / 8) * np.diff(ex)[:, None] * np.diff(ey)[None, :]

    def region_selector(self, row_mask, x_um, y_um) -> np.ndarray:
        sx = (self.xc * 1e6 >= x_um[0]) & (self.xc * 1e6 < x_um[1])
        sy = (self.yc * 1e6 >= y_um[0]) & (self.yc * 1e6 < y_um[1])
        return np.asarray(row_mask, bool)[:, None, None] & sx[None, :, None] & sy[None, None, :]

    # ------------------------------------------------------------------ time integration
    def _lu(self, I, M, beta):
        key = round(beta, 30)
        if key not in self._lu_cache:
            # 3D LU factors are large: keep only the two most recent step coefficients (memory, not accuracy)
            while len(self._lu_cache) >= 2:
                self._lu_cache.pop(next(iter(self._lu_cache)))
            self._lu_cache[key] = splu((I - beta * M).tocsc())
        return self._lu_cache[key]

    def run(self, Q: np.ndarray, t_pulse: float, t_end: float, steps_pulse: int = 40,
            steps_after: int = 120) -> VoxelResult:
        """Deposit heat Q (J per voxel, shape (nz, nx, ny)) uniformly over [0, t_pulse], then cool to t_end.

        BDF2 with history (y_prev, y, h_prev). Whenever the source switches (t = 0, t = t_pulse) the
        history is discarded (a multistep method would otherwise extrapolate the heating ramp and inject
        spurious energy) and the fast transient is resolved with 4 sub-steps of h/4 (BDF1, then BDF2).
        A step-size change uses variable-step BDF2:  w = h/h_prev, b = h(1+w)/(1+2w),
            y_new - b (M y_new + s) = ((1+w)^2 y - w^2 y_prev) / (1+2w).
        """
        if Q.shape != (self.nz, self.nx, self.ny):
            raise ValueError("Q has the wrong shape")
        if not 0 < t_pulse < t_end:
            raise ValueError("need 0 < t_pulse < t_end")
        Cinv = 1.0 / self.Cflat
        src_on = Cinv * Q.ravel() / t_pulse
        I = identity(self.Cflat.size, format="csc")
        M = (diags(Cinv) @ self.G).tocsc()
        self._lu_cache = {}

        st = {"y": np.zeros(self.Cflat.size), "y_prev": None, "h_prev": None, "t": 0.0}
        ts, ys = [0.0], [st["y"].copy()]

        def step(h, s):
            y, yp, hp = st["y"], st["y_prev"], st["h_prev"]
            if yp is None:
                y_new = self._lu(I, M, h).solve(y + h * s)
            else:
                w = h / hp
                b = h * (1 + w) / (1 + 2 * w)
                y_new = self._lu(I, M, b).solve(((1 + w) ** 2 * y - w ** 2 * yp) / (1 + 2 * w) + b * s)
            st.update(y_prev=y, y=y_new, h_prev=h, t=st["t"] + h)
            ts.append(st["t"]); ys.append(y_new.copy())

        def segment(t_stop, h, s, restart):
            if restart:
                st["y_prev"] = st["h_prev"] = None
                for _ in range(4):
                    step(h / 4, s)
            n = max(int(round((t_stop - st["t"]) / h)), 1)
            h = (t_stop - st["t"]) / n
            for _ in range(n):
                step(h, s)

        h = t_pulse / steps_pulse
        segment(t_pulse, h, src_on, restart=True)
        # cooling: first phase keeps the pulse step, later phases grow the step x4 each
        t_phase_end, first = t_pulse, True
        per_phase = max(steps_after // 4, 1)
        while st["t"] < t_end * (1 - 1e-12):
            if not first:
                h *= 4
            t_phase_end = min(st["t"] + per_phase * h, t_end)
            segment(t_phase_end, h, 0.0, restart=first)
            first = False
        t = np.array(ts)
        dT = np.array(ys).reshape(t.size, self.nz, self.nx, self.ny)
        return VoxelResult(t=t, dT=dT, solver=self)
