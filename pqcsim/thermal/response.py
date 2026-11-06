"""Pulse heat response of a stack, computed once and reused for every write in an array.

The heat equation is linear, so the temperature rise for any absorbed depth profile is a weighted
sum of unit responses, one per heated z row:
    dT(r, t) = sum_rows E_row * G_row(r, t),   E_row = absorbed peak fluence in that row (J/m^2)
This lets the optical absorption profile (which changes with the crystalline fraction) be applied
exactly without re-running the solver. For a continuous film heated by a round Gaussian spot the
temperature depends only on r, so the history at a neighbour is the history at r = its distance.
Approximation: the absorbed profile uses the addressed cell's optics everywhere (neighbours with a
different crystalline fraction absorb the optical tail slightly differently).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np

from ..optics.tmm import absorption_profile
from .diffusion import StackHeatSolver


class PulseHeatResponse:
    def __init__(self, solver: StackHeatSolver, active_layer: int, spot_radius_um: float,
                 t_pulse: float, t_end: float, heated_layers: Sequence[int] = None, n_out: int = 400):
        if t_end <= t_pulse:
            raise ValueError("t_end must exceed t_pulse")
        self.solver, self.active_layer = solver, active_layer
        self.spot_radius_um, self.t_pulse, self.t_end = spot_radius_um, t_pulse, t_end
        heated_layers = [active_layer] if heated_layers is None else list(heated_layers)
        self.rows = np.concatenate([solver.rows_of_layer(L) for L in heated_layers])
        self.t = None
        self._unit = {}
        for row in self.rows:
            f = np.zeros(solver.nz)
            f[row] = 1.0
            res = solver.run(solver.row_source(f, spot_radius_um), t_pulse, t_end, n_out=n_out)
            if self.t is None:
                self.t, self._result = res.t, res
            elif not np.array_equal(res.t, self.t):
                raise RuntimeError("inconsistent output times between basis runs")
            self._unit[int(row)] = res

    @lru_cache(maxsize=4096)
    def _unit_trace(self, row: int, r_um: float) -> Tuple[float, ...]:
        return tuple(self._unit[row].layer_average(self.active_layer, r_um * 1e-6))

    def rise(self, r_um: float, row_fluence: np.ndarray) -> np.ndarray:
        """Active-layer temperature rise history (K) at distance r_um for absorbed peak fluence per row."""
        row_fluence = np.asarray(row_fluence, float)
        if row_fluence.shape != (self.solver.nz,):
            raise ValueError(f"row_fluence must have shape ({self.solver.nz},)")
        outside = np.delete(row_fluence, self.rows)
        if np.any(np.abs(outside) > 1e-9 * max(np.abs(row_fluence).max(), 1e-300)):
            raise ValueError("heat deposited in rows that have no basis response; add them to heated_layers")
        r = round(float(r_um), 9)
        out = np.zeros(self.t.size)
        for row in self.rows:
            if row_fluence[row] != 0.0:
                out += row_fluence[row] * np.asarray(self._unit_trace(int(row), r))
        return out

    def uniform_rows(self, layer: int, E_abs_peak: float) -> np.ndarray:
        """Row fluences for E_abs_peak spread uniformly over one layer's thickness."""
        f = np.zeros(self.solver.nz)
        rows = self.solver.rows_of_layer(layer)
        f[rows] = E_abs_peak * self.solver.dz[rows] / self.solver.dz[rows].sum()
        return f


def optical_row_fractions(lam_nm: float, n_inc: complex, n_layers: Sequence[complex], d_nm: Sequence[float],
                          result, solver: StackHeatSolver, first_layer: int = 0) -> np.ndarray:
    """Absorbed fraction of the incident fluence in every z row of the solver.

    Optical layer j is mapped to thermal layer first_layer + j; thicknesses must agree. Thermal
    layers without an optical counterpart (e.g. a thick substrate) receive no heat.
    """
    frac = np.zeros(solver.nz)
    splits = []
    for j, d in enumerate(d_nm):
        L = solver.layers[first_layer + j]
        if abs(L.thickness_nm - d) > 1e-9:
            raise ValueError(f"optical layer {j} is {d} nm but thermal layer {L.name} is {L.thickness_nm} nm")
        splits.append(L.cells)
    prof = absorption_profile(lam_nm, n_inc, n_layers, d_nm, result, splits)
    for j, p in enumerate(prof):
        frac[solver.rows_of_layer(first_layer + j)] = p
    return frac


class PatternedHeatResponse:
    """Heat response of a pixelated array (pqcsim.thermal.voxel), spot centred on the cell at (0, 0).

    Light through a PCM pixel and light through the fill between pixels are absorbed differently, so
    the basis is (z row) x (column type: 'pcm' or 'fill'): lateral source = Gaussian x mask. Any
    pair of depth profiles then gives the temperature by superposition. Cells are squares of side
    cell_um on a grid of pitch_um; the solver domain must contain the neighbours of interest.
    Approximation: each lateral column absorbs as the 1D (local) TMM of its own layer stack.
    """

    def __init__(self, solver, active_layer: int, spot_radius_um: float, t_pulse: float, t_end: float,
                 pitch_um: float, cell_um: float, heated_layers: Sequence[int] = None,
                 steps_pulse: int = 40, steps_after: int = 120):
        self.solver, self.active_layer = solver, active_layer
        self.spot_radius_um, self.t_pulse, self.t_end = spot_radius_um, t_pulse, t_end
        self.pitch_um, self.cell_um = pitch_um, cell_um
        heated_layers = [active_layer] if heated_layers is None else list(heated_layers)
        self.rows = np.concatenate([solver.rows_of_layer(L) for L in heated_layers])
        lat = solver.gaussian_lateral(spot_radius_um)
        L = solver.layers[active_layer]
        self.pcm_mask = np.array([[L.material_at(x, y) is not L.background for y in solver.yc * 1e6]
                                  for x in solver.xc * 1e6])
        if not self.pcm_mask.any():
            raise ValueError("active layer has no inclusions (PCM pixels)")
        self._unit = {}
        self.t = None
        for kind, mask in (("pcm", self.pcm_mask), ("fill", ~self.pcm_mask)):
            for row in self.rows:
                Q = np.zeros((solver.nz, solver.nx, solver.ny))
                Q[row] = lat * mask
                if not Q.any():
                    continue
                res = solver.run(Q, t_pulse, t_end, steps_pulse, steps_after)
                if self.t is None:
                    self.t = res.t
                self._unit[(kind, int(row))] = res
        self._active_rows = solver.layer_of_row == active_layer
        self._cache = {}

    def _cell_trace(self, key, offset):
        ck = (key, offset)
        if ck not in self._cache:
            cx, cy, h = offset[0] * self.pitch_um, offset[1] * self.pitch_um, self.cell_um / 2
            self._cache[ck] = self._unit[key].region_average(self._active_rows, (cx - h, cx + h), (cy - h, cy + h))
        return self._cache[ck]

    def covers(self, offset) -> bool:
        cx, cy, h = offset[0] * self.pitch_um, offset[1] * self.pitch_um, self.cell_um / 2
        xs, ys = self.solver.xf * 1e6, self.solver.yf * 1e6
        return xs[0] <= cx - h and cx + h <= xs[-1] and ys[0] <= cy - h and cy + h <= ys[-1]

    def rise(self, offset, rows_pcm: np.ndarray, rows_fill: np.ndarray) -> np.ndarray:
        """Mean active-layer temperature rise of the pixel at grid offset (di, dj)."""
        if not self.covers(offset):
            raise ValueError(f"cell offset {offset} lies outside the thermal domain")
        out = np.zeros(self.t.size)
        for kind, rows in (("pcm", rows_pcm), ("fill", rows_fill)):
            for row in range(self.solver.nz):
                if rows[row] == 0.0:
                    continue
                if (kind, row) not in self._unit:
                    if row in self.rows:
                        continue          # this column type does not exist in that row's source
                    raise ValueError("heat deposited in a row without a basis response; add it to heated_layers")
                out += rows[row] * self._cell_trace((kind, row), tuple(offset))
        return out
