"""Arrays of optically programmed PCM cells: programming with crosstalk, and optical MVM readout.

Write disturb on a neighbour at centre distance r from the addressed cell comes from
  (1) the optical tail of a Gaussian spot with 1/e^2 intensity radius w:
          F(r) = F_peak exp(-2 r^2 / w^2),  absorbed with the neighbour's own A_active;
  (2) lateral heat conduction:  Delta T_neighbour(t) = kappa(r) Delta T_target(t).
      In the lumped model Delta T is linear in absorbed fluence, so this equals an extra absorbed
      fluence kappa(r) E_abs,target. Same time profile for target and neighbour is an approximation
      (conduction really delays and broadens the neighbour's transient); kappa(r) must be calibrated
      against a heat-diffusion solver (e.g. array_memristor_heatdiffusion_validation.py).
Both contributions superpose because the heat equation is linear (constant material parameters).
The addressed cell itself is treated with its peak fluence (cell small compared with the spot).

Alternatively pass heat_response (pqcsim.thermal.PulseHeatResponse for a continuous film, or
PatternedHeatResponse for pixelated cells in a fill material): the full heat-diffusion
temperature history at the target and at every neighbour drives the kinetics, with heat deposited
where the TMM field actually absorbs it (every absorbing layer, resolved in depth). It already contains
the optical tail (as absorbed heat) and conduction, so it replaces spot_radius_um and thermal_kernel.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np

from ..devices.pcm import PCMCell, PulseResult
from ..thermal.response import PatternedHeatResponse, PulseHeatResponse, optical_row_fractions
from ..optics.tmm import tmm

Kernel = Callable[[float], float]


def exponential_kernel(L_th_um: float) -> Kernel:
    """Placeholder conduction kernel kappa(r) = exp(-r / L_th). Calibrate L_th with a heat solver."""
    if L_th_um <= 0:
        raise ValueError("L_th_um must be positive")
    return lambda r_um: math.exp(-r_um / L_th_um)


class PCMArray:
    def __init__(self, cell: PCMCell, rows: int, cols: int, pitch_um: float,
                 spot_radius_um: Optional[float] = None, thermal_kernel: Optional[Kernel] = None,
                 x0: float = 0.0, heat_response=None, thermal_first_layer: int = 0,
                 fill_index: complex = 1.45, optics: str = "local_tmm", rcwa_orders: int = 5):
        if rows < 1 or cols < 1 or pitch_um <= 0:
            raise ValueError("rows, cols must be >= 1 and pitch_um > 0")
        if heat_response is not None and (spot_radius_um or thermal_kernel):
            raise ValueError("heat_response already includes optical tail and conduction; "
                             "do not combine it with spot_radius_um or thermal_kernel")
        self.heat_response = heat_response
        self.thermal_first_layer = thermal_first_layer   # thermal layer index of the first optical layer
        self.fill_index = complex(fill_index)            # optical index between pixels (patterned arrays)
        if optics not in ("local_tmm", "rcwa"):
            raise ValueError("optics must be 'local_tmm' or 'rcwa'")
        if optics == "rcwa" and not isinstance(heat_response, PatternedHeatResponse):
            raise ValueError("optics='rcwa' needs a PatternedHeatResponse (pixelated array)")
        self.optics, self.rcwa_orders = optics, rcwa_orders
        self._rcwa_cache = {}
        if isinstance(heat_response, PatternedHeatResponse) and abs(heat_response.pitch_um - pitch_um) > 1e-12:
            raise ValueError("patterned heat response was computed for a different pitch")
        self.cell, self.rows, self.cols, self.pitch_um = cell, rows, cols, pitch_um
        self.spot_radius_um = spot_radius_um
        self.thermal_kernel = thermal_kernel
        self.X = np.full((rows, cols), float(x0))

    def _distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return self.pitch_um * math.hypot(a[0] - b[0], a[1] - b[1])

    def write(self, i: int, j: int, fluence: float, lam_nm: float, t_pulse: float,
              channel: str = "set", kind: Optional[str] = None) -> PulseResult:
        """Program cell (i, j); every other cell receives its optical-tail and conducted heat."""
        if self.heat_response is not None:
            return self._write_diffusion(i, j, fluence, lam_nm, t_pulse, channel)
        x_before = self.X.copy()
        target = self.cell.apply_pulse(x_before[i, j], fluence, lam_nm, t_pulse, channel, kind)
        for a in range(self.rows):
            for b in range(self.cols):
                if (a, b) == (i, j):
                    continue
                r = self._distance((i, j), (a, b))
                E = 0.0
                if self.spot_radius_um:
                    F_tail = fluence * math.exp(-2 * r ** 2 / self.spot_radius_um ** 2)
                    if F_tail > 0:
                        E += F_tail * self.cell.active_absorption(x_before[a, b], lam_nm, channel)
                if self.thermal_kernel:
                    E += self.thermal_kernel(r) * target.E_abs
                if E > 0:
                    self.X[a, b] = self.cell.apply_pulse(x_before[a, b], 0.0, lam_nm, t_pulse,
                                                         channel, kind, E_abs=E).x
        self.X[i, j] = target.x
        return target

    def _write_diffusion(self, i, j, fluence, lam_nm, t_pulse, channel) -> PulseResult:
        hr = self.heat_response
        if abs(t_pulse - hr.t_pulse) > 1e-15:
            raise ValueError(f"heat_response was computed for t_pulse = {hr.t_pulse}, not {t_pulse}")
        x_before = self.X.copy()
        x0 = x_before[i, j]
        n_layers = self.cell.indices(x0, lam_nm, channel)
        opt = self.cell.optics(x0, lam_nm, channel)
        frac = optical_row_fractions(lam_nm, self.cell.n_inc, n_layers, self.cell.thicknesses_nm, opt,
                                     hr.solver, self.thermal_first_layer)
        A = float(opt.A_layers[self.cell._active])
        rows_F = fluence * frac
        E_peak = fluence * A
        T_cm = self.cell.material.kinetics.T_cryst_min
        target = None
        if isinstance(hr, PatternedHeatResponse) and self.optics == "rcwa":
            rows_F, rows_fill, A = self._rcwa_rows(x0, lam_nm, channel, hr)
            rows_F, rows_fill = fluence * rows_F, fluence * rows_fill
            E_peak = fluence * A
        elif isinstance(hr, PatternedHeatResponse):
            fill_layers = [self.fill_index if k == self.cell._active else n for k, n in enumerate(n_layers)]
            opt_fill = tmm(lam_nm, self.cell.n_inc, fill_layers, self.cell.thicknesses_nm, self.cell.n_sub)
            rows_fill = fluence * optical_row_fractions(lam_nm, self.cell.n_inc, fill_layers,
                                                        self.cell.thicknesses_nm, opt_fill, hr.solver,
                                                        self.thermal_first_layer)
        for a in range(self.rows):
            for b in range(self.cols):
                if isinstance(hr, PatternedHeatResponse):
                    off = (a - i, b - j)
                    if not hr.covers(off):
                        if (a, b) == (i, j):
                            raise ValueError("thermal domain does not contain the addressed cell")
                        continue                               # outside the simulated thermal domain
                    T = self.cell.T0 + hr.rise(off, rows_F, rows_fill)
                else:
                    T = self.cell.T0 + hr.rise(self._distance((i, j), (a, b)), rows_F)
                if (a, b) != (i, j) and T.max() < T_cm:
                    continue                                   # below the crystallization window
                res = self.cell.apply_temperature_history(x_before[a, b], hr.t, T)
                self.X[a, b] = res.x
                if (a, b) == (i, j):
                    target = PulseResult(res.x, E_peak, A, res.T_peak, res.T_avg, res.erased)
        return target

    def _rcwa_rows(self, x0, lam_nm, channel, hr):
        """Per-row absorbed fractions from RCWA of the periodic pixel array (plane-wave illumination).

        The fill is lossless, so absorption in the pixel layer occurs only inside pixels: per pixel area it
        is A_row * pitch^2 / side^2 (heat source on the 'pcm' columns). Uniform layers absorb uniformly
        over the cell (same value for 'pcm' and 'fill' columns). Returns (rows_pcm, rows_fill, A_pixel)."""
        from ..optics.rcwa import RCWA, RCWALayer, pixel_eps
        key = (round(float(x0), 12), float(lam_nm), channel)
        if key not in self._rcwa_cache:
            solver = hr.solver
            n_act = self.cell.material.nk(x0, lam_nm, channel)
            eps_map = pixel_eps(n_act, self.fill_index, self.pitch_um, hr.cell_um, grid=128)
            layers, row_ids = [], []
            for j, (L, d) in enumerate(zip(self.cell.layers, self.cell.thicknesses_nm)):
                th = solver.rows_of_layer(self.thermal_first_layer + j)
                if abs(solver.layers[self.thermal_first_layer + j].thickness_nm - d) > 1e-9:
                    raise ValueError("optical and thermal layer thicknesses differ")
                for r in th:
                    eps = eps_map if j == self.cell._active else complex(L) ** 2
                    layers.append(RCWALayer(d / th.size, eps))
                    row_ids.append((r, j == self.cell._active))
            res = RCWA(lam_nm, (self.pitch_um, self.pitch_um), orders=self.rcwa_orders,
                       n_inc=self.cell.n_inc, n_sub=self.cell.n_sub).solve_unpolarised(layers)
            ff = (hr.cell_um / self.pitch_um) ** 2
            rows_pcm, rows_fill = np.zeros(solver.nz), np.zeros(solver.nz)
            A_pix = 0.0
            for (r, is_act), a in zip(row_ids, res.A_layers):
                if abs(a) < 1e-12:
                    a = 0.0                                # round-off in lossless layers
                if is_act:
                    rows_pcm[r] = a / ff
                    A_pix += a / ff
                else:
                    rows_pcm[r] = rows_fill[r] = a
            self._rcwa_cache[key] = (rows_pcm, rows_fill, A_pix)
        return self._rcwa_cache[key]

    def weights(self) -> np.ndarray:
        return np.vectorize(self.cell.weight)(self.X)

    def mvm(self, P_in: np.ndarray) -> np.ndarray:
        """Incoherent optical matrix-vector product: row output power y_i = sum_j W_ij P_j."""
        P_in = np.asarray(P_in, float)
        if P_in.shape != (self.cols,):
            raise ValueError(f"input must have shape ({self.cols},)")
        if np.any(P_in < 0):
            raise ValueError("optical powers must be non-negative")
        return self.weights() @ P_in
