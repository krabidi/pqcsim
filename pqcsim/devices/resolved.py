"""Spatially resolved optical programming of a PCM film: optics, nonlinear heat flow and melt-quench.

One pulse, step by step:
  1. Optics per radial column: the active layer is split into its thermal rows, each with its own
     crystalline fraction x(row, column); a 1D TMM per column gives the absorbed fraction in every
     row of every layer (local-TMM approximation: columns are optically independent).
  2. Heat: absorbed energy E(r) = F exp(-2 r^2 / w^2) per annulus (exact ring integrals) is deposited
     over the pulse; the enthalpy solver (latent heat, T- and phase-dependent conductivity) evolves T.
  3. Phase per cell (melt-quench RESET, no phenomenological erase):
       - cells that reach T_melt become fully amorphous (x = 0) at melting, then recrystallize only
         through the JMAK kinetics during the quench below T_melt;
       - cells that never melt follow the JMAK SET kinetics along their own T(t).
The returned x field feeds the next pulse.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from ..optics.tmm import absorption_profile, tmm
from ..thermal.nonlinear import NonlinearStackHeatSolver
from .pcm import PCMCell


@dataclass
class ResolvedPulse:
    x_field: np.ndarray          # (rows of the active layer, radial columns) after the pulse
    x_mean: float                # heat-capacity-weighted mean over the spot region (r <= spot radius)
    T_peak_field: np.ndarray     # peak temperature per active-layer cell, K
    melted: np.ndarray           # bool per active-layer cell
    absorbed_J: float            # total absorbed energy, J
    t: np.ndarray
    T_active_mean: np.ndarray    # mean active-layer temperature at r = 0 column, K


def _melt_quench_cell(cell: PCMCell) -> PCMCell:
    """Copy of the cell whose RESET is physical melt-quench: full amorphization iff T_peak >= T_melt."""
    kin = replace(cell.material.kinetics, reset_full_on_melt=True,
                  T_reset_on=cell.material.kinetics.T_melt)       # no partial erase below the melt
    return replace(cell, material=replace(cell.material, kinetics=kin))


def resolved_pulse(cell: PCMCell, solver: NonlinearStackHeatSolver, x_field: np.ndarray, fluence: float,
                   lam_nm: float, t_pulse: float, t_end: float, spot_radius_um: float,
                   first_layer: int = 0, channel: str = "set", n_out: int = 200,
                   dT_step_K: float = 5.0) -> ResolvedPulse:
    g = solver.geo
    act_t = first_layer + cell._active                    # thermal layer index of the PCM
    rows = g.rows_of_layer(act_t)
    x_field = np.asarray(x_field, float)
    if x_field.shape != (rows.size, g.nr):
        raise ValueError(f"x_field must have shape ({rows.size}, {g.nr})")
    for j, d in enumerate(cell.thicknesses_nm):
        if abs(solver.layers[first_layer + j].thickness_nm - d) > 1e-9:
            raise ValueError("optical and thermal layer thicknesses differ")

    # 1. optics per column, active layer split into its rows
    ring_E = g.ring_energies(fluence, spot_radius_um)    # incident energy per annulus, J
    Q = np.zeros((g.nz, g.nr))
    for c in range(g.nr):
        n_layers, d_layers, row_map = [], [], []
        for j, (L, d) in enumerate(zip(cell.layers, cell.thicknesses_nm)):
            th_rows = g.rows_of_layer(first_layer + j)
            if j == cell._active:
                for k, r in enumerate(th_rows):
                    n_layers.append(cell.material.nk(float(x_field[k, c]), lam_nm, channel))
                    d_layers.append(d / th_rows.size)
                    row_map.append([r])
            else:
                n_layers.append(complex(L))
                d_layers.append(d)
                row_map.append(list(th_rows))
        res = tmm(lam_nm, cell.n_inc, n_layers, d_layers, cell.n_sub)
        prof = absorption_profile(lam_nm, cell.n_inc, n_layers, d_layers, res, [len(m) for m in row_map])
        for p, m in zip(prof, row_map):
            Q[m, c] = p * ring_E[c]

    # 2. nonlinear heat flow
    T_init = np.full(solver.n, cell.T0)
    out = solver.run(Q, t_pulse, t_end, n_out=n_out, T_init=T_init, dT_step_K=dT_step_K)
    T_hist = out.dT + solver.T0                           # (t, nz, nr)

    # 3. per-cell phase evolution with melt-quench RESET
    mq = _melt_quench_cell(cell)
    Tm = cell.material.kinetics.T_melt
    x_new = np.empty_like(x_field)
    T_peak = np.empty_like(x_field)
    for k, r in enumerate(rows):
        for c in range(g.nr):
            res_c = mq.apply_temperature_history(float(x_field[k, c]), out.t, T_hist[:, r, c])
            x_new[k, c] = res_c.x
            T_peak[k, c] = res_c.T_peak
    melted = T_peak >= Tm
    C = g.C[rows, :]
    in_spot = g.rc <= spot_radius_um * 1e-6
    w = C[:, in_spot]
    return ResolvedPulse(x_field=x_new, x_mean=float((x_new[:, in_spot] * w).sum() / w.sum()),
                         T_peak_field=T_peak, melted=melted, absorbed_J=float(Q.sum()), t=out.t,
                         T_active_mean=(T_hist[:, rows, 0] * C[:, 0]).sum(axis=1) / C[:, 0].sum())
