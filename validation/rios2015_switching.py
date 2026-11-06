"""End-to-end reproduction: switching energy vs pulse length, Ríos et al., Nat. Photon. 9, 725 (2015), SI.

Measured (SI Table S2): minimum *absorbed* switching energy (pJ) vs pulse length, for 1 µm and 5 µm GST cells,
and cell attenuations 0.97 dB (1 µm) and 7.14 dB (5 µm).

Model chain (no parameter fitted to Table S2):
  optics  : absorbed power decays along the cell at the cell's measured attenuation per length;
            across the waveguide it follows |E|^2 of the computed quasi-TE mode in the GST film.
  heat    : 3D voxel solver; Si3N4 rib (335 nm, 165 nm etch, 1.3 µm) on 3.35 µm SiO2 over the Si wafer
            (fixed temperature); GST 10 nm + ITO 10 nm on the rib. Thermal properties from SI Table S1.
  switching onset: the hottest GST voxel reaches the melting point T_m = 900 K (amorphisation needs melt-quench).
Heat flow is linear (constant properties below melting), so one run per pulse length gives T_peak per pJ.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]

import numpy as np

from pqcsim.optics.modes import graded_axis, quasi_te_modes
from pqcsim.thermal.voxel import Material, PatternedLayer, VoxelHeatSolver

# SI Table S1 (density kg/m^3, heat capacity J/kg/K, conductivity W/m/K)
SIO2 = Material("SiO2", 1.38, 2200, 740)
SIN = Material("Si3N4", 30.0, 3200, 700)
GST = Material("GST_cry", 0.59, 6200, 217)
ITO = Material("ITO", 11.0, 7100, 1290)
AIR = Material("air", 0.026, 1.2, 1005)

W, T0, T_MELT = 1.3, 293.0, 900.0
MEASURED = {  # pulse ns: (1 µm energy, err, 5 µm energy, err) -- SI Table S2
    10: (13.4, 0.6, 42.0, 0.3), 25: (30.1, 1.3, 105.2, 0.7), 50: (48.5, 2.6, 186.7, 0.7),
    100: (97.2, 5.1, 298.0, 2.0), 200: (194.0, 10.0, 596.0, 4.0)}
ATTEN_dB = {1.0: 0.97, 5.0: 7.14}


def lateral_profile(y_centres_um):
    """|E_x|^2 of the crystalline-cell quasi-TE mode in the GST film, sampled at lateral positions (µm)."""
    from pqcsim.optics.nkdata import load_nk
    import importlib.util
    spec = importlib.util.spec_from_file_location("rios", ROOT / "validation" / "rios2015_gst_waveguide.py")
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    m = quasi_te_modes(1560.0, R.x, R.y, R.structure(load_nk("GST225_Frantz2023_crystalline").nk(1560.0), 1.8), 1.75)[0]
    j = int(np.argmin(np.abs(R.y - (R.H_SIN + R.D_GST / 2))))
    I = np.abs(m.field[:, j]) ** 2
    return np.interp(np.asarray(y_centres_um) * 1e3, R.x, I)


def build(L_um, fine=0.1):
    half = L_um / 2
    xf = graded_axis([-half - 4, -half, half, half + 4], [8 * fine, fine, 8 * fine], 1.3)
    yf = graded_axis([-3.5, -W / 2, W / 2, 3.5], [8 * fine, fine, 8 * fine], 1.3)
    rib = [(SIN, -1e3, 1e3, -W / 2, W / 2)]
    cell = [(GST, -half, half, -W / 2, W / 2)]
    cap = [(ITO, -half, half, -W / 2, W / 2)]
    layers = [PatternedLayer("ITO", 10, AIR, 1, cap), PatternedLayer("GST", 10, AIR, 2, cell),
              PatternedLayer("rib", 165, AIR, 3, rib), PatternedLayer("slab", 170, SIN, 3),
              PatternedLayer("box_top", 350, SIO2, 4), PatternedLayer("box", 3000, SIO2, 5)]
    return VoxelHeatSolver(layers, xf, yf, top="insulated", bottom="fixed", sides="fixed")


def source(s, L_um, lateral):
    """Heat (J) per voxel for 1 pJ absorbed: exponential decay along x from the cell entry, |E|^2 across y."""
    alpha = ATTEN_dB[L_um] / L_um * np.log(10) / 10          # 1/µm (power)
    x, y = s.xc * 1e6, s.yc * 1e6
    half = L_um / 2
    inx = (x > -half) & (x < half)
    iny = np.abs(y) < W / 2
    wx = np.where(inx, np.exp(-alpha * (x + half)), 0.0) * s.dx
    wy = np.where(iny, lateral(y), 0.0) * s.dy
    rows = s.rows_of_layer(1)
    Q = np.zeros((s.nz, s.nx, s.ny))
    plane = wx[:, None] * wy[None, :]
    plane /= plane.sum()
    for r in rows:
        Q[r] = 1e-12 * plane * s.dz[r] / s.dz[rows].sum()
    return Q


def peak_rise_per_pJ(s, Q, tp_ns):
    tp = tp_ns * 1e-9
    res = s.run(Q, tp, 1.6 * tp, steps_pulse=40, steps_after=24)
    gst = s.layer_of_row == 1
    return float(res.dT[:, gst, :, :].max())


if __name__ == "__main__":
    for L in (1.0, 5.0):
        s = build(L)
        Q = source(s, L, lateral_profile)
        print(f"\n{L:.0f} µm cell: grid {s.nz} x {s.nx} x {s.ny} = {s.nz * s.nx * s.ny} voxels")
        print("  pulse (ns) | predicted E_switch (pJ) | measured (pJ)")
        for tp in sorted(MEASURED):
            k = peak_rise_per_pJ(s, Q, tp)
            E = (T_MELT - T0) / k
            meas = MEASURED[tp][0 if L == 1.0 else 2], MEASURED[tp][1 if L == 1.0 else 3]
            print(f"  {tp:9d}  |  {E:10.1f}             | {meas[0]:6.1f} ± {meas[1]}")
