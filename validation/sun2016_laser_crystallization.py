"""Free-space laser crystallisation thresholds of GST films: Sun et al., Sci. Rep. 6, 28246 (2016), open access.

Experiment (paper): ~90 nm PLD Ge2Sb2Te5 (as-deposited amorphous) on 500 nm SiO2 / Si(100); single KrF pulse,
248 nm, 20 ns, top-hat 24 x 6 mm^2 (=> 1D heat flow). Measured: reflectivity change from >= 15 mJ/cm^2,
crystalline XRD from >= 36 mJ/cm^2, full crystallisation (25 % reflectivity contrast) at 67-130 mJ/cm^2.
Model: TMM absorption with the paper's absorption coefficient 8e5 /cm (k = 1.58 at 248 nm; n not given,
bracketed 1.2-2.2), depth-resolved; 1D heat flow (bottom of the oxide held at T0: Si heat sink);
as-deposited GST kinetics (Orava 2012). Predicted thresholds: growth of 1 nm (onset) and 25 nm
(about the observed grain radius, ~ full crystallisation) at the hottest point; melting onset.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]

import numpy as np
from scipy.optimize import brentq

from pqcsim._compat import trapezoid
from pqcsim.devices.growth import orava2012_gst
from pqcsim.optics.tmm import tmm
from pqcsim.thermal.diffusion import StackHeatSolver, ThermalLayer
from pqcsim.thermal.response import optical_row_fractions

LAM, TP, T0, T_MELT = 248.0, 20e-9, 293.0, 900.0
K_GST = 8e7 * LAM * 1e-9 / (4 * np.pi)            # alpha = 8e5 /cm  ->  k = alpha lambda / 4 pi
N_SIO2_248 = 1.508                                  # fused silica at 248 nm (Malitson Sellmeier)
G = orava2012_gst()
MEASURED = {"onset (reflectivity)": 15.0, "XRD crystalline": 36.0, "full crystallisation": 67.0}


def run(n_gst, k_th=0.2):
    layers = [ThermalLayer("gst", 90, k_th, 6150, 220, 45), ThermalLayer("sio2", 500, 1.38, 2200, 740, 20)]
    s = StackHeatSolver(layers, np.array([0.0, 1e-3]), top="insulated", bottom="fixed", outer="insulated")
    nl = [complex(n_gst, K_GST), N_SIO2_248]
    res = tmm(LAM, 1.0, nl, [90.0, 500.0], 3.5)    # Si substrate below the oxide (optically irrelevant here)
    frac = optical_row_fractions(LAM, 1.0, nl, [90.0, 500.0], res, s)
    out = s.run(s.row_source(frac, 1e9), TP, 10 * TP, n_out=200, rtol=1e-8)   # 1 J/m^2, laterally uniform
    T_hot = out.dT[:, :, 0].max(axis=1)              # hottest depth at each time (surface region)
    T_top = out.dT[:, 0, 0]                          # top row (probed by the reflectivity measurement)
    return out.t, T_hot, res.R, T_top


def thresholds(n_gst):
    t, T1, R, _ = run(n_gst)
    def grow(F):
        return trapezoid(G.velocity(T0 + F * T1), t)
    F_melt = (T_MELT - T0) / T1.max()
    f1 = brentq(lambda F: grow(F) - 1e-9, 1e-3, F_melt)
    f25 = brentq(lambda F: grow(F) - 25e-9, 1e-3, F_melt) if grow(F_melt) > 25e-9 else float("nan")
    return R, f1 / 10, f25 / 10, F_melt / 10      # J/m^2 -> mJ/cm^2


def nucleation_thresholds(n_gst=1.6, fractions=(0.01, 0.5, 0.99)):
    """Fluences (mJ/cm^2) at which nucleation + growth (CNT, Orava & Greer 2017) crystallises the given
    fraction of the top of the film (fresh, as-deposited film: no pre-existing nuclei)."""
    from pqcsim.devices.nucleation import CNTNucleation, jmak_crystallised_fraction
    nucl = CNTNucleation()
    t, _, _, T_top = run(n_gst)
    X = lambda F: jmak_crystallised_fraction(t, T0 + F * T_top, nucl)
    grid = np.geomspace(10, 3000, 120)                # J/m^2
    Xs = np.array([X(F) for F in grid])
    out = {}
    for f in fractions:
        i = np.where(Xs >= f)[0]
        out[f] = float(grid[i[0]] / 10) if i.size else float("nan")
    peak_T = T0 + grid * T_top.max()
    return out, grid / 10, Xs, peak_T


if __name__ == "__main__" and "--nucleation" in sys.argv:
    th, F, Xs, Tpk = nucleation_thresholds()
    print("nucleation + growth (CNT, Orava & Greer 2017), n_GST = 1.6:")
    for f, v in th.items():
        print(f"  X = {f:4.2f} at {v:6.1f} mJ/cm^2")
    print("measured: onset 15, XRD 36, full crystallisation 67 mJ/cm^2")
    j = np.where(Tpk >= T_MELT)[0]
    print(f"  (top of film melts above {F[j[0]] if j.size else float('nan'):.1f} mJ/cm^2; crystallisation then proceeds on cooling)")
    sys.exit()

if __name__ == "__main__":
    print(f"k_GST(248 nm) = {K_GST:.2f} from alpha = 8e5 /cm")
    print(" n_GST |  R    | 1 nm growth | 25 nm growth | melting onset   (mJ/cm^2)")
    for n in (1.2, 1.6, 2.0, 2.2):
        R, a, b, c = thresholds(n)
        print(f"  {n:3.1f}  | {R:.3f} |   {a:6.1f}    |   {b:6.1f}     |   {c:6.1f}")
    print("measured:", MEASURED)
