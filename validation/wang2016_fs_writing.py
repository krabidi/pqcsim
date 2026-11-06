"""Femtosecond-laser writing and erasing of pixels in a GST film: Wang et al., Nat. Photon. 10, 60 (2016),
doi:10.1038/nphoton.2015.247.

Experiment (Methods, SI S1, S3): 70 nm ZnS-SiO2 / 70 nm Ge2Sb2Te5 / 70 nm ZnS-SiO2 on glass; 85 fs pulses at
730 nm focused with NA 0.8 (marks 0.59 um). Crystallisation: trains of up to 80 pulses at ~140 mJ/cm^2 (1 MHz);
erasure (re-amorphisation): single pulse at ~445 mJ/cm^2. 0.39 nJ <-> 140 mJ/cm^2 implies an area of 0.28 um^2,
i.e. the quoted value is energy per diffraction-limited spot area.

Test (calibration-free, as for Schlich 2015): the measured ratio F_cryst / F_erase = 0.31 must lie between
  lower: amorphous film grows >= 1 nm in 80 pulses,    upper: amorphous film starts to melt,
both divided by the full-melt fluence of the crystalline film. At the spot centre the 85 fs pulse heats the film
before heat can move (laterally or vertically), so the 1D model at the centre is used for the peak temperatures.
The ZnS-SiO2 index is not given in the paper: n = 2.1 bracketed over 2.0-2.3.
Absolute check: adiabatic temperature rise at the quoted fluence.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]

import numpy as np

from pqcsim._compat import trapezoid
from pqcsim.devices.growth import sebastian2014_gst
from pqcsim.optics.nkdata import load_nk
from pqcsim.optics.tmm import tmm
from pqcsim.thermal.nonlinear import NonlinearStackHeatSolver, PhaseChangeLayer
from pqcsim.thermal.response import optical_row_fractions

LAM, T0, T_MELT, TP = 730.0, 293.0, 895.0, 1e-12
L_GST = 12.13e3 / 0.1141
F_CRYST, F_ERASE, N_PULSES = 1400.0, 4450.0, 80
G = sebastian2014_gst()
CR, AM = load_nk("GST225_Frantz2023_crystalline"), load_nk("GST225_Frantz2023_amorphous")
N_GLASS = load_nk("SiO2_Malitson1965").nk(LAM).real


def model(phase, n_zs=2.1, k_gst=None, k_zs=0.6):
    k_gst = k_gst if k_gst is not None else (0.5 if phase == "cr" else 0.225)
    layers = [PhaseChangeLayer("cap", 70, k_zs, 3650, 560, 4),
              PhaseChangeLayer("gst", 70, k_gst, 6150, 220, 14, k_liquid=1.5, latent_J_per_kg=L_GST,
                               T_melt=T_MELT, melt_width_K=3.0, tbr_below=7.07e-9),
              PhaseChangeLayer("under", 70, k_zs, 3650, 560, 4),
              PhaseChangeLayer("glass", 3000, 1.0, 2500, 800, 8)]
    s = NonlinearStackHeatSolver(layers, np.array([0.0, 1e-3]), T0=T0, top="insulated", bottom="fixed",
                                 outer="insulated")
    n = (CR if phase == "cr" else AM).nk(LAM)
    nl, d = [n_zs, n, n_zs], [70.0, 70.0, 70.0]
    res = tmm(LAM, 1.0, nl, d, N_GLASS)
    return s, optical_row_fractions(LAM, 1.0, nl, d, res, s.geo), res


def pulse(s, frac, F, t_end=30e-9):
    out = s.run(s.geo.row_source(F * frac, 1e9), TP, t_end, n_out=200, dT_step_K=10.0)
    return out.t, T0 + out.dT[:, s.geo.rows_of_layer(1), 0]


def bisect(pred, lo, hi):
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        lo, hi = (lo, mid) if pred(mid) else (mid, hi)
        if hi / lo < 1.003:
            break
    return hi


def window(melt_depth=1.0, **kw):
    """melt_depth: fraction of the 70 nm film that must melt to erase (1 = whole film; the optical contrast
    comes from the top ~2 penetration depths, 2 x 18 nm, so 0.5 is also a defensible criterion)."""
    s, f, rc = model("cr", **kw)
    n_rows = s.geo.rows_of_layer(1).size
    need = max(int(np.ceil(melt_depth * n_rows)), 1)
    F_full = bisect(lambda F: (pulse(s, f, F)[1].max(axis=0)[:need] >= T_MELT).all(), 1.0, 1e4)
    sa, fa, ra = model("am", **kw)
    F_on = bisect(lambda F: pulse(sa, fa, F)[1].max() >= T_MELT, 1.0, 1e4)
    def grown(F):
        t, T = pulse(sa, fa, F)
        return trapezoid(G.velocity(np.minimum(T.max(axis=1), T_MELT - 1e-6)), t) * N_PULSES >= 1e-9
    F_g = bisect(grown, 0.05 * F_on, F_on)
    A_am = fa[sa.geo.rows_of_layer(1)].sum()
    return F_g / F_full, F_on / F_full, F_full, A_am, rc.R, ra.R


if __name__ == "__main__":
    print(f"measured F_cryst / F_erase = {F_CRYST / F_ERASE:.3f}")
    for name, kw in [("nominal n_ZnS-SiO2 2.1", {}), ("n_ZnS-SiO2 2.0", dict(n_zs=2.0)),
                     ("n_ZnS-SiO2 2.3", dict(n_zs=2.3)), ("k_ZnS-SiO2 1.0", dict(k_zs=1.0)),
                     ("top half melts", dict(melt_depth=0.5)), ("top 30 % melts", dict(melt_depth=0.3))]:
        lo, hi, F_full, A_am, Rc, Ra = window(**kw)
        dT = A_am * F_CRYST / (6150 * 220 * 70e-9)
        print(f"  {name:24s} R_cr {Rc:.2f} R_am {Ra:.2f} | window {lo:.3f} - {hi:.3f} | full melt {F_full/10:5.1f} mJ/cm^2 "
              f"(quoted erase {F_ERASE/10:.0f}); adiabatic rise at quoted 140 mJ/cm^2: {dT:5.0f} K", flush=True)
