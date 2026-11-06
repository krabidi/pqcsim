"""Reproduction: Zheng et al., Opt. Mater. Express 8, 1551 (2018), "GST-on-silicon hybrid nanophotonic
integrated circuits" (open access, DOI 10.1364/OME.8.001551).

Geometry (paper, Sec. 2): SOI 220 nm Si on 3 um BOX, 500 nm wide ridge, 190 nm etch (30 nm slab), air top;
20 nm GST + 11 nm ITO on the waveguide (ITO 'relatively transparent', k < 0.002 by ellipsometry; its
real index is not stated -> scanned). TE mode, 1550 nm.
GST optical constants: the paper's own ellipsometry, aGST = 3.89 + 0.02i, cGST = 6.63 + 1.09i
(the paper writes n - ik, i.e. e^{+iwt}; same physics). Si: Li (1980), SiO2: Malitson (1965).
Paper results to compare:
  simulated (FEM): n_eff aGST = 2.62 - 0.0024i, cGST = 2.97 - 0.22i; alpha 0.08 and 7.7 dB/um
  measured (ring resonators, several GST lengths): alpha_aGST = 0.27 +- 0.04 dB/um, alpha_cGST = 7.6 +- 1.0 dB/um
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]

import numpy as np

from pqcsim.optics.modes import graded_axis, quasi_te_modes
from pqcsim.optics.nkdata import load_nk

LAM = 1550.0
W, H_SI, SLAB, D_GST, D_ITO = 500.0, 220.0, 30.0, 20.0, 11.0
N_SI = load_nk("Si_Li1980_293K").nk(LAM).real
N_OX = load_nk("SiO2_Malitson1965").nk(LAM).real
A_GST, C_GST = 3.89 + 0.02j, 6.63 + 1.09j


def structure(n_gst, n_ito, cover):
    """cover='ridge': film on the ridge top only; 'conformal': also on the slab beside the ridge."""
    def index(X, Y):
        n = np.full(X.shape, 1.0 + 0j)
        n[Y < 0] = N_OX
        ridge = np.abs(X) < W / 2
        n[(Y >= 0) & (Y < SLAB)] = N_SI
        n[ridge & (Y >= SLAB) & (Y < H_SI)] = N_SI
        top = np.where(ridge, H_SI, SLAB)
        film = ridge if cover == "ridge" else np.ones_like(ridge)
        n[film & (Y >= top) & (Y < top + D_GST)] = n_gst
        n[film & (Y >= top + D_GST) & (Y < top + D_GST + D_ITO)] = n_ito
        return n
    return index


x = graded_axis([-2500, -W / 2, W / 2, 2500], [40, 8, 40])
y = graded_axis([-2000, 0, SLAB, SLAB + D_GST, SLAB + D_GST + D_ITO, H_SI, H_SI + D_GST, H_SI + D_GST + D_ITO, 1500],
                [30, 4, 1.0, 1.5, 6, 1.0, 1.5, 30])

if __name__ == "__main__":
    print(f"Si {N_SI:.4f}, SiO2 {N_OX:.4f}; grid {x.size} x {y.size}")
    print(" cover      n_ITO | aGST n_eff          alpha(dB/um) | cGST n_eff          alpha(dB/um)")
    for cover in ("ridge", "conformal"):
        for n_ito in (1.7, 1.9, 2.0):
            out = []
            for n_gst in (A_GST, C_GST):
                m = quasi_te_modes(LAM, x, y, structure(n_gst, n_ito + 0.002j, cover), 2.7)[0]
                out.append((m.neff, m.loss_dB_per_um(LAM)))
            (na, la), (nc, lc) = out
            print(f" {cover:9s}  {n_ito:4.2f} | {na.real:.3f}+{na.imag:.4f}i   {la:6.3f}    | {nc.real:.3f}+{nc.imag:.4f}i   {lc:6.2f}")
    print("\npaper FEM:      aGST 2.62+0.0024i, 0.08 dB/um | cGST 2.97+0.22i, 7.7 dB/um")
    print("paper measured: aGST 0.27 +- 0.04 dB/um      | cGST 7.6 +- 1.0 dB/um")
