"""Reproduction: Rios et al., Nature Photonics 9, 725 (2015), "Integrated all-photonic non-volatile
multi-level memory" -- 5 um GST cell on a Si3N4 rib waveguide.

Geometry (paper, Methods): 335 nm Si3N4 on 3.35 um SiO2, 165 nm rib etch (170 nm slab), width 1.3 um,
TE mode; GST ~10 nm with ~10 nm ITO cap on the rib; cell length 5 um.
Measured (paper, text):
  - crystalline cell: optical attenuation 7.14 dB past the device at 1560 nm (80.7 % absorbed)
  - single 100 ns write pulse: readout (1570 nm) transmission change of 21 %
Inputs not stated in the paper: GST225 optical constants (Xu et al., RSC Adv. 8, 21040 (2018), 1550 nm:
amorphous 3.85+0.05i, crystalline 6.5+0.9i), Si3N4 1.99, SiO2 1.444, ITO n = 1.8 with k scanned.
Model: propagation loss of the GST-loaded section from its complex n_eff (quasi-TE FD mode solver);
mode mismatch and reflections at the cell ends are neglected (they add loss to both states).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]

import numpy as np

from pqcsim.optics.modes import graded_axis, quasi_te_modes

W_RIB, H_SIN, SLAB, D_GST, D_ITO, L_CELL = 1300.0, 335.0, 170.0, 10.0, 10.0, 5.0
N_SIN, N_OX = 1.99, 1.444


def structure(n_gst, n_ito):
    def index(X, Y):
        n = np.full(X.shape, 1.0 + 0j)
        n[Y < 0] = N_OX
        n[(Y >= 0) & (Y < SLAB)] = N_SIN
        rib = np.abs(X) < W_RIB / 2
        n[rib & (Y >= SLAB) & (Y < H_SIN)] = N_SIN
        n[rib & (Y >= H_SIN) & (Y < H_SIN + D_GST)] = n_gst
        n[rib & (Y >= H_SIN + D_GST) & (Y < H_SIN + D_GST + D_ITO)] = n_ito
        return n
    return index


x = graded_axis([-3500, -W_RIB / 2, W_RIB / 2, 3500], [60, 15, 60])
y = graded_axis([-2500, 0, SLAB, H_SIN, H_SIN + D_GST, H_SIN + D_GST + D_ITO, 2500], [40, 8, 8, 1.0, 1.0, 40])


def loss_dB(n_gst, n_ito, lam):
    m = quasi_te_modes(lam, x, y, structure(n_gst, n_ito), n_guess=1.75)[0]
    return m.loss_dB_per_um(lam) * L_CELL, m.neff


def measured_run():
    """All optical constants from measured datasets (data/nk): GST Frantz 2023, Si3N4 Luke 2015,
    SiO2 Malitson 1965, ITO Minenkov 2024 (two depositions) plus a transparent-ITO bound."""
    global N_SIN, N_OX
    from pqcsim.optics.nkdata import load_nk
    am, cr = load_nk("GST225_Frantz2023_amorphous"), load_nk("GST225_Frantz2023_crystalline")
    N_SIN, N_OX = load_nk("Si3N4_Luke2015").nk(1560.0).real, load_nk("SiO2_Malitson1965").nk(1560.0).real
    print(f"measured inputs: n_SiN {N_SIN:.4f}, n_SiO2 {N_OX:.4f}, GST cr {cr.nk(1560.0):.3f}, am {am.nk(1570.0):.3f}")
    rows = []
    for label, ito in (("transparent ITO 1.8", lambda lam: 1.8 + 0j),
                       ("ITO Minenkov wafer-top", load_nk("ITO_Minenkov-wafer-top").nk),
                       ("ITO Minenkov glass", load_nk("ITO_Minenkov-glass").nk)):
        L_cr, _ = loss_dB(cr.nk(1560.0), ito(1560.0), 1560.0)
        L_cr70, _ = loss_dB(cr.nk(1570.0), ito(1570.0), 1570.0)
        L_am70, _ = loss_dB(am.nk(1570.0), ito(1570.0), 1570.0)
        change = 1 - 10 ** (-(L_cr70 - L_am70) / 10)
        rows.append((label, L_cr, L_am70, change))
        print(f"  {label:24s}: crystalline {L_cr:5.2f} dB @1560 | amorphous {L_am70:5.2f} dB @1570 | full-switch change {change*100:5.1f} %")
    return rows


if __name__ == "__main__" and "--measured" in sys.argv:
    measured_run()
    sys.exit()

if __name__ == "__main__":
    GST_AM, GST_CR = 3.85 + 0.05j, 6.5 + 0.9j
    print(f"grid {x.size} x {y.size} nodes")
    bare, nb = loss_dB(1.0, 1.0, 1560.0)
    print(f"bare waveguide n_eff = {nb.real:.4f}")
    print("\n k_ITO | cryst. loss @1560 (dB, 5 um) | amorph. loss @1570 | full-switch readout change @1570")
    for k_ito in (0.0, 0.05, 0.1, 0.3):
        n_ito = 1.8 + 1j * k_ito
        L_cr_1560, _ = loss_dB(GST_CR, n_ito, 1560.0)
        L_cr_1570, _ = loss_dB(GST_CR, n_ito, 1570.0)
        L_am_1570, _ = loss_dB(GST_AM, n_ito, 1570.0)
        T_cr, T_am = 10 ** (-L_cr_1570 / 10), 10 ** (-L_am_1570 / 10)
        print(f"  {k_ito:4.2f} |            {L_cr_1560:5.2f}            |       {L_am_1570:5.2f}       |   "
              f"{(T_am - T_cr) / T_am * 100:5.1f} %  (T_am/T_cr = {T_am / T_cr:.2f})")
    print("\nmeasured: crystalline 7.14 dB @1560; single-pulse readout change 21 % @1570 (partial switching ->"
          " lower bound of the full-switch change)")
