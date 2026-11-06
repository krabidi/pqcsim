"""End-to-end thermal validation against the measured GST transient of Ríos et al., Nat. Photon. 9, 725 (2015),
SI Fig. S3b (90 ns pump, 540 pJ; GST temperature derived from the thermo-optic response).

Measured values are read from the published figure (+-2 C). Model: rios2015_switching geometry and Table S1
thermal properties; optics as there (5 µm cell, absorbed fraction 80.7 % from the measured 7.14 dB).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build"), str(ROOT), str(ROOT / "validation")]

import numpy as np

import rios2015_switching as S

FIG_S3B_C = {0: 20, 25: 41, 50: 51, 90: 58, 100: 49, 150: 39, 200: 33, 250: 30}   # read from the figure, +-2 C
E_PUMP_PJ, ABSORBED = 540.0, 0.807


def model_transient(L_um=5.0):
    s = S.build(L_um)
    Q = S.source(s, L_um, S.lateral_profile)
    res = s.run(Q, 90e-9, 250e-9, steps_pulse=60, steps_after=60)
    gst = s.layer_of_row == 1
    T = np.array([(res.dT[i][gst] * s.C[gst]).sum() / s.C[gst].sum() for i in range(res.t.size)])
    return res.t, T                                   # K per absorbed pJ, GST-average


if __name__ == "__main__":
    t, T = model_transient()
    peak = np.interp(90e-9, t, T)
    print("t (ns) | measured rise (K) | model rise (K) for 540 pJ x 80.7 %")
    for tt, c in FIG_S3B_C.items():
        print(f"{tt:5d}  |      {c - 20:5.1f} ± 2     |   {np.interp(tt * 1e-9, t, T) * E_PUMP_PJ * ABSORBED:6.1f}")
    print(f"\nconsistency of the paper's own data: melting (+607 K) at 90 ns would need about "
          f"{607 / ((58 - 20) / E_PUMP_PJ):.0f} pJ pump, vs ~300 pJ reported as the 100 ns switching energy")
