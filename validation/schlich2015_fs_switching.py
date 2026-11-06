"""Reversible femtosecond-laser switching of ultrathin GST on a gold mirror: Schlich, Zalden, Lindenberg & Spolenak,
ACS Photonics 2, 178 (2015), doi:10.1021/ph500402r.

Experiment (paper, Fig. 1a TEM and text): 40 nm SiO2 / 18 nm Ge2Sb2Te5 / 5 nm Ti+TiN / 158 nm Au / 12 nm Cr /
50 nm SiNx / 50 nm SiO2 / Si. 60 fs pulses at 800 nm, normal incidence, spot diameter 270 um (=> 1D heat flow).
  - amorphisation: single pulse, 14.3 +- 0.8 mJ/cm^2
  - recrystallisation of the melt-quenched spot: repeated pulses at 4.0 +- 0.3 mJ/cm^2, 40 Hz for 16 s (640 pulses)
The paper does not say whether the quoted fluence is the peak or the spot average of the Gaussian beam;
the peak is then 1x or 2x the quoted value. Both are reported.

Model: the 60 fs pulse is absorbed before the lattice heats (no thermo-optic feedback), so the absorbed depth
profile follows exactly from TMM with measured n,k. Heat is deposited over 1 ps (electron-phonon coupling)
and conducted in 1D with latent heat (enthalpy solver).
Predictions:
  - F_onset: top of the GST reaches T_melt; F_full: the whole 18 nm film is molten (amorphisation threshold).
  - window check: at the crystallisation fluence the amorphous film must stay solid, and 640 pulses must give
    nanometre-scale growth (melt-quenched kinetics, Sebastian 2014).
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

LAM, T0, T_MELT, TP = 800.0, 293.0, 895.0, 1e-12
L_GST = 12.13e3 / 0.1141                       # J/kg (12.13 kJ/mol of atoms, mean atomic mass 114.1 g/mol)
F_AMORPH, F_CRYST, N_PULSES = 143.0, 40.0, 640  # J/m^2 (quoted), pulses for recrystallisation
G = sebastian2014_gst()
NK = {m: load_nk(f) for m, f in [("ox", "SiO2_Malitson1965"), ("cr", "GST225_Frantz2023_crystalline"),
                                  ("am", "GST225_Frantz2023_amorphous"), ("tin", "TiN_Beliaev-sputtering"),
                                  ("tin_alt", "TiN_Pfluger1984"), ("ti", "Ti_Johnson1974"), ("au", "Au_Johnson1972")]}
NOMINAL = dict(k_gst=0.5, k_gst_am=0.225, k_liq=1.5, k_ox=1.2, k_tin=15.0, k_ti=10.0, k_au=205.0, k_sin=2.5,
               tbr_gst=7.07e-9, cryst_fraction=1.0, tin="tin")


def gst_nk(phase, p):
    if phase == "am":
        return NK["am"].nk(LAM)
    f = p["cryst_fraction"]                    # partially crystallised film: linear mix of permittivities
    eps = f * NK["cr"].nk(LAM) ** 2 + (1 - f) * NK["am"].nk(LAM) ** 2
    return complex(np.sqrt(eps))


def model(phase, p):
    k_gst = p["k_gst"] if phase == "cr" else p["k_gst_am"]
    layers = [PhaseChangeLayer("cap", 40, p["k_ox"], 2200, 740, 4),
              PhaseChangeLayer("gst", 18, k_gst, 6150, 220, 9, k_liquid=p["k_liq"], latent_J_per_kg=L_GST,
                               T_melt=T_MELT, melt_width_K=3.0, tbr_below=p["tbr_gst"]),
              PhaseChangeLayer("tin", 2.5, p["k_tin"], 5220, 600, 1),
              PhaseChangeLayer("ti", 2.5, p["k_ti"], 4500, 520, 1),
              PhaseChangeLayer("au", 158, p["k_au"], 19300, 129, 8),
              PhaseChangeLayer("cr", 12, 90.0, 7190, 450, 1),
              PhaseChangeLayer("sin", 50, p["k_sin"], 3000, 700, 2),
              PhaseChangeLayer("sio2", 50, 1.38, 2200, 740, 2),
              PhaseChangeLayer("si", 3000, 148.0, 2330, 700, 6)]
    s = NonlinearStackHeatSolver(layers, np.array([0.0, 1e-3]), T0=T0, top="insulated", bottom="fixed",
                                 outer="insulated")
    nl = [NK["ox"].nk(LAM).real, gst_nk(phase, p), NK[p["tin"]].nk(LAM), NK["ti"].nk(LAM), NK["au"].nk(LAM)]
    d = [40.0, 18.0, 2.5, 2.5, 158.0]
    res = tmm(LAM, 1.0, nl, d, NK["au"].nk(LAM))      # 158 nm Au is opaque: what lies below is irrelevant
    frac = optical_row_fractions(LAM, 1.0, nl, d, res, s.geo)
    return s, frac, res


def pulse(s, frac, F, t_end=20e-9):
    out = s.run(s.geo.row_source(F * frac, 1e9), TP, t_end, n_out=200, dT_step_K=10.0)
    rows = s.geo.rows_of_layer(1)
    T = T0 + out.dT[:, rows, 0]                  # (t, GST rows), top row first
    return out.t, T


def melt_state(s, frac, F):
    """(top row melts, whole film melts) for one pulse of quoted fluence F."""
    t, T = pulse(s, frac, F)
    pk = T.max(axis=0)
    return pk[0] >= T_MELT, pk.min() >= T_MELT, t, T


def threshold(s, frac, which, lo=5.0, hi=2000.0):
    idx = 0 if which == "onset" else 1
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if melt_state(s, frac, mid)[idx]:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.003:
            break
    return hi


def analyse(p):
    s, frac, res = model("cr", p)
    A_gst = frac[s.geo.rows_of_layer(1)].sum()
    on, full = threshold(s, frac, "onset"), threshold(s, frac, "full")
    _, _, t, T = melt_state(s, frac, 1.05 * full)
    Tm = T.mean(axis=1)
    i = np.argmax(Tm)
    cool = np.where((t > t[i]) & (Tm < 700.0))[0]
    q_rate = (T_MELT - 700.0) / (t[cool[0]] - t[np.where((t > t[i]) & (Tm < T_MELT))[0][0]]) if cool.size else np.nan
    sa, fa, ra = model("am", p)
    ta, Ta = pulse(sa, fa, F_CRYST)
    growth = trapezoid(G.velocity(np.minimum(Ta.max(axis=1), T_MELT - 1e-6)), ta)   # per pulse, hottest row
    return dict(R_cr=res.R, A_gst_cr=A_gst, F_onset=on, F_full=full, quench_K_per_s=q_rate,
                R_am=ra.R, A_gst_am=fa[sa.geo.rows_of_layer(1)].sum(), Tpk_cryst=Ta.max(),
                growth_per_pulse_nm=growth * 1e9, growth_640_nm=growth * 1e9 * N_PULSES)


def adiabatic_rise(F, A_gst, d_nm=18.0):
    """Upper-bound-free check: absorbed energy in the GST, before any heat can leave (60 fs pulse)."""
    return A_gst * F / (6150 * 220 * d_nm * 1e-9)


def ratio_window(p):
    """Calibration-free test: if every fluence is scaled by an unknown factor (fluence definition, energy leaving
    the GST before it thermalises), the ratio F_cryst / F_amorph must lie between
      lower: amorphous film grows >= 1 nm in 640 pulses,   upper: amorphous film starts to melt,
    both divided by the full-melt threshold of the crystalline film."""
    s, frac, _ = model("cr", p)
    F_full = threshold(s, frac, "full")
    sa, fa, _ = model("am", p)
    F_on_am = threshold(sa, fa, "onset")
    def grown(F):
        t, T = pulse(sa, fa, F)
        return trapezoid(G.velocity(np.minimum(T.max(axis=1), T_MELT - 1e-6)), t) * N_PULSES
    lo, hi = 0.05 * F_on_am, F_on_am
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        lo, hi = (lo, mid) if grown(mid) >= 1e-9 else (mid, hi)
        if hi / lo < 1.003:
            break
    return hi / F_full, F_on_am / F_full, F_full


if __name__ == "__main__" and "--ratio" in sys.argv:
    print(f"measured F_cryst / F_amorph = {F_CRYST / F_AMORPH:.3f} (independent of how fluence is defined)")
    for name, upd in [("nominal", {}), ("TBR 5e-8", dict(tbr_gst=5e-8)), ("60 % crystalline", dict(cryst_fraction=0.6)),
                      ("TiN Pfluger n,k", dict(tin="tin_alt"))]:
        lo, hi, F_full = ratio_window(dict(NOMINAL, **upd))
        print(f"  {name:18s} predicted window {lo:.3f} - {hi:.3f};  implied scale factor model/quoted "
              f"{F_full / F_AMORPH:.2f} (peak) or {F_full / (2 * F_AMORPH):.2f} (average)", flush=True)
    sys.exit()

if __name__ == "__main__":
    cases = [("nominal", {}), ("k_GST 0.4", dict(k_gst=0.4)), ("k_GST 0.6", dict(k_gst=0.6)),
             ("TBR 1e-9", dict(tbr_gst=1e-9)), ("TBR 5e-8", dict(tbr_gst=5e-8)),
             ("TiN k 30", dict(k_tin=30.0)), ("TiN Pfluger n,k", dict(tin="tin_alt")),
             ("60 % crystalline", dict(cryst_fraction=0.6)), ("k_Au 100", dict(k_au=100.0))]
    print(f"quoted: amorphisation {F_AMORPH/10:.1f} mJ/cm^2, crystallisation {F_CRYST/10:.1f} mJ/cm^2 x {N_PULSES} pulses")
    print(f"        peak fluence = 1x (if quoted = peak) or 2x (if quoted = spot average): "
          f"{F_AMORPH/10:.1f}-{2*F_AMORPH/10:.1f} and {F_CRYST/10:.1f}-{2*F_CRYST/10:.1f} mJ/cm^2\n")
    print(f"{'case':18s} R_cr  A_GST | melt onset  full melt | quench   | R_am  A_GST  T_pk@4.0  growth/640 pulses")
    for name, upd in cases:
        p = dict(NOMINAL, **upd)
        r = analyse(p)
        print(f"{name:18s} {r['R_cr']:.2f}  {r['A_gst_cr']:.2f}  | {r['F_onset']/10:6.2f}     {r['F_full']/10:6.2f}   "
              f"| {r['quench_K_per_s']:.1e} | {r['R_am']:.2f}  {r['A_gst_am']:.2f}  {r['Tpk_cryst']:6.0f} K   "
              f"{r['growth_640_nm']:.2e} nm", flush=True)
    print("(fluences in mJ/cm^2, peak)")
