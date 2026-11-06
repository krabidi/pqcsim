"""Optically programmed phase-change memristor cell (planar thin-film stack).

Optics : exact normal-incidence TMM (pqcsim.optics.tmm), absorption resolved per layer.
Thermal: lumped RC per unit area for the active layer, rectangular optical pulse
           C dT/dt = P - (T - T0) / R,   C = rho cp d_active,   tau = R C
         heating  T(t) = T0 + P R (1 - e^{-t/tau}),                 0 <= t <= t_p
         cooling  T(t) = T0 + (T_peak - T0) e^{-(t - t_p)/tau},     t > t_p
Kinetics (model="physical", default):
  SET  : first-order JMAK  dx/dt = k(T(t)) (1 - x),  k = k0 exp(-Ea/kB T) [x (1 - exp(-dG/kB T))
         if dH_fusion_eV is given, Wilson-Frenkel driving force -> 0 at T_melt], integrated over the whole thermal
         transient (heating and cooling) wherever T_cryst_min <= T < T_melt:
             x' = 1 - (1 - x) exp(-∫ k(T(t)) dt)
  RESET: phenomenological analog erase applied at the temperature peak,
             x' = x (1 - u^gamma),  u = (min(T_peak, T_melt) - T_reset_on) / (T_melt - T_reset_on)
         Crystallization during the heating ramp is applied before the erase and recrystallization
         during the quench after it, so one pulse can both SET and RESET, as physics dictates.
Model "photomem_legacy" reproduces the original scripts: SET uses k(T_avg) t_p only, RESET uses
the erase only, and the caller chooses which one with `kind`.

Note: with pure Arrhenius kinetics fitted at low T, rates near T_melt can be enormous, so a slow quench
(large tau_th) recrystallizes a melted film. That is a real prediction of the model, not a numerical
artefact; the legacy model hides it because it ignores crystallization during RESET.

Approximations (state them when quoting results): absorption is evaluated at the pre-pulse state;
latent heat, temperature-dependent rho/cp and lateral heat flow are neglected (see pqcsim.arrays
for lateral coupling); the RESET rule is phenomenological, not a melt-quench calculation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np
from scipy.integrate import quad

from ..materials import PCMMaterial
from ..optics.tmm import TMMResult, tmm
from .._compat import trapezoid

KB_EV = 8.617333262e-5  # Boltzmann constant, eV/K

Layer = Union[complex, float, str]  # a fixed index, or "active" for the PCM layer


@dataclass
class PulseResult:
    x: float           # crystalline fraction after the pulse
    E_abs: float       # absorbed fluence in the active layer, J/m^2
    A_active: float    # active-layer absorption fraction at the pulse wavelength
    T_peak: float      # K
    T_avg: float       # time-averaged temperature during the pulse, K
    erased: float      # fraction removed by RESET (0 if none)


@dataclass
class PCMCell:
    """air | layers ... | substrate, with exactly one layer equal to the string "active"."""

    material: PCMMaterial
    layers: Sequence[Layer]
    thicknesses_nm: Sequence[float]
    n_inc: complex = 1.0
    n_sub: complex = 1.45
    readout: str = "R"            # weight readout: "R", "T" or "1-R"
    lam_probe_nm: float = 1550.0
    T0: float = 300.0
    model: str = "physical"
    _active: int = field(init=False, repr=False)

    def __post_init__(self):
        idx = [k for k, L in enumerate(self.layers) if isinstance(L, str)]
        if len(idx) != 1 or self.layers[idx[0]] != "active":
            raise ValueError('layers must contain exactly one "active" entry')
        if len(self.layers) != len(self.thicknesses_nm):
            raise ValueError("layers and thicknesses_nm must have the same length")
        if self.readout not in ("R", "T", "1-R"):
            raise ValueError("readout must be 'R', 'T' or '1-R'")
        if self.model not in ("physical", "photomem_legacy"):
            raise ValueError("model must be 'physical' or 'photomem_legacy'")
        self._active = idx[0]

    # ---------------------------------------------------------------- optics
    @property
    def d_active_nm(self) -> float:
        return float(self.thicknesses_nm[self._active])

    def indices(self, x: float, lam_nm: float, channel: str = "probe") -> List[complex]:
        n_act = self.material.nk(x, lam_nm, channel)
        return [n_act if k == self._active else complex(L) for k, L in enumerate(self.layers)]

    def optics(self, x: float, lam_nm: Optional[float] = None, channel: str = "probe") -> TMMResult:
        lam = self.lam_probe_nm if lam_nm is None else lam_nm
        return tmm(lam, self.n_inc, self.indices(x, lam, channel), self.thicknesses_nm, self.n_sub)

    def active_absorption(self, x: float, lam_nm: float, channel: str = "probe") -> float:
        return float(self.optics(x, lam_nm, channel).A_layers[self._active])

    def weight(self, x: float) -> float:
        res = self.optics(x)
        return {"R": res.R, "T": res.T, "1-R": 1.0 - res.R}[self.readout]

    # --------------------------------------------------------------- thermal
    def _thermal(self):
        th = self.material.thermal
        if th is None:
            raise ValueError(f"material {self.material.name} has no thermal parameters")
        C = th.rho * th.cp * self.d_active_nm * 1e-9          # J / (m^2 K)
        return th.tau_th, th.tau_th / C                        # tau (s), R (K m^2 / W)

    def temperatures(self, E_abs: float, t_pulse: float, T0: Optional[float] = None):
        """(T_peak, T_avg) for absorbed fluence E_abs (J/m^2) delivered uniformly over t_pulse (s)."""
        if t_pulse <= 0 or E_abs < 0:
            raise ValueError("t_pulse must be > 0 and E_abs >= 0")
        T0 = self.T0 if T0 is None else T0
        tau, R = self._thermal()
        dT_ss = (E_abs / t_pulse) * R
        s = t_pulse / tau
        T_peak = T0 + dT_ss * -math.expm1(-s)
        T_avg = T0 + dT_ss * (1.0 + math.expm1(-s) / s)
        return T_peak, T_avg

    # -------------------------------------------------------------- kinetics
    def _rate(self, T: float) -> float:
        kin = self.material.kinetics
        if T < kin.T_cryst_min or T >= kin.T_melt:
            return 0.0
        k = kin.k0 * math.exp(-kin.Ea_eV / (KB_EV * T))
        if kin.dH_fusion_eV is not None:
            # Wilson-Frenkel driving force with Turnbull's  dG = dH_f (T_m - T) / T_m  per atom
            k *= -math.expm1(-kin.dH_fusion_eV * (kin.T_melt - T) / (kin.T_melt * KB_EV * T))
        return k

    def _rate_integral(self, T_of_t, t0: float, t1: float) -> float:
        if t1 <= t0:
            return 0.0
        val, _ = quad(lambda t: self._rate(T_of_t(t)), t0, t1, limit=200, epsrel=1e-10, epsabs=0.0)
        return val

    def crystallization_exponent(self, dT_ss: float, t_pulse: float, T0: float):
        """∫ k(T(t)) dt over (heating, cooling) for steady-state rise dT_ss; returns (I_heat, I_cool, T_peak)."""
        kin = self.material.kinetics
        if T0 >= kin.T_cryst_min:
            raise ValueError("ambient T0 must be below T_cryst_min (retention at ambient is not modelled)")
        tau, _ = self._thermal()
        T_peak = T0 + dT_ss * -math.expm1(-t_pulse / tau)
        if T_peak <= kin.T_cryst_min:
            return 0.0, 0.0, T_peak

        def t_heat(Tx):  # time at which the heating ramp reaches Tx (Tx < T_peak)
            return -tau * math.log1p(-(Tx - T0) / dT_ss)

        def t_cool(Tx):  # time after the pulse end at which cooling reaches Tx (Tx < T_peak)
            return tau * math.log((T_peak - T0) / (Tx - T0))

        heat = lambda t: T0 + dT_ss * -math.expm1(-t / tau)
        cool = lambda t: T0 + (T_peak - T0) * math.exp(-t / tau)
        a = t_heat(kin.T_cryst_min)
        b = t_heat(kin.T_melt) if T_peak > kin.T_melt else t_pulse
        I_heat = self._rate_integral(heat, a, b)
        c = t_cool(kin.T_melt) if T_peak > kin.T_melt else 0.0
        d = t_cool(kin.T_cryst_min)
        I_cool = self._rate_integral(cool, c, d)
        return I_heat, I_cool, T_peak

    def _erase(self, x: float, T_peak: float) -> float:
        kin = self.material.kinetics
        if T_peak < kin.T_reset_on:
            return 0.0
        if T_peak >= kin.T_melt and kin.reset_full_on_melt:
            return x
        u = (min(T_peak, kin.T_melt) - kin.T_reset_on) / max(kin.T_melt - kin.T_reset_on, 1e-12)
        return x * u ** kin.reset_gamma

    def apply_pulse(self, x: float, fluence: float, lam_nm: float, t_pulse: float,
                    channel: str = "set", kind: Optional[str] = None,
                    E_abs: Optional[float] = None, T0: Optional[float] = None) -> PulseResult:
        """Apply one rectangular optical pulse of incident fluence (J/m^2) and duration t_pulse (s).

        E_abs overrides the optically computed absorbed fluence (used for thermal crosstalk, where a
        neighbour is heated by conduction rather than by light). `kind` is only used by the legacy model.
        """
        if not 0.0 <= x <= 1.0:
            raise ValueError("x must be in [0, 1]")
        if self.material.kinetics is None:
            raise ValueError(f"material {self.material.name} has no kinetics")
        T0 = self.T0 if T0 is None else T0
        A_act = self.active_absorption(x, lam_nm, channel) if fluence > 0 else 0.0
        if E_abs is None:
            E_abs = max(fluence, 0.0) * A_act
        T_peak, T_avg = self.temperatures(E_abs, t_pulse, T0)

        if self.model == "photomem_legacy":
            if kind not in ("set", "reset"):
                raise ValueError("legacy model needs kind='set' or 'reset'")
            if kind == "set":
                dx = (1 - x) * -math.expm1(-self._rate(T_avg) * t_pulse)
                return PulseResult(min(x + dx, 1.0), E_abs, A_act, T_peak, T_avg, 0.0)
            er = self._erase(x, T_peak)
            return PulseResult(x - er, E_abs, A_act, T_peak, T_avg, er)

        tau, R = self._thermal()
        I_heat, I_cool, _ = self.crystallization_exponent(E_abs / t_pulse * R, t_pulse, T0)
        x1 = x + (1.0 - x) * -math.expm1(-I_heat)       # = 1 - (1-x) e^{-I}, exact when I = 0
        er = self._erase(x1, T_peak)
        x2 = x1 - er
        x3 = x2 + (1.0 - x2) * -math.expm1(-I_cool)
        return PulseResult(float(np.clip(x3, 0.0, 1.0)), E_abs, A_act, T_peak, T_avg, er)

    def apply_temperature_history(self, x: float, t: np.ndarray, T: np.ndarray,
                                  samples_per_segment: int = 64) -> PulseResult:
        """Evolve x along an arbitrary temperature history T(t) (e.g. from pqcsim.thermal).

        Same physics as apply_pulse: JMAK crystallization wherever T_cryst_min <= T < T_melt, with the
        RESET erase applied at the temperature maximum, crystallization before it and recrystallization
        after it. T is interpolated linearly between samples; each interval is subdivided so that the
        exponentially temperature-sensitive rate is integrated accurately.
        """
        if not 0.0 <= x <= 1.0:
            raise ValueError("x must be in [0, 1]")
        t, T = np.asarray(t, float), np.asarray(T, float)
        if t.ndim != 1 or t.shape != T.shape or t.size < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("t and T must be 1-D, equally long, with strictly increasing t")
        if self.model != "physical":
            raise ValueError("temperature histories need model='physical'")
        u = np.linspace(0.0, 1.0, samples_per_segment + 1)[:-1]
        tt = np.concatenate([t[:-1, None] + u[None, :] * np.diff(t)[:, None], t[-1:, None]], axis=None)
        TT = np.interp(tt, t, T)
        k = np.vectorize(self._rate, otypes=[float])(TT)
        ip = int(np.argmax(TT))
        I_heat = trapezoid(k[: ip + 1], tt[: ip + 1])
        I_cool = trapezoid(k[ip:], tt[ip:])
        x1 = x + (1.0 - x) * -math.expm1(-I_heat)
        er = self._erase(x1, float(TT[ip]))
        x2 = x1 - er
        x3 = x2 + (1.0 - x2) * -math.expm1(-I_cool)
        dt_total = t[-1] - t[0]
        return PulseResult(float(np.clip(x3, 0.0, 1.0)), float("nan"), float("nan"), float(TT[ip]),
                           float(trapezoid(TT, tt) / dt_total), er)

