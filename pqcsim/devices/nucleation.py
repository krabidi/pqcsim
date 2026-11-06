"""Classical nucleation theory for GST and non-isothermal JMAK crystallisation (nucleation + growth).

CNT exactly as in Orava & Greer, Acta Mater. 139, 226 (2017):
    W(n)   = n dmu + (36 pi)^{1/3} v^{2/3} n^{2/3} sigma                         (1)
    n*     = 32 pi v^2 sigma^3 / (3 |dmu|^3)                                       (2)
    N_eq   = N_A exp(-W(n*)/kT)                                                    (3)
    I_st   = gamma (sigma/kT)^{1/2} v^{1/3} 4 (8/(81 pi^2))^{1/6} N_eq(n*)  per mole per s    (4)
    gamma  = 2 kT / (pi lambda^3 eta)            (Stokes-Einstein, lambda = 0.3 nm)     (5)
    theta  = kT (32 pi v^2 / 3)^{1/3} sigma / (gamma |dmu|^2)   (time lag)          (6)
    dmu    = dH_m dT/T_m * 2T/(T_m + T) / N_A    (Thompson-Spaepen)                  (7)
GST: T_m = 900 K, dH_m = 12.13 kJ/mol, v = 3.06e-29 m^3, sigma = 0.075 J/m^2 (Table 1).
Viscosity: Orava et al. 2012 decoupled form, eta(T) = eta(T_m) (U_kin(T_m)/U_kin(T))^{1/xi}, xi = 0.67,
eta(T_m) = 1.15e-3 Pa s (gives eta(T_g) ~ 1e12 Pa s).

Transient nucleation: Kashchiev's series in reduced time s = ∫ dt / tau(T), tau = 4 theta / pi^2 (so the
asymptotic delay of the cumulative nuclei equals theta):
    I(t) = I_st(T) [1 + 2 sum_{m>=1} (-1)^m exp(-m^2 s)]
JMAK (3D, random nucleation in the transforming volume, interface-controlled growth u(T)):
    X_ext(t) = (4 pi / 3) ∫_0^t I(t') [G(t) - G(t')]^3 dt',   G(t) = ∫_0^t u dt,   X = 1 - exp(-X_ext)
Clusters formed while the material is molten do not survive: integration starts after the last time
above T_m, with the transient restarted there.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .growth import CohenGrestGrowth

KB = 1.380649e-23
NA = 6.02214076e23


@dataclass(frozen=True)
class CNTNucleation:
    T_m: float = 900.0
    dH_m_J_per_mol: float = 12.13e3
    v: float = 3.06e-29
    sigma: float = 0.075
    jump: float = 0.3e-9
    xi: float = 0.67
    eta_Tm: float = 1.15e-3
    growth: CohenGrestGrowth = CohenGrestGrowth()

    def dmu(self, T):
        T = np.asarray(T, float)
        return -self.dH_m_J_per_mol * (self.T_m - T) / self.T_m * 2 * T / (self.T_m + T) / NA

    def eta(self, T):
        return self.eta_Tm * (self.growth.U_kin(self.T_m) / self.growth.U_kin(T)) ** (1 / self.xi)

    def gamma(self, T):
        T = np.asarray(T, float)
        return 2 * KB * T / (math.pi * self.jump ** 3 * self.eta(T))

    def barrier(self, T):
        """W* = 16 pi v^2 sigma^3 / (3 dmu^2) (from eqs. 1-2)."""
        return 16 * math.pi * self.v ** 2 * self.sigma ** 3 / (3 * self.dmu(T) ** 2)

    def I_st_per_mol(self, T):
        T = np.asarray(T, float)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            pref = self.gamma(T) * np.sqrt(self.sigma / (KB * T)) * self.v ** (1 / 3) * 4 * (8 / (81 * math.pi ** 2)) ** (1 / 6)
            out = pref * NA * np.exp(-self.barrier(T) / (KB * T))
        return np.where(T < self.T_m, np.nan_to_num(out, nan=0.0, posinf=0.0), 0.0)

    def I_st(self, T):
        """Steady-state homogeneous nucleation rate per unit volume, m^-3 s^-1."""
        return self.I_st_per_mol(T) / (NA * self.v)

    def time_lag(self, T):
        T = np.asarray(T, float)
        with np.errstate(divide="ignore"):
            return KB * T * (32 * math.pi * self.v ** 2 / 3) ** (1 / 3) * self.sigma / (self.gamma(T) * self.dmu(T) ** 2)


def jmak_crystallised_fraction(t, T, nucl: CNTNucleation, samples_per_segment: int = 32, n_terms: int = 30):
    """Crystallised fraction X(t_end) for a temperature history T(t) starting fully amorphous (no nuclei)."""
    t, T = np.asarray(t, float), np.asarray(T, float)
    u = np.linspace(0, 1, samples_per_segment + 1)[:-1]
    tt = np.concatenate([(t[:-1, None] + u[None, :] * np.diff(t)[:, None]).ravel(), t[-1:]])
    TT = np.interp(tt, t, T)
    molten = np.where(TT >= nucl.T_m)[0]
    start = molten.max() + 1 if molten.size else 0
    if start >= tt.size - 1:
        return 0.0
    tt, TT = tt[start:], TT[start:]
    dt = np.diff(tt)
    Tm_mid = 0.5 * (TT[1:] + TT[:-1])
    tau = 4 * nucl.time_lag(Tm_mid) / math.pi ** 2
    s = np.concatenate([[0.0], np.cumsum(dt / np.where(tau > 0, tau, np.inf))])
    s_mid = 0.5 * (s[1:] + s[:-1])
    m = np.arange(1, n_terms + 1)
    transient = 1 + 2 * np.sum(((-1.0) ** m)[None, :] * np.exp(-np.outer(s_mid, m ** 2)), axis=1)
    transient = np.clip(transient, 0.0, 1.0)
    I = nucl.I_st(Tm_mid) * transient
    u_mid = nucl.growth.velocity(Tm_mid)
    G = np.concatenate([[0.0], np.cumsum(u_mid * dt)])
    G_mid = 0.5 * (G[1:] + G[:-1])
    X_ext = 4 * math.pi / 3 * np.sum(I * dt * (G[-1] - G_mid) ** 3)
    return float(-math.expm1(-X_ext))


@dataclass(frozen=True)
class KineticallyScaledCNT(CNTNucleation):
    """CNT with the attachment kinetics of another material state. Nucleation and growth share the attachment
    kinetics (Stokes-Einstein, gamma ~ 1/eta), so eta is scaled by u_ref(T) / u_target(T), with u_ref the growth
    model behind this CNT (Orava 2012, as-deposited) and u_target e.g. melt-quenched glass (Sebastian 2014).
    I_st falls and the time lag rises by exactly that ratio; the barrier is unchanged."""
    target: object = None

    def eta(self, T):
        T = np.asarray(T, float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio = self.growth.velocity(T) / self.target.velocity(np.minimum(T, self.target.T_m - 1e-6))
        return super().eta(T) * np.where(np.isfinite(ratio) & (ratio > 0), ratio, np.inf)


def transient_integral(S: float) -> float:
    """F(S) = ∫_0^S f(s) ds for Kashchiev's transient f(s) = 1 + 2 sum_m (-1)^m exp(-m^2 s) (s in units of
    tau = 4 theta / pi^2). F -> S - pi^2/6 for large S. For S < 1 f is integrated in its dual (Jacobi theta_4) form
    f(s) = 2 sqrt(pi/s) sum_k exp(-(2k+1)^2 pi^2 / (4s)), exponentially small for small s (incubation)."""
    if S >= 1:                                   # sum_m (-1)^m / m^2 = -pi^2/12 taken exactly; the rest converges fast
        m = np.arange(1, 30)
        return float(S - math.pi ** 2 / 6 - 2 * np.sum((-1.0) ** m * np.exp(-m ** 2 * S) / m ** 2))
    if S <= 0:
        return 0.0
    from .._compat import trapezoid
    s = np.linspace(S * 1e-6, S, 20001)
    k = np.arange(0, 8)
    f = 2 * np.sqrt(np.pi / s) * np.sum(np.exp(-np.outer(1 / s, (2 * k + 1) ** 2) * np.pi ** 2 / 4), axis=1)
    return float(trapezoid(f, s))


def per_pulse_nucleation(nucl: CNTNucleation, t, T, refine: int = 20):
    """For one temperature history: J1 = ∫ I_st dt (nuclei per m^3 at steady state) and s1 = ∫ dt / tau."""
    from .._compat import trapezoid
    t, T = np.asarray(t, float), np.asarray(T, float)
    tt = np.linspace(t[0], t[-1], refine * t.size)
    TT = np.interp(tt, t, T)
    with np.errstate(divide="ignore"):
        tau = 4 * nucl.time_lag(TT) / math.pi ** 2
    inv = np.where((tau > 0) & np.isfinite(tau), 1 / tau, 0.0)
    return float(trapezoid(nucl.I_st(TT), tt)), float(trapezoid(inv, tt))


def repeated_pulse_nuclei(J1: float, s1: float, N: float):
    """Nuclei per m^3 after N identical pulses, (steady-state bound N J1, with transient). Between pulses the
    material is cold and the cluster distribution frozen, so the reduced time accumulates: n = (J1/s1) F(N s1).
    The per-pulse effective values assume I_st and 1/tau peak together (both dominated by the peak temperature).
    Starts from no clusters; a quench that leaves clusters makes the transient value optimistic, the bound not."""
    ss = N * J1
    return ss, (0.0 if s1 <= 0 else (J1 / s1) * transient_integral(N * s1))
