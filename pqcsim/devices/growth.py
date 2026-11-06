"""Non-Arrhenius crystal growth of phase-change materials (fragile supercooled liquid + glass).

Growth velocity (Salinga et al., Nat. Commun. 4, 2371 (2013), eq. 1):
    u(T) = 4 r_atom k_B T / (3 pi lambda^2 R_hyd eta(T)) * [1 - exp(-dG(T) / k_B T)]
Thompson-Spaepen driving force (eq. 2):  dG = dH_m (T_m - T)/T_m * 2T/(T_m + T)
Supercooled-liquid viscosity, MYEGA (Mauro et al., PNAS 106, 19780 (2009)):
    log10 eta = log10 eta_inf + (12 - log10 eta_inf) (T_g/T) exp[(m / (12 - log10 eta_inf) - 1)(T_g/T - 1)]
Melt-quenched glass: Arrhenius growth u_g(T) = u_ref exp(-E_a/k_B (1/T - 1/T_ref)), converted to a glass
viscosity by inverting eq. 1. The glass exists only below the temperature T_x where it joins the
supercooled liquid (the paper fits MYEGA to the highest-temperature data, so the curves touch there);
T_x is taken as their point of closest approach. eta = eta_glass for T < T_x, eta_liquid above.
(A min() rule would be wrong: a fragile liquid flattens at high T, so the glass extrapolation would
stay below it and give unphysical speeds.)
Growth-dominated recrystallisation of a disc-shaped amorphous mark of radius R0 from its rim:
    R(t) = R(0) - ∫ u(T(t)) dt,   amorphous fraction (1 - x) = (R / R0)^2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

KB = 1.380649e-23
EV = 1.602176634e-19


@dataclass(frozen=True)
class GrowthModel:
    T_m: float                      # melting temperature, K
    dH_m_eV: float                  # heat of fusion per atom, eV
    T_g: float                      # glass transition temperature, K
    fragility: float                # kinetic fragility m
    eta_inf: float                  # MYEGA viscosity at infinite temperature, Pa s
    r_atom: float = 1.5e-10         # m
    jump: float = 1.0e-10           # diffusional jump distance lambda, m
    R_hyd: float = 0.5e-10          # hydrodynamic radius, m
    glass_Ea_eV: Optional[float] = None    # Arrhenius growth of the melt-quenched glass
    glass_u_ref: Optional[float] = None    # m/s at glass_T_ref
    glass_T_ref: Optional[float] = None
    T_x_fixed: Optional[float] = None       # glass -> liquid transition if stated by the source (else closest approach)

    def driving_force(self, T):
        """Thompson-Spaepen dG(T) per atom, J."""
        T = np.asarray(T, float)
        return self.dH_m_eV * EV * (self.T_m - T) / self.T_m * 2 * T / (self.T_m + T)

    def eta_liquid(self, T):
        T = np.asarray(T, float)
        a = math.log10(self.eta_inf)
        with np.errstate(over="ignore"):            # far below T_g the liquid viscosity overflows to inf: u -> 0
            return 10 ** (a + (12 - a) * (self.T_g / T) * np.exp((self.fragility / (12 - a) - 1) * (self.T_g / T - 1)))

    def _prefactor(self, T):
        return 4 * self.r_atom * KB * np.asarray(T, float) / (3 * math.pi * self.jump ** 2 * self.R_hyd)

    def _thermo(self, T):
        T = np.asarray(T, float)
        return -np.expm1(-self.driving_force(T) / (KB * T))

    def u_glass_arrhenius(self, T):
        if self.glass_Ea_eV is None:
            raise ValueError("no glass Arrhenius data")
        T = np.asarray(T, float)
        return self.glass_u_ref * np.exp(-self.glass_Ea_eV * EV / KB * (1 / T - 1 / self.glass_T_ref))

    def eta_glass(self, T):
        """Glass viscosity implied by the measured Arrhenius growth (eq. 1 inverted)."""
        return self._prefactor(T) * self._thermo(T) / self.u_glass_arrhenius(T)

    @property
    def T_x(self) -> float:
        """Glass -> supercooled-liquid junction: closest approach of the two viscosity curves."""
        if self.glass_Ea_eV is None:
            return self.T_g
        if self.T_x_fixed is not None:
            return self.T_x_fixed
        T = np.linspace(self.T_g + 1, self.T_m - 1, 4000)
        return float(T[np.argmax(np.log(self.eta_glass(T)) - np.log(self.eta_liquid(T)))])

    def eta(self, T):
        T = np.asarray(T, float)
        eta_l = self.eta_liquid(T)
        if self.glass_Ea_eV is None:
            return eta_l
        return np.where(T < self.T_x, self.eta_glass(T), eta_l)

    def velocity(self, T):
        """Crystal growth velocity u(T), m/s (0 at and above T_m)."""
        T = np.asarray(T, float)
        u = self._prefactor(T) * self._thermo(T) / self.eta(T)
        return np.where(T < self.T_m, u, 0.0)

    def regrow_mark(self, x: float, R0_m: float, t: np.ndarray, T: np.ndarray,
                    samples_per_segment: int = 64) -> float:
        """Crystalline fraction after a temperature history, for a disc mark regrowing from its rim."""
        if not 0 <= x <= 1 or R0_m <= 0:
            raise ValueError("need 0 <= x <= 1 and R0 > 0")
        t, T = np.asarray(t, float), np.asarray(T, float)
        u = np.linspace(0, 1, samples_per_segment + 1)[:-1]
        tt = np.concatenate([(t[:-1, None] + u[None, :] * np.diff(t)[:, None]).ravel(), t[-1:]])
        uu = self.velocity(np.interp(tt, t, T))
        from .._compat import trapezoid
        R = R0_m * math.sqrt(1 - x) - trapezoid(uu, tt)
        return 1.0 - (max(R, 0.0) / R0_m) ** 2


def sebastian2014_gst() -> GrowthModel:
    """Melt-quenched Ge2Sb2Te5 measured inside PCM cells, Sebastian et al., Nat. Commun. 5, 4314 (2014):
    glass branch measured isothermally (414-552 K): v = 2.4e24 exp(-3.01 eV / kB T) m/s, valid below
    T_glass = 534 K; supercooled liquid: eq. 1 with MYEGA T_g = 472 K, m = 140, eta_inf = 0.012 Pa s,
    lambda = 0.299 nm, r_atom = R_hyd = 0.1365 nm, T_melt = 877 K, dH_m = 152.4 meV/atom.
    Measured temperature of maximum growth: T_cryst = 749-752 K."""
    Ea = 3.01
    T_ref = 534.0
    u_ref = 2.4e24 * math.exp(-Ea * EV / (KB * T_ref))
    return GrowthModel(T_m=877.0, dH_m_eV=0.1524, T_g=472.0, fragility=140.0, eta_inf=0.012,
                       r_atom=0.1365e-9, jump=0.299e-9, R_hyd=0.1365e-9,
                       glass_Ea_eV=Ea, glass_u_ref=u_ref, glass_T_ref=T_ref, T_x_fixed=534.0)


def salinga2013_aist() -> GrowthModel:
    """Melt-quenched AgInSbTe, all parameters as stated in Salinga et al., Nat. Commun. 4, 2371 (2013):
    eq. 1 constants, dH_m = 173 meV/atom, T_m = 808 K, MYEGA fit m = 135, T_g = 445 K, eta_inf = 1.22e-3 Pa s,
    glass growth Arrhenius with E_a = 2.7 eV and u = 3.4 m/s at 553 K."""
    return GrowthModel(T_m=808.0, dH_m_eV=0.173, T_g=445.0, fragility=135.0, eta_inf=1.22e-3,
                       glass_Ea_eV=2.7, glass_u_ref=3.4, glass_T_ref=553.0)


R_GAS = 8.314462618


@dataclass(frozen=True)
class CohenGrestGrowth:
    """Crystal growth of supercooled-liquid GST after Orava et al., Nat. Mater. 11, 279 (2012):
        log10 U_kin = A - 2B / (T - T0 + sqrt((T - T0)^2 + 4 C T))          (eq. 1, Cohen-Grest form)
        U = U_kin [1 - exp(-dG / R T)]                                     (eq. 2, continuous growth)
        dG: Thompson-Spaepen with molar dH_m and T_m
    A is fixed by the absolute scale U_kin(T_m) from Stokes-Einstein (SI eqs. S7-S8:
    U_kin = k T / (3 pi a^2 eta) with a = 0.3 nm and eta(T_m) = 1.1-1.2e-3 Pa s -> 13.5 m/s)."""
    B: float = 121.0
    C: float = 1.71
    T0: float = 427.0
    T_m: float = 900.0
    dH_m_J_per_mol: float = 12.13e3
    U_kin_Tm: float = 13.5

    def _f(self, T):
        T = np.asarray(T, float)
        return -2 * self.B / (T - self.T0 + np.sqrt((T - self.T0) ** 2 + 4 * self.C * T))

    def U_kin(self, T):
        return self.U_kin_Tm * 10 ** (self._f(T) - self._f(self.T_m))

    def driving_force_molar(self, T):
        T = np.asarray(T, float)
        return self.dH_m_J_per_mol * (self.T_m - T) / self.T_m * 2 * T / (self.T_m + T)

    def velocity(self, T):
        T = np.asarray(T, float)
        u = self.U_kin(T) * -np.expm1(-self.driving_force_molar(T) / (R_GAS * T))
        return np.where(T < self.T_m, u, 0.0)

    regrow_mark = GrowthModel.regrow_mark


def stokes_einstein_U_kin(T: float, eta: float, a: float = 0.3e-9) -> float:
    """U_kin = D / a with D = k T / (3 pi a eta) (Orava et al. SI eqs. S7-S8)."""
    return KB * T / (3 * math.pi * a * a * eta)


def orava2012_gst() -> CohenGrestGrowth:
    """Supercooled-liquid Ge2Sb2Te5, parameters from Orava et al., Nat. Mater. 11, 279 (2012) and its SI."""
    return CohenGrestGrowth()
