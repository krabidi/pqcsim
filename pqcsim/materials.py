"""Optical, thermal and kinetic description of phase-change materials (PCMs).

Conventions used throughout pqcsim
----------------------------------
- Time dependence exp(-i w t); complex refractive index  n~ = n + i k  with k >= 0 for absorbing media.
- Relative permittivity  eps = n~^2.
- Crystalline fraction x in [0, 1]: x = 0 fully amorphous, x = 1 fully crystalline.

The JSON record format is the one used in the photomem / array scripts, so existing material
files (materials.json, ...) load unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

MIXING_RULES = ("lorentz_lorenz", "linear_eps", "linear_nk")


def mix_index(n_am: complex, n_cr: complex, x: float, rule: str = "lorentz_lorenz") -> complex:
    """Effective complex index of a partially crystallized film.

    lorentz_lorenz : (eps-1)/(eps+2) = x (eps_cr-1)/(eps_cr+2) + (1-x) (eps_am-1)/(eps_am+2)
                     Standard effective-medium rule for PCM partial crystallization.
    linear_eps     : eps = x eps_cr + (1-x) eps_am  (Wiener bound for layers parallel to E).
    linear_nk      : n~ = x n_cr + (1-x) n_am  (used in the original photomem scripts; heuristic).
    """
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"crystalline fraction x={x} outside [0, 1]")
    if rule == "linear_nk":
        return (1.0 - x) * n_am + x * n_cr
    eps_am, eps_cr = n_am**2, n_cr**2
    if rule == "linear_eps":
        eps = (1.0 - x) * eps_am + x * eps_cr
    elif rule == "lorentz_lorenz":
        f = x * (eps_cr - 1) / (eps_cr + 2) + (1.0 - x) * (eps_am - 1) / (eps_am + 2)
        eps = (1 + 2 * f) / (1 - f)
    else:
        raise ValueError(f"unknown mixing rule {rule!r}; choose from {MIXING_RULES}")
    n = np.sqrt(complex(eps))
    # Principal sqrt gives Re(n) >= 0; for passive media Im(n) must also be >= 0.
    if n.imag < 0:
        n = -n
    return complex(n)


@dataclass
class Kinetics:
    """Crystallization (SET) and amorphization (RESET) parameters.

    SET  : first-order JMAK (Avrami n = 1)  dx/dt = k(T) (1 - x),  k(T) = k0 exp(-Ea / kB T),
           active for T_cryst_min <= T < T_melt.
    RESET: phenomenological analog erase between T_reset_on and T_melt (see devices.pcm).
    """

    Ea_eV: float
    k0: float
    T_cryst_min: float
    T_reset_on: float
    T_melt: float
    reset_gamma: float = 1.2
    reset_full_on_melt: bool = False
    dH_fusion_eV: Optional[float] = None   # enthalpy of fusion per atom; enables the driving-force factor

    def __post_init__(self):
        if self.dH_fusion_eV is not None and self.dH_fusion_eV <= 0:
            raise ValueError("dH_fusion_eV must be positive")
        if not (self.T_cryst_min < self.T_melt and self.T_reset_on <= self.T_melt):
            raise ValueError("need T_cryst_min < T_melt and T_reset_on <= T_melt")
        if self.Ea_eV <= 0 or self.k0 <= 0 or self.reset_gamma <= 0:
            raise ValueError("Ea_eV, k0 and reset_gamma must be positive")


@dataclass
class Thermal:
    """Lumped thermal parameters of the active layer: density (kg/m^3), heat capacity (J/kg/K),
    thermal time constant to the heat sink (s)."""

    rho: float
    cp: float
    tau_th: float

    def __post_init__(self):
        if min(self.rho, self.cp, self.tau_th) <= 0:
            raise ValueError("rho, cp and tau_th must be positive")


@dataclass
class PCMMaterial:
    name: str
    active: Dict[str, Any]
    thermal: Optional[Thermal] = None
    kinetics: Optional[Kinetics] = None
    mixing: str = "lorentz_lorenz"
    record: Dict[str, Any] = field(default_factory=dict, repr=False)

    def nk_state(self, state: str, lam_nm: float, channel: str = "probe") -> complex:
        """Complex index of the pure amorphous ('am') or crystalline ('cr') phase."""
        if state not in ("am", "cr"):
            raise ValueError("state must be 'am' or 'cr'")
        mode = str(self.active.get("mode", "constant")).lower()
        if mode == "constant":
            return complex(float(self.active[f"n_{state}"]), float(self.active[f"k_{state}"]))
        if mode == "parametric3":
            block = self.active.get(channel if channel in ("probe", "set", "reset") else "probe", {})
            return complex(float(block[f"n_{state}"]), float(block[f"k_{state}"]))
        if mode == "tabulated":
            lam = np.asarray(self.active["lam_nm"], float)
            n = np.asarray(self.active[f"n_{state}"], float)
            k = np.asarray(self.active[f"k_{state}"], float)
            if lam.ndim != 1 or lam.size < 2 or n.shape != lam.shape or k.shape != lam.shape:
                raise ValueError(f"{self.name}: tabulated data needs matching lam_nm/n/k arrays (>= 2 points)")
            if lam_nm < lam[0] or lam_nm > lam[-1]:
                raise ValueError(f"{self.name}: wavelength {lam_nm} nm outside tabulated range "
                                 f"[{lam[0]}, {lam[-1]}] nm (no extrapolation)")
            return complex(np.interp(lam_nm, lam, n), np.interp(lam_nm, lam, k))
        raise ValueError(f"{self.name}: unknown active.mode {mode!r}")

    def nk(self, x: float, lam_nm: float, channel: str = "probe") -> complex:
        """Effective complex index at crystalline fraction x."""
        return mix_index(self.nk_state("am", lam_nm, channel), self.nk_state("cr", lam_nm, channel),
                         x, self.mixing)

    @classmethod
    def from_record(cls, rec: Dict[str, Any], mixing: str = "lorentz_lorenz",
                    reset_gamma: float = 1.2) -> "PCMMaterial":
        th = rec.get("thermal") or {}
        pc = rec.get("pcm") or {}
        thermal = Thermal(float(th["rho"]), float(th["cp"]), float(th["tau_th"])) if th else None
        kinetics = None
        if pc:
            kinetics = Kinetics(Ea_eV=float(pc["Ea_eV"]), k0=float(pc["k0"]),
                                T_cryst_min=float(pc["T_cryst_min"]), T_reset_on=float(pc["T_reset_on"]),
                                T_melt=float(pc["T_melt"]),
                                reset_gamma=float(pc.get("reset_gamma", reset_gamma)),
                                dH_fusion_eV=float(pc["dH_fusion_eV"]) if "dH_fusion_eV" in pc else None)
        return cls(name=str(rec["name"]), active=dict(rec["active"]), thermal=thermal,
                   kinetics=kinetics, mixing=mixing, record=rec)


def load_materials(path: str, mixing: str = "lorentz_lorenz") -> List[PCMMaterial]:
    """Load a materials JSON file (a list, or {"materials": [...]}); keys starting with '_' are ignored."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("materials", data)
    if not isinstance(data, list):
        raise ValueError("materials JSON must be a list or contain a 'materials' list")
    return [PCMMaterial.from_record(r, mixing=mixing) for r in data if isinstance(r, dict) and "name" in r]
