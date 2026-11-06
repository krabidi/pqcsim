"""Non-Arrhenius growth model: reproduce the statements of Salinga et al., Nat. Commun. 4, 2371 (2013)."""
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from pqcsim.devices.growth import KB, GrowthModel, salinga2013_aist


@pytest.fixture
def aist():
    return salinga2013_aist()


def test_driving_force_range_matches_paper(aist):
    """Paper: 'in the temperature range probed ... 0.9 < dG(T)/kBT < 1.6' (418-553 K)."""
    T = np.linspace(418, 553, 500)
    r = aist.driving_force(T) / (KB * T)
    assert 0.9 < r.min() < r.max() < 1.6


def test_myega_definition(aist):
    assert aist.eta_liquid(445.0) == pytest.approx(1e12, rel=1e-12)       # eta(T_g) = 10^12 Pa s
    h = 1e-5                                                               # m = d log10 eta / d(T_g/T) at T_g
    m = (math.log10(aist.eta_liquid(445 / (1 + h))) - math.log10(aist.eta_liquid(445 / (1 - h)))) / (2 * h)
    assert m == pytest.approx(135.0, rel=1e-6)
    assert aist.eta_liquid(1e9) == pytest.approx(1.22e-3, rel=1e-3)        # eta -> eta_inf


def test_measured_glass_growth_and_extrapolation(aist):
    # measured: ~100 nm/s at 418 K ... 3.4 m/s at 553 K, Arrhenius 2.7 eV ("eight orders of magnitude")
    assert aist.u_glass_arrhenius(553.0) == pytest.approx(3.4)
    assert 10e-9 < aist.u_glass_arrhenius(418.0) < 300e-9
    # paper: extending the Arrhenius law, 1000 m/s would be reached 'at around 625 K'
    T_sound = brentq(lambda T: aist.u_glass_arrhenius(T) - 1000.0, 553, 800)
    assert T_sound == pytest.approx(625.0, abs=15.0)


def test_maximum_growth_velocity_within_measured_bounds(aist):
    """Paper (two-pulse experiment): maximum growth velocity between 10 and 100 m/s."""
    T = np.linspace(450, 807, 8000)
    u = aist.velocity(T)
    assert 10.0 <= u.max() <= 100.0
    assert aist.T_x == pytest.approx(532, abs=20)          # glass joins liquid at the top of the data
    assert aist.velocity(808.0) == 0.0


def test_regrowth_of_mark_at_constant_temperature(aist):
    """Isothermal rim growth: R = R0 - u t, so 1 - x = (1 - u t / R0)^2."""
    R0, T = 300e-9, 500.0
    u = float(aist.velocity(T))
    t_end = 0.5 * R0 / u
    x = aist.regrow_mark(0.0, R0, np.array([0.0, t_end]), np.array([T, T]))
    assert x == pytest.approx(1 - 0.5 ** 2, rel=1e-9)
    assert aist.regrow_mark(0.0, R0, np.array([0.0, 3 * t_end]), np.array([T, T])) == 1.0


def test_fragile_kinetics_are_non_arrhenius(aist):
    """Apparent activation energy of u(T) falls strongly above the glass-liquid junction."""
    def Ea(T, dT=1.0):
        return KB * (math.log(aist.velocity(T + dT)) - math.log(aist.velocity(T - dT))) / (1 / (T - dT) - 1 / (T + dT)) / 1.602e-19
    assert Ea(480.0) == pytest.approx(2.7, rel=0.05)        # glass branch: the measured 2.7 eV
    assert Ea(650.0) < 1.0


# ----------------------------------------------------------------------------- GST, Orava et al. 2012
def test_orava_stokes_einstein_scale():
    """SI: a = 0.3 nm and eta(T_m) = 1.1-1.2e-3 Pa s give U_kin(T_m) = 13.5 m/s."""
    from pqcsim.devices.growth import stokes_einstein_U_kin
    lo, hi = stokes_einstein_U_kin(900.0, 1.2e-3), stokes_einstein_U_kin(900.0, 1.1e-3)
    assert lo < 13.5 * 1.02 and hi > 13.5 * 0.98


def test_orava_growth_maximum_reproduces_paper():
    """Paper: maximum of U at 0.76 T_m; Fig. 5: U_max just above 3 m/s."""
    from pqcsim.devices.growth import orava2012_gst
    g = orava2012_gst()
    T = np.linspace(390, 899, 20000)
    u = g.velocity(T)
    k = int(np.argmax(u))
    assert T[k] / g.T_m == pytest.approx(0.76, abs=0.01)
    assert 2.8 < u[k] < 3.5
    assert g.U_kin(900.0) == pytest.approx(13.5)


def test_orava_kinetics_strongly_non_arrhenius():
    from pqcsim.devices.growth import orava2012_gst
    g = orava2012_gst()
    def Ea(T, d=0.5):
        return R_GAS_EV * (math.log(g.U_kin(T + d)) - math.log(g.U_kin(T - d))) / (1 / (T - d) - 1 / (T + d))
    assert Ea(450.0) > 3 * Ea(700.0)           # fragile: apparent activation energy collapses at high T


R_GAS_EV = 8.617333262e-5


# ----------------------------------------------------------------------------- melt-quenched GST, Sebastian 2014
def test_sebastian_glass_branch_and_measured_maximum():
    from pqcsim.devices.growth import sebastian2014_gst
    g = sebastian2014_gst()
    for T in (414.0, 483.0, 530.0):                    # measured isothermal range, glass branch
        assert g.velocity(T) == pytest.approx(2.4e24 * math.exp(-3.01 / (8.617333262e-5 * T)), rel=1e-6)
    T = np.linspace(540, 876, 20000)
    k = int(np.argmax(g.velocity(T)))
    assert T[k] == pytest.approx(750.0, abs=15.0)       # measured T_cryst = 749-752 K
    assert g.velocity(877.0) == 0.0
