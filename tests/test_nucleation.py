"""CNT nucleation (Orava & Greer 2017) and non-isothermal JMAK."""
import math

import numpy as np
import pytest

from pqcsim.devices.nucleation import CNTNucleation, jmak_crystallised_fraction


@pytest.fixture
def c():
    return CNTNucleation()


def test_viscosity_reaches_glass_value_at_Tg(c):
    assert c.eta(383.0) == pytest.approx(1e12, rel=0.3)          # Orava 2012: eta(T_g) = 1e12 Pa s


def test_critical_size_and_barrier_consistent(c):
    T = 550.0
    dmu, v, s = c.dmu(T), c.v, c.sigma
    n_star = 32 * math.pi * v ** 2 * s ** 3 / (3 * abs(dmu) ** 3)
    W = lambda n: n * dmu + (36 * math.pi) ** (1 / 3) * v ** (2 / 3) * n ** (2 / 3) * s
    ns = np.linspace(0.5 * n_star, 1.5 * n_star, 2001)
    assert ns[np.argmax(W(ns))] == pytest.approx(n_star, rel=1e-3)     # eq. (2) is the maximum of eq. (1)
    assert W(n_star) == pytest.approx(c.barrier(T), rel=1e-9)


def test_rate_and_time_lag_match_orava_greer_fig3(c):
    """Fig. 3 (read from the log plots, +-1-2 decades): I_st max ~1e24-1e25 /mol/s near T/T_m ~ 0.6;
    ~1e10 at 0.43; ~1e17-1e18 at 0.47; time lag ~1e3-1e4 s at 0.43 and ~1e-9 s above 0.55."""
    T = np.linspace(390, 880, 4000)
    I = c.I_st_per_mol(T)
    assert T[np.argmax(I)] / 900 == pytest.approx(0.60, abs=0.02)
    assert 1e23 < I.max() < 1e26
    assert 1e9 < c.I_st_per_mol(0.43 * 900) < 1e11
    assert 1e16 < c.I_st_per_mol(0.47 * 900) < 1e19
    assert 1e2 < c.time_lag(0.43 * 900) < 1e5
    assert 1e-11 < c.time_lag(0.65 * 900) < 1e-8
    assert c.I_st_per_mol(900.0) == 0.0


def test_isothermal_jmak_against_closed_form(c):
    """Constant T, t >> time lag theta: nuclei appear effectively theta late (Kashchiev), so
    X = 1 - exp(-(pi/3) I u^3 (t - theta)^3 (t + theta/3)) ~ 1 - exp(-(pi/3) I u^3 (t - theta)^4) to O(theta/t)."""
    T = 560.0
    I, u, theta = c.I_st(T), float(c.growth.velocity(T)), c.time_lag(T)
    t_end = (0.7 * 3 / (math.pi * I * u ** 3)) ** 0.25 + theta
    assert theta < 1e-2 * t_end
    t = np.linspace(0, t_end, 800)
    X = jmak_crystallised_fraction(t, np.full_like(t, T), c, samples_per_segment=64)
    X_ref = -math.expm1(-math.pi / 3 * I * u ** 3 * (t_end - theta) ** 4)
    assert X == pytest.approx(X_ref, rel=1e-2)
    X_no_lag = -math.expm1(-math.pi / 3 * I * u ** 3 * t_end ** 4)
    assert abs(X - X_ref) < abs(X - X_no_lag)                   # the transient delay is captured


def test_molten_history_resets_nucleation(c):
    t = np.linspace(0, 200e-9, 400)
    hot = np.where(t < 150e-9, 950.0, 300.0)                     # molten, then frozen far below growth range
    assert jmak_crystallised_fraction(t, hot, c) == pytest.approx(0.0, abs=1e-12)


def test_transient_integral_limits_and_brute_force():
    from pqcsim._compat import trapezoid
    from pqcsim.devices.nucleation import transient_integral
    m = np.arange(1, 4001)
    for S in (0.2, 0.5, 0.999, 3.0):
        s = np.linspace(1e-4, S, 40001)
        f = 1 + 2 * np.sum(((-1.0) ** m)[None, :] * np.exp(-np.outer(s, m ** 2)), axis=1)
        assert transient_integral(S) == pytest.approx(trapezoid(np.clip(f, 0, None), s), rel=2e-3)
    assert transient_integral(50.0) == pytest.approx(50.0 - math.pi ** 2 / 6, rel=1e-12)
    assert 0 < transient_integral(0.05) < 1e-20                               # incubation
    a, b = transient_integral(0.999999), transient_integral(1.0)
    assert b == pytest.approx(a, rel=1e-4)                                   # branches join


def test_kinetic_scaling_and_repeated_pulses():
    from pqcsim.devices.growth import sebastian2014_gst
    from pqcsim.devices.nucleation import KineticallyScaledCNT, per_pulse_nucleation, repeated_pulse_nuclei
    c = CNTNucleation()
    q = KineticallyScaledCNT(target=sebastian2014_gst())
    for T in (420.0, 480.0):
        r = c.growth.velocity(T) / q.target.velocity(T)
        assert c.I_st(T) / q.I_st(T) == pytest.approx(r, rel=1e-9)
        assert q.time_lag(T) / c.time_lag(T) == pytest.approx(r, rel=1e-9)
    t = np.linspace(0, 1e-6, 40)
    T = np.full_like(t, 470.0)
    J1, s1 = per_pulse_nucleation(c, t, T)
    assert J1 == pytest.approx(c.I_st(470.0) * 1e-6, rel=1e-9)
    assert s1 == pytest.approx(1e-6 * math.pi ** 2 / (4 * c.time_lag(470.0)), rel=1e-9)
    # non-isothermal pulses with frozen clusters in between: the per-pulse transient equals the direct integration
    # of I_st(T) f(s(t)) over the concatenated history, also in the incubation regime where N J1 overshoots
    tp = 1e-7
    t1 = np.linspace(0, tp, 2001)
    T1 = 380 + 60 * np.sin(np.pi * t1 / tp)
    J1, s1 = per_pulse_nucleation(c, t1, T1)
    dt, Tm = np.diff(t1), 0.5 * (T1[1:] + T1[:-1])
    I, ds = c.I_st(Tm), math.pi ** 2 / (4 * c.time_lag(Tm)) * np.diff(t1)
    m, k = np.arange(1, 80), np.arange(8)
    for N in (33, 97):
        ref, s0 = 0.0, 0.0
        for _ in range(N):
            s = s0 + np.cumsum(ds) - ds / 2
            fd = 1 + 2 * np.sum(((-1.0) ** m)[None, :] * np.exp(-np.outer(s, m ** 2)), axis=1)
            fs = 2 * np.sqrt(math.pi / s) * np.sum(np.exp(-np.outer(1 / s, (2 * k + 1) ** 2) * math.pi ** 2 / 4), axis=1)
            ref += np.sum(I * np.where(s < 1, fs, fd) * dt)
            s0 += ds.sum()
        ss, tr = repeated_pulse_nuclei(J1, s1, N)
        assert tr == pytest.approx(ref, rel=1e-2)
        assert ss > tr
