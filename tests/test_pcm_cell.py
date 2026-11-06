"""PCM memristor cell: thermal model, kinetics vs independent ODE integration, legacy compatibility."""
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pqcsim.devices.pcm import KB_EV, PCMCell


def test_active_absorption_is_tmm_layer_value(cell):
    res = cell.optics(0.5)
    assert cell.active_absorption(0.5, 1550) == pytest.approx(res.A_layers[1])
    assert res.R + res.T + res.A_total == pytest.approx(1.0, abs=1e-12)
    assert cell.weight(0.5) == pytest.approx(1 - res.R)


def test_temperature_limits(cell):
    th = cell.material.thermal
    C = th.rho * th.cp * cell.d_active_nm * 1e-9
    E = 2.0
    # Short pulse (t_p << tau): adiabatic heating, T_peak = T0 + E / C
    assert cell.temperatures(E, 1e-14)[0] == pytest.approx(cell.T0 + E / C, rel=1e-5)
    # Long pulse (t_p >> tau): steady state T0 + P R
    tp = 1e3 * th.tau_th
    assert cell.temperatures(E, tp)[0] == pytest.approx(cell.T0 + E / tp * th.tau_th / C, rel=1e-9)


def _ode_reference(cell, x0, E_abs, tp):
    """Integrate C dT/dt = P - (T-T0)/R together with dx/dt = k(T)(1-x) numerically."""
    th, kin = cell.material.thermal, cell.material.kinetics
    C = th.rho * th.cp * cell.d_active_nm * 1e-9
    R = th.tau_th / C

    def rhs(t, y):
        T, x = y
        P = E_abs / tp if t < tp else 0.0
        k = kin.k0 * math.exp(-kin.Ea_eV / (KB_EV * T)) if kin.T_cryst_min <= T < kin.T_melt else 0.0
        return [(P - (T - cell.T0) / R) / C, k * (1 - x)]

    y = [cell.T0, x0]
    for span in [(0, tp), (tp, tp + 40 * th.tau_th)]:
        sol = solve_ivp(rhs, span, y, method="LSODA", rtol=1e-10, atol=1e-12)
        y = sol.y[:, -1]
    return y[1]


@pytest.mark.parametrize("T_peak,tp", [(520.0, 100e-9), (600.0, 100e-9), (650.0, 10e-9), (640.0, 1e-6)])
def test_set_kinetics_match_ode(cell, E_for_peak, T_peak, tp):
    E_abs = E_for_peak(T_peak, tp)
    out = cell.apply_pulse(0.1, 0.0, 1550, tp, E_abs=E_abs)
    assert out.T_peak == pytest.approx(T_peak, rel=1e-10)
    assert out.T_peak < cell.material.kinetics.T_reset_on        # pure SET regime
    assert out.x > 0.1 + 1e-6                                     # crystallization actually happened
    assert out.x == pytest.approx(_ode_reference(cell, 0.1, E_abs, tp), abs=1e-6)


def test_no_change_below_crystallization_threshold(cell, E_for_peak):
    out = cell.apply_pulse(0.3, 0.0, 1550, 100e-9, E_abs=E_for_peak(445.0, 100e-9))
    assert out.T_peak < cell.material.kinetics.T_cryst_min
    assert out.x == 0.3


def test_reset_erase_rule(cell):
    kin = cell.material.kinetics
    cell_fast = PCMCell(cell.material, cell.layers, cell.thicknesses_nm, readout="1-R",
                        model="photomem_legacy")
    th = cell.material.thermal
    C = th.rho * th.cp * cell.d_active_nm * 1e-9
    T_target = 0.5 * (kin.T_reset_on + kin.T_melt)
    E = (T_target - cell.T0) * C / -math.expm1(-10e-9 / th.tau_th) * 10e-9 / th.tau_th
    out = cell_fast.apply_pulse(0.8, 0.0, 1550, 10e-9, kind="reset", E_abs=E)
    u = (out.T_peak - kin.T_reset_on) / (kin.T_melt - kin.T_reset_on)
    assert out.T_peak == pytest.approx(T_target, rel=1e-9)
    assert out.x == pytest.approx(0.8 * (1 - u ** kin.reset_gamma), rel=1e-12)


def test_legacy_set_formula(cell):
    legacy = PCMCell(cell.material, cell.layers, cell.thicknesses_nm, readout="1-R", model="photomem_legacy")
    kin = cell.material.kinetics
    out = legacy.apply_pulse(0.2, 0.0, 1550, 100e-9, kind="set", E_abs=1.5)
    k = kin.k0 * math.exp(-kin.Ea_eV / (KB_EV * out.T_avg))
    assert out.x == pytest.approx(0.2 + 0.8 * (1 - math.exp(-k * 100e-9)), rel=1e-12)


def _quench_reference(cell, T_peak):
    """Cooling-tail crystallization of a melted, fully erased film, integrated over TEMPERATURE.

    On T(t) = T0 + (T_peak - T0) e^{-t/tau},  dt = -tau dT / (T - T0), so
        -ln(1 - x) = ∫_{T_cryst_min}^{min(T_peak, T_melt)} k(T) tau / (T - T0) dT
    (a different integration route from the time-domain quadrature in the code).
    """
    from scipy.integrate import quad
    th, kin = cell.material.thermal, cell.material.kinetics
    hi = min(T_peak, kin.T_melt)
    I, _ = quad(lambda T: cell._rate(T) * th.tau_th / (T - cell.T0), kin.T_cryst_min, hi,
                limit=400, epsrel=1e-11, epsabs=0.0)
    return -math.expm1(-I)


@pytest.mark.parametrize("dH", [None, 0.2])
def test_melting_pulse_erases_then_quench_recrystallizes(gete, E_for_peak, dH):
    from dataclasses import replace
    mat = replace(gete, kinetics=replace(gete.kinetics, dH_fusion_eV=dH))
    cell = PCMCell(mat, [1.45, "active", 1.45, 0.55 + 11.5j], [20.0, 20.0, 270.0, 100.0], readout="1-R")
    out = cell.apply_pulse(1.0, 0.0, 1550, 10e-9, E_abs=E_for_peak(1100.0, 10e-9))
    assert out.T_peak > mat.kinetics.T_melt
    assert out.erased == pytest.approx(1.0)
    assert out.x == pytest.approx(_quench_reference(cell, out.T_peak), abs=1e-6)


def test_fast_quench_material_stays_amorphous(E_for_peak):
    from pqcsim.materials import load_materials
    from tests.conftest import ROOT
    gst = {m.name: m for m in load_materials(str(ROOT / "data" / "materials.json"))}["GST225"]
    cell = PCMCell(gst, [1.45, "active", 1.45, 0.55 + 11.5j], [20.0, 20.0, 270.0, 100.0], readout="1-R")
    out = cell.apply_pulse(1.0, 0.0, 1550, 10e-9, E_abs=E_for_peak(1000.0, 10e-9))
    assert out.erased == pytest.approx(1.0) and out.x < 0.05


def test_driving_force_vanishes_at_melt(gete):
    """Near T_m the Wilson-Frenkel factor -> dG/kT = dH (T_m - T) / (T_m kB T), i.e. linear in undercooling."""
    from dataclasses import replace
    mat = replace(gete, kinetics=replace(gete.kinetics, dH_fusion_eV=0.2))
    with_df = PCMCell(mat, [1.45, "active", 1.45, 0.55 + 11.5j], [20.0, 20.0, 270.0, 100.0], readout="1-R")
    plain = PCMCell(gete, [1.45, "active", 1.45, 0.55 + 11.5j], [20.0, 20.0, 270.0, 100.0], readout="1-R")
    Tm = mat.kinetics.T_melt
    for dT in (1e-3, 1e-1):
        T = Tm - dT
        assert with_df._rate(T) / plain._rate(T) == pytest.approx(0.2 * dT / (Tm * KB_EV * T), rel=1e-3)


@pytest.mark.parametrize("T_peak", [560.0, 640.0, 760.0])
def test_temperature_history_matches_rc_pulse(cell, E_for_peak, T_peak):
    """Feeding the analytic RC transient as a sampled history must reproduce apply_pulse."""
    th = cell.material.thermal
    tp = 100e-9
    E = E_for_peak(T_peak, tp)
    out = cell.apply_pulse(0.05, 0.0, 1550, tp, E_abs=E)
    tau = th.tau_th
    dTss = (out.T_peak - cell.T0) / -math.expm1(-tp / tau)
    t_heat = np.linspace(0, tp, 400)
    t_cool = tp + np.geomspace(1e-12, 40 * tau, 400)
    t = np.concatenate([t_heat, t_cool])
    T = np.concatenate([cell.T0 + dTss * -np.expm1(-t_heat / tau),
                        cell.T0 + (out.T_peak - cell.T0) * np.exp(-(t_cool - tp) / tau)])
    hist = cell.apply_temperature_history(0.05, t, T)
    assert hist.T_peak == pytest.approx(out.T_peak, rel=1e-9)
    assert hist.x == pytest.approx(out.x, abs=2e-4)
    assert hist.erased == pytest.approx(out.erased, abs=1e-9)
