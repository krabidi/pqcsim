import numpy as np
import pytest

from pqcsim.thermal.selfconsistent import solve_selfconsistent


def test_constant_power_reproduces_step_response():
    t = np.linspace(0, 1e-6, 1001)
    tau = 1e-7
    S = 1 - np.exp(-t / tau)                       # first-order system, unit power
    dT, _ = solve_selfconsistent(t, S, lambda T: 3.0, t_pulse=2e-6)
    assert dT == pytest.approx(3.0 * S, abs=1e-12)


def test_rectangular_pulse_then_cooling():
    t = np.linspace(0, 1e-6, 2001)
    tau = 1e-7
    S = 1 - np.exp(-t / tau)
    tp = 2e-7
    dT, _ = solve_selfconsistent(t, S, lambda T: 1.0, t_pulse=tp)
    exact = np.where(t <= tp, 1 - np.exp(-t / tau), (1 - np.exp(-tp / tau)) * np.exp(-(t - tp) / tau))
    assert dT == pytest.approx(exact, abs=2e-3)    # grid-resolution error at the switch-off only


def test_feedback_matches_analytic_first_order_system():
    """dT/dt = (P(T) - T)/tau with P = P0 (1 + a T): exact solution of a linear ODE."""
    tau, P0, a = 1e-7, 1.0, 0.3
    t = np.linspace(0, 5e-7, 20001)
    S = 1 - np.exp(-t / tau)
    dT, _ = solve_selfconsistent(t, S, lambda T: P0 * (1 + a * T), t_pulse=1.0)
    k = (1 - P0 * a) / tau
    exact = P0 / (1 - P0 * a) * (1 - np.exp(-k * t))
    assert dT == pytest.approx(exact, rel=2e-3, abs=1e-6)
