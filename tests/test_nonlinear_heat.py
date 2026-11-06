"""Enthalpy-method solver: linear limit, energy, Stefan problem, temperature-dependent conductivity."""
import math

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import erf

from pqcsim.thermal.diffusion import StackHeatSolver, ThermalLayer, radial_grid
from pqcsim.thermal.nonlinear import NonlinearStackHeatSolver, PhaseChangeLayer


def test_linear_limit_matches_linear_solver():
    lin = [ThermalLayer("pcm", 20, 0.5, 6140, 220, 3), ThermalLayer("ox", 200, 1.4, 2200, 740, 6)]
    nl = [PhaseChangeLayer("pcm", 20, 0.5, 6140, 220, 3), PhaseChangeLayer("ox", 200, 1.4, 2200, 740, 6)]
    rf = radial_grid(5, 0.05, 1.1)
    a = StackHeatSolver(lin, rf)
    b = NonlinearStackHeatSolver(nl, rf)
    Q = a.gaussian_source(0, 30.0, 0.8)
    ra = a.run(Q, 50e-9, 300e-9, rtol=1e-10)
    rb = b.run(Q, 50e-9, 300e-9, rtol=1e-8, method="Radau")
    # compare at identical output times (interpolating across the pulse-end kink would add error)
    common = np.intersect1d(np.round(ra.t, 18), np.round(rb.t, 18))
    assert common.size > 50
    Ta = ra.layer_average(0, 0.0)[np.isin(np.round(ra.t, 18), common)]
    Tb = rb.layer_average(0, 0.0)[np.isin(np.round(rb.t, 18), common)]
    assert Tb == pytest.approx(Ta, abs=1e-6 * Ta.max())
    # the default phase-change stepper (full-Newton BDF2, step control on dT) on the same linear problem
    rn = b.run(Q, 50e-9, 300e-9, dT_step_K=0.5)
    Tn = np.interp(ra.t, rn.t, rn.layer_average(0, 0.0))
    assert np.max(np.abs(Tn - ra.layer_average(0, 0.0))) < 0.01 * Ta.max()


def test_energy_conserved_through_melting():
    layers = [PhaseChangeLayer("pcm", 20, 0.5, 6140, 220, 4, k_liquid=1.5, latent_J_per_kg=2.0e5, T_melt=900.0,
                               melt_width_K=5.0),
              PhaseChangeLayer("ox", 100, 1.4, 2200, 740, 4)]
    s = NonlinearStackHeatSolver(layers, radial_grid(3, 0.25, 1.2), top="insulated", bottom="insulated",
                                 outer="insulated")
    Q = s.geo.gaussian_source(0, 120.0, 1.0)             # short, strong pulse: the film melts and refreezes
    res = s.run(Q, 5e-9, 40e-9, n_out=60)
    assert res.dT.max() + 300 > 900                       # it really melts
    assert s.total_enthalpy(res.H[-1]) == pytest.approx(Q.sum(), rel=1e-6)


def test_one_phase_stefan_problem():
    """Solid at T_m, surface held at T_w > T_m: front s(t) = 2 lam sqrt(alpha t),
    lam exp(lam^2) erf(lam) = St / sqrt(pi), St = cp (T_w - T_m) / L  (liquid properties)."""
    rho, cp, k, L, Tm, Tw = 6000.0, 220.0, 1.0, 2.0e5, 900.0, 1000.0
    D, w = 150.0, 2.0                                # 2 nm cells; front moves 28-56 nm
    layer = PhaseChangeLayer("pcm", D, k, rho, cp, cells=75, k_liquid=k, latent_J_per_kg=L, T_melt=Tm,
                             melt_width_K=w)
    # T0 = T_w so the 'fixed' top boundary is the hot wall. The solid starts 8 smoothing widths below T_m
    # (liquid fraction ~3e-4). Measured convergence with refinement (dz, w): 1.8 % -> 1.3 % -> 0.9 % at 20 ns.
    s = NonlinearStackHeatSolver([layer], np.array([0.0, 1e-6]), T0=Tw, top="fixed", bottom="insulated",
                                 outer="insulated")
    res = s.run(np.zeros((s.nz, s.nr)), 0.0, 20e-9, n_out=20, T_init=np.full(s.nz * s.nr, Tm - 8 * w))
    St = cp * (Tw - Tm) / L
    lam = brentq(lambda l: l * math.exp(l * l) * erf(l) - St / math.sqrt(math.pi), 1e-6, 5)
    alpha = k / (rho * cp)
    z = res.z_centres
    for t in (5e-9, 10e-9, 20e-9):
        i = np.argmin(abs(res.t - t))
        f = s.liquid_fraction(res.dT[i].ravel() + Tw).reshape(s.nz, s.nr)[:, 0]
        front = np.sum(f * s.geo.dz)                       # melted thickness
        assert front == pytest.approx(2 * lam * math.sqrt(alpha * res.t[i]), rel=0.03)


def test_temperature_dependent_conductivity_kirchhoff():
    """Steady flux q through a slab with k = k0 (1 + b (T - T0)), cold side at T0:
    q L = k0 (dT + b dT^2 / 2)  ->  dT = (sqrt(1 + 2 b q L / k0) - 1) / b."""
    k0, b, Lnm, q = 1.4, 2e-3, 500.0, 2e9
    heater = PhaseChangeLayer("h", 1, 400.0, 2000, 100, cells=1)
    slab = PhaseChangeLayer("s", Lnm, k0, 2200, 740, cells=40, k_beta_per_K=b)
    s = NonlinearStackHeatSolver([heater, slab], radial_grid(1, 0.5, 1.0), top="insulated", bottom="fixed",
                                 outer="insulated")
    Q = np.zeros((s.nz, s.nr))
    Q[0, :] = q * s.geo.V[0, :] / 1e-9 * 1e-6                # power q per area for a 1 us pulse
    res = s.run(Q, 1e-6, 1.001e-6)
    i = np.argmin(abs(res.t - 1e-6))
    # first slab cell centre sits dz/2 below the top: U(T(z)) = q (L - z)
    z = 0.5 * Lnm * 1e-9 / 40
    U = q * (Lnm * 1e-9 - z)
    dT_expected = (math.sqrt(1 + 2 * b * U / k0) - 1) / b
    assert res.dT[i, 1, 0] == pytest.approx(dT_expected, rel=1e-3)
