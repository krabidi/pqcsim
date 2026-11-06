"""Heat-diffusion solver against analytic solutions and conservation laws."""
import math

import numpy as np
import pytest

from pqcsim.thermal.diffusion import StackHeatSolver, ThermalLayer, radial_grid

SIO2 = dict(k=1.4, rho=2200.0, cp=740.0)


def test_energy_conserved_with_insulated_boundaries():
    layers = [ThermalLayer("a", 20, 0.5, 6000, 220), ThermalLayer("b", 200, **SIO2), ThermalLayer("c", 100, 300, 19300, 129)]
    s = StackHeatSolver(layers, radial_grid(10, 0.05), top="insulated", bottom="insulated", outer="insulated")
    Q = s.gaussian_source(0, E_abs_peak=50.0, w_um=1.0)
    res = s.run(Q, 10e-9, 200e-9)
    E_in = 0.5 * math.pi * (1e-6) ** 2 * 50.0 * (1 - math.exp(-2 * 10 ** 2))   # ∫ E(r) 2 pi r dr to r_max
    assert Q.sum() == pytest.approx(E_in, rel=1e-12)
    assert res.heat_content()[-1] == pytest.approx(E_in, rel=1e-5)
    i = np.searchsorted(res.t, 5e-9)
    assert res.heat_content()[i] == pytest.approx(E_in * res.t[i] / 10e-9, rel=1e-4)   # linear deposit


def test_radial_gaussian_spreading_in_insulated_film():
    """2D diffusion of a Gaussian:  dT = A s0^2/(s0^2 + 2Dt) exp(-r^2 / (2 (s0^2 + 2Dt))),  s0 = w/2.

    Compared as averages over each annular cell (what a finite-volume solver computes):
    the ring average of exp(-r^2 / 2 sig) over [ra, rb] is 2 sig (e^{-ra^2/2sig} - e^{-rb^2/2sig}) / (rb^2 - ra^2).
    """
    L = ThermalLayer("film", 50, 1.4, 2200, 740, cells=2)
    D = 1.4 / (2200 * 740)
    w = 1.0e-6
    s = StackHeatSolver([L], radial_grid(15, 0.02, 1.04), top="insulated", bottom="insulated", outer="insulated")
    res = s.run(s.gaussian_source(0, 1.0, w * 1e6), 0.0, 400e-9)
    A = 1.0 / (2200 * 740 * 50e-9)          # initial peak rise for unit absorbed fluence
    s0sq = w ** 2 / 4
    ra, rb = res.r_faces[:-1], res.r_faces[1:]
    for t in (0.0, 50e-9, 200e-9, 400e-9):
        k = np.argmin(abs(res.t - t))
        sig = s0sq + 2 * D * res.t[k]
        ring_avg = 2 * sig * (np.exp(-ra ** 2 / (2 * sig)) - np.exp(-rb ** 2 / (2 * sig))) / (rb ** 2 - ra ** 2)
        exact = A * s0sq / sig * ring_avg
        assert res.dT[k, 0, :] == pytest.approx(exact, abs=2e-3 * A)


def test_steady_state_uniform_heating_parabola():
    """Slab, top insulated, bottom at T0, uniform q:  dT(z) = q (L^2 - z^2) / (2 k)."""
    L_nm, k = 500.0, 1.4
    s = StackHeatSolver([ThermalLayer("s", L_nm, k, 2200, 740, cells=50)], radial_grid(5, 0.5, 1.0),
                        top="insulated", bottom="fixed", outer="insulated")
    q = 1e14                                                  # W/m^3
    Q = q * s.V                                               # per-cell power, deposited for t_pulse = 1 s
    tau = (L_nm * 1e-9) ** 2 * 2200 * 740 / k
    res = s.run(Q * 1.0, 1.0, 1.0 + tau)                      # long pulse: steady state reached at t = 1 s
    k_end = np.argmin(abs(res.t - 1.0))
    z = res.z_centres
    exact = q * ((L_nm * 1e-9) ** 2 - z ** 2) / (2 * k)
    # second-order truncation error at the Dirichlet half-cell: compare relative to the peak
    assert res.dT[k_end, :, 0] == pytest.approx(exact, abs=2e-4 * exact.max())


def test_series_thermal_resistance():
    """Thin heated layer on two layers to a sink: dT_top = q'' (d1/k1 + d2/k2)."""
    layers = [ThermalLayer("heater", 1, 200, 2000, 100, cells=1), ThermalLayer("a", 300, 1.4, 2200, 740, 20),
              ThermalLayer("b", 2000, 148, 2330, 700, 20)]
    s = StackHeatSolver(layers, radial_grid(5, 0.5, 1.0), top="insulated", bottom="fixed", outer="insulated")
    flux = 1e9                                                # W/m^2
    Q = s.gaussian_source(0, flux * 1.0, 1e9)                 # w -> infinity: uniform flux, 1 s pulse
    res = s.run(Q, 1.0, 1.1)
    k_end = np.argmin(abs(res.t - 1.0))
    expected = flux * (300e-9 / 1.4 + 2000e-9 / 148) + flux * 0.5e-9 / 200
    assert res.dT[k_end, 0, 0] == pytest.approx(expected, rel=1e-3)


def test_lumped_rc_limit():
    """High-k film on a thin, low-heat-capacity resistive spacer reduces to the RC model."""
    film = ThermalLayer("film", 20, 500, 6000, 220, cells=2)
    spacer = ThermalLayer("spacer", 10, 0.05, 1.0, 1.0, cells=2)    # R = d/k, negligible capacity
    s = StackHeatSolver([film, spacer], radial_grid(5, 0.5, 1.0), top="insulated", bottom="fixed", outer="insulated")
    C = 6000 * 220 * 20e-9
    R = 10e-9 / 0.05
    tau = R * C
    tp, E = 100e-9, 30.0
    res = s.run(s.gaussian_source(0, E, 1e9), tp, 3 * tp)
    T = res.layer_average(0, 0.0)
    k = np.argmin(abs(res.t - tp))
    assert T[k] == pytest.approx((E / tp) * R * (1 - math.exp(-res.t[k] / tau)), rel=2e-3)


def test_linearity_in_absorbed_energy():
    layers = [ThermalLayer("pcm", 20, 0.5, 6000, 220), ThermalLayer("sio2", 270, **SIO2)]
    s = StackHeatSolver(layers, radial_grid(10, 0.1))
    a = s.run(s.gaussian_source(0, 10.0, 1.0), 20e-9, 100e-9).dT
    b = s.run(s.gaussian_source(0, 30.0, 1.0), 20e-9, 100e-9).dT
    assert b == pytest.approx(3 * a, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("Rb", [0.0, 1e-8, 5e-8])
def test_thermal_boundary_resistance_adds_in_series(Rb):
    """Steady flux through heater | a | b | sink with interface resistances: dT = q'' (d1/k1 + d2/k2 + R_ab + R_heater_a)."""
    layers = [ThermalLayer("heater", 1, 200, 2000, 100, cells=1, tbr_below=Rb),
              ThermalLayer("a", 300, 1.4, 2200, 740, 20, tbr_below=2 * Rb),
              ThermalLayer("b", 2000, 148, 2330, 700, 20)]
    s = StackHeatSolver(layers, radial_grid(5, 0.5, 1.0), top="insulated", bottom="fixed", outer="insulated")
    flux = 1e9
    res = s.run(s.gaussian_source(0, flux * 1.0, 1e9), 1.0, 1.1)
    k_end = np.argmin(abs(res.t - 1.0))
    expected = flux * (300e-9 / 1.4 + 2000e-9 / 148 + 0.5e-9 / 200 + 3 * Rb)
    assert res.dT[k_end, 0, 0] == pytest.approx(expected, rel=1e-3)


def test_nonlinear_solver_honours_boundary_resistance():
    from pqcsim.thermal.nonlinear import NonlinearStackHeatSolver, PhaseChangeLayer
    lin = [ThermalLayer("pcm", 20, 0.5, 6140, 220, 3, tbr_below=3e-8), ThermalLayer("ox", 200, 1.4, 2200, 740, 6)]
    nl = [PhaseChangeLayer("pcm", 20, 0.5, 6140, 220, 3, tbr_below=3e-8), PhaseChangeLayer("ox", 200, 1.4, 2200, 740, 6)]
    rf = radial_grid(4, 0.1, 1.15)
    a, b = StackHeatSolver(lin, rf), NonlinearStackHeatSolver(nl, rf)
    Q = a.gaussian_source(0, 30.0, 0.8)
    ra, rb = a.run(Q, 50e-9, 200e-9, rtol=1e-10), b.run(Q, 50e-9, 200e-9, rtol=1e-8, method="Radau")
    common = np.intersect1d(np.round(ra.t, 18), np.round(rb.t, 18))
    Ta = ra.layer_average(0, 0.0)[np.isin(np.round(ra.t, 18), common)]
    Tb = rb.layer_average(0, 0.0)[np.isin(np.round(rb.t, 18), common)]
    assert Tb == pytest.approx(Ta, abs=1e-6 * Ta.max())
    noR = StackHeatSolver([ThermalLayer("pcm", 20, 0.5, 6140, 220, 3), ThermalLayer("ox", 200, 1.4, 2200, 740, 6)], rf)
    assert noR.run(Q, 50e-9, 200e-9).layer_average(0, 0.0).max() < ra.layer_average(0, 0.0).max()   # TBR traps heat
