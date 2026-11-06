"""Spatially resolved programming: energy bookkeeping, linear-route consistency, melt-quench RESET."""
import numpy as np
import pytest

from pqcsim.devices.pcm import PCMCell
from pqcsim.devices.resolved import resolved_pulse
from pqcsim.thermal.diffusion import radial_grid
from pqcsim.thermal.nonlinear import NonlinearStackHeatSolver, PhaseChangeLayer

RF = radial_grid(4, 0.2, 1.25)


def stack(latent=0.0, Tm=None, k_liq=None):
    return [PhaseChangeLayer("cap", 20, 1.4, 2200, 740, 1),
            PhaseChangeLayer("pcm", 20, 0.5, 6140, 220, 3, k_liquid=k_liq, latent_J_per_kg=latent, T_melt=Tm),
            PhaseChangeLayer("spacer", 270, 1.4, 2200, 740, 4),
            PhaseChangeLayer("Au", 100, 300, 19300, 129, 1),
            PhaseChangeLayer("sub", 1000, 1.4, 2200, 740, 3)]


def test_absorbed_energy_and_linear_route_consistency(cell):
    """No latent heat, constant k: the direct nonlinear run must equal the linear superposition route."""
    from pqcsim.thermal.diffusion import StackHeatSolver, ThermalLayer
    from pqcsim.thermal.response import PulseHeatResponse, optical_row_fractions
    s = NonlinearStackHeatSolver(stack(), RF)
    x0 = np.zeros((3, s.nr))
    F, tp = 150.0, 100e-9
    out = resolved_pulse(cell, s, x0, F, 1550.0, tp, 1e-6, 1.0, dT_step_K=1.0)
    opt = cell.optics(0.0, 1550.0, "set")
    ring = s.geo.ring_energies(F, 1.0)
    assert out.absorbed_J == pytest.approx(ring.sum() * opt.A_total, rel=1e-10)
    lin = StackHeatSolver([L.linear() for L in stack()], RF)
    hr = PulseHeatResponse(lin, 1, 1.0, tp, 1e-6, heated_layers=[1, 3])
    frac = optical_row_fractions(1550.0, 1.0, cell.indices(0.0, 1550.0, "set"), cell.thicknesses_nm, opt, lin)
    T_lin = cell.T0 + hr.rise(0.0, F * frac)
    # interpolate the densely sampled linear solution onto the resolved run's samples (not the reverse:
    # interpolating the coarser series across the pulse-end kink adds spurious error)
    T_lin_at = np.interp(out.t, hr.t, T_lin)
    assert np.max(np.abs(out.T_active_mean - T_lin_at)) < 0.003 * (T_lin.max() - cell.T0)
    assert out.T_active_mean.max() == pytest.approx(T_lin.max(), abs=0.01)


def test_melt_quench_reset_slow_crystallizer():
    """GST225 (slow kinetics): melted cells end amorphous; cells that never melt stay crystalline."""
    from pathlib import Path
    from pqcsim.materials import load_materials
    gst = {m.name: m for m in load_materials(str(Path(__file__).resolve().parent.parent / "data" / "materials.json"))}["GST225"]
    cell = PCMCell(gst, [1.45, "active", 1.45, 0.55 + 11.5j], [20.0, 20.0, 270.0, 100.0], readout="1-R")
    Tm = gst.kinetics.T_melt
    s = NonlinearStackHeatSolver(stack(latent=1.0e5, Tm=Tm, k_liq=1.5), RF)
    out = resolved_pulse(cell, s, np.ones((3, s.nr)), 1500.0, 1550.0, 10e-9, 300e-9, 1.0)
    assert out.melted.any() and not out.melted.all()
    assert np.all(out.melted == (out.T_peak_field >= Tm))
    assert np.all(out.x_field[out.melted] < 0.05)                     # melt-quenched -> amorphous
    cold = out.T_peak_field < gst.kinetics.T_cryst_min
    assert np.all(out.x_field[cold] == 1.0)                           # far cells untouched
    assert 0.0 < out.x_mean < 1.0
