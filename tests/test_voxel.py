"""3D voxel heat solver: conservation, analytic steady states, and cross-check vs the axisymmetric solver."""
import math

import numpy as np
import pytest

from pqcsim.thermal.diffusion import StackHeatSolver, ThermalLayer, radial_grid
from pqcsim.thermal.voxel import Material, PatternedLayer, VoxelHeatSolver, uniform_faces

SIO2 = Material("SiO2", 1.4, 2200, 740)
PCM = Material("GeTe", 0.5, 6140, 220)
AU = Material("Au", 300, 19300, 129)


def test_energy_conservation_insulated():
    layers = [PatternedLayer("pcm", 20, SIO2, 2, [(PCM, -0.4, 0.4, -0.4, 0.4)]), PatternedLayer("sp", 200, SIO2, 4)]
    f = uniform_faces(2.0, 0.1)
    s = VoxelHeatSolver(layers, f, f, top="insulated", bottom="insulated", sides="insulated")
    lat = s.gaussian_lateral(0.8)
    Q = np.zeros((s.nz, s.nx, s.ny))
    Q[0:2] = 50.0 * lat[None] / 2
    res = s.run(Q, 20e-9, 200e-9)
    assert res.heat_content()[-1] == pytest.approx(Q.sum(), rel=1e-9)
    assert lat.sum() == pytest.approx(0.5 * math.pi * (0.8e-6) ** 2 * math.erf(2 * math.sqrt(2) / 0.8) ** 2, rel=1e-12)


def test_steady_series_resistance_with_patterned_layer():
    """Uniform flux through a laterally uniform stack: dT_top = q'' sum d/k (patterning irrelevant if k equal)."""
    layers = [PatternedLayer("heater", 2, Material("h", 200, 2000, 100), 1), PatternedLayer("a", 300, SIO2, 10),
              PatternedLayer("b", 1000, Material("Si", 148, 2330, 700), 10)]
    f = uniform_faces(1.0, 0.5)
    s = VoxelHeatSolver(layers, f, f, top="insulated", bottom="fixed", sides="insulated")
    flux = 1e9
    Q = np.zeros((s.nz, s.nx, s.ny))
    Q[0] = flux * s.dx[:, None] * s.dy[None, :] * 1e-3          # deposited over t_pulse = 1 ms
    res = s.run(Q, 1e-3, 1.01e-3, steps_pulse=200)
    expected = flux * (300e-9 / 1.4 + 1000e-9 / 148 + 1e-9 / 200)
    assert res.dT[np.argmin(abs(res.t - 1e-3)), 0, 0, 0] == pytest.approx(expected, rel=1e-3)


def test_matches_axisymmetric_solver_for_continuous_film():
    """Independent discretisations (Cartesian BDF2 vs cylindrical adaptive BDF) of the same problem."""
    tp, tend, w = 50e-9, 400e-9, 0.8
    ax = StackHeatSolver([ThermalLayer("pcm", 20, 0.5, 6140, 220, 2), ThermalLayer("sio2", 300, 1.4, 2200, 740, 8)],
                         radial_grid(6, 0.02, 1.05), top="insulated", bottom="fixed", outer="fixed")
    ra = ax.run(ax.gaussian_source(0, 40.0, w), tp, tend)
    f = uniform_faces(2.5, 0.1)
    vx = VoxelHeatSolver([PatternedLayer("pcm", 20, PCM, 2), PatternedLayer("sio2", 300, SIO2, 8)], f, f,
                         top="insulated", bottom="fixed", sides="fixed")
    Q = np.zeros((vx.nz, vx.nx, vx.ny))
    Q[0:2] = 40.0 * vx.gaussian_lateral(w)[None] / 2
    rv = vx.run(Q, tp, tend, steps_pulse=60, steps_after=200)
    peak = ra.layer_average(0, 0.0).max()
    for ix in (25, 30, 36):                      # voxel columns at y = 0.05 um
        iy = 25
        r = math.hypot(vx.xc[ix], vx.yc[iy])
        Tv = (rv.dT[:, 0:2, ix, iy] * vx.C[0:2, ix, iy]).sum(axis=1) / vx.C[0:2, ix, iy].sum()
        Ta = np.interp(rv.t, ra.t, ra.layer_average(0, r))
        assert np.max(np.abs(Tv - Ta)) < 0.02 * peak


def test_patterned_array_symmetry():
    p, a = 1.0, 0.3
    inc = [(PCM, i * p - a, i * p + a, j * p - a, j * p + a) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    layers = [PatternedLayer("pcm", 20, SIO2, 2, inc), PatternedLayer("sio2", 270, SIO2, 4)]
    f = uniform_faces(2.5, 0.1)
    s = VoxelHeatSolver(layers, f, f)
    Q = np.zeros((s.nz, s.nx, s.ny))
    lat = s.gaussian_lateral(0.6)
    mask = np.array([[s.layers[0].material_at(x, y) is PCM for y in s.yc * 1e6] for x in s.xc * 1e6])
    Q[0:2] = 100.0 * (lat * mask)[None] / 2
    res = s.run(Q, 50e-9, 300e-9)
    rows = s.layer_of_row == 0
    T = {c: res.region_average(rows, (c[0] - a, c[0] + a), (c[1] - a, c[1] + a)).max() for c in
         [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]}
    assert T[(1, 0)] == pytest.approx(T[(-1, 0)], rel=1e-9) == pytest.approx(T[(0, 1)], rel=1e-9)
    assert T[(1, 1)] == pytest.approx(T[(-1, -1)], rel=1e-9)
    assert T[(0, 0)] > T[(1, 0)] > T[(1, 1)] > 0


@pytest.fixture(scope="module")
def patterned_response():
    from pqcsim.thermal.response import PatternedHeatResponse
    p, a = 1.0, 0.35
    inc = [(PCM, i * p - a, i * p + a, j * p - a, j * p + a) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    layers = [PatternedLayer("cap", 20, SIO2, 1), PatternedLayer("pcm", 20, SIO2, 2, inc),
              PatternedLayer("spacer", 270, SIO2, 3), PatternedLayer("Au", 100, AU, 1)]
    f = uniform_faces(1.8, 0.1)
    s = VoxelHeatSolver(layers, f, f)
    return PatternedHeatResponse(s, active_layer=1, spot_radius_um=0.6, t_pulse=100e-9, t_end=1e-6,
                                 pitch_um=p, cell_um=2 * a, heated_layers=[1, 3], steps_pulse=30, steps_after=80)


def test_patterned_array_write_matches_direct_voxel_run(cell, patterned_response):
    from pqcsim.arrays.crossbar import PCMArray
    from pqcsim.optics.tmm import tmm
    from pqcsim.thermal.response import optical_row_fractions
    hr = patterned_response
    arr = PCMArray(cell, 3, 3, pitch_um=1.0, heat_response=hr)
    F = 400.0
    res = arr.write(1, 1, F, 1550.0, 100e-9)
    # direct run with the combined depth- and column-resolved source
    s = hr.solver
    nl = cell.indices(0.0, 1550.0, "set")
    fp = optical_row_fractions(1550.0, 1.0, nl, cell.thicknesses_nm, cell.optics(0.0, 1550.0, "set"), s)
    fl = [1.45 if k == 1 else n for k, n in enumerate(nl)]
    ff = optical_row_fractions(1550.0, 1.0, fl, cell.thicknesses_nm, tmm(1550.0, 1.0, fl, cell.thicknesses_nm, 1.45), s)
    lat = s.gaussian_lateral(0.6)
    Q = F * (fp[:, None, None] * (lat * hr.pcm_mask)[None] + ff[:, None, None] * (lat * ~hr.pcm_mask)[None])
    direct = s.run(Q, 100e-9, 1e-6, 30, 80)
    T = cell.T0 + direct.region_average(s.layer_of_row == 1, (-0.35, 0.35), (-0.35, 0.35))
    assert res.T_peak == pytest.approx(T.max(), rel=1e-9)
    # symmetric disturbance on the four nearest neighbours
    X = arr.X
    assert X[0, 1] == pytest.approx(X[2, 1], rel=1e-9) and X[1, 0] == pytest.approx(X[1, 2], rel=1e-9)
    assert X[0, 1] == pytest.approx(X[1, 0], rel=1e-9)


def test_rcwa_optics_in_patterned_array_conserves_absorbed_energy(cell, patterned_response):
    """RCWA row fractions: heat per unit cell equals the RCWA cell-averaged absorption."""
    from pqcsim.arrays.crossbar import PCMArray
    hr = patterned_response
    arr = PCMArray(cell, 3, 3, pitch_um=1.0, heat_response=hr, optics="rcwa", rcwa_orders=3)
    rows_pcm, rows_fill, A_pix = arr._rcwa_rows(0.0, 1550.0, "set", hr)
    ff = (hr.cell_um / 1.0) ** 2
    act = hr.solver.rows_of_layer(1)
    assert rows_fill[act] == pytest.approx(0.0)                     # lossless fill absorbs nothing
    assert (rows_pcm[act] * ff).sum() == pytest.approx(A_pix * ff)
    res = arr.write(1, 1, 400.0, 1550.0, 100e-9)
    assert res.T_peak > cell.T0 and 0.0 <= res.x <= 1.0
    assert arr.X[0, 1] == pytest.approx(arr.X[1, 0], rel=1e-9)


def test_interface_resistance_steady_series():
    """Uniform flux through layers a | b with a Kapitza resistance at the a/b interface and at the
    sidewalls of an inclusion: steady dT_top = q'' (sum d/k + R_b) (exact for a laterally uniform stack)."""
    R_b = 2e-8
    A_ = Material("A", 1.4, 2200, 740)
    B_ = Material("B", 148, 2330, 700)
    layers = [PatternedLayer("heater", 2, Material("h", 200, 2000, 100), 1), PatternedLayer("a", 300, A_, 10),
              PatternedLayer("b", 1000, B_, 10)]
    f = uniform_faces(1.0, 0.25)
    s = VoxelHeatSolver(layers, f, f, top="insulated", bottom="fixed", sides="insulated", tbr={("A", "B"): R_b})
    q = 1e6                                                   # W/m^2
    Q = np.zeros((s.nz, s.nx, s.ny))
    area = np.outer(s.dx, s.dy)
    Q[0] = q * area * 1e-3                                    # J over a 1 ms pulse -> steady at the end
    res = s.run(Q, 1e-3, 1.0001e-3, steps_pulse=400, steps_after=2)
    i_end = np.argmin(np.abs(res.t - 1e-3))
    expect = q * (2e-9 / 2 / 200 + 300e-9 / 1.4 + 1000e-9 / 148 + R_b)
    assert res.dT[i_end, 0].mean() == pytest.approx(expect, rel=5e-3)
    # no resistance between equal materials, and the default (no tbr) is unchanged
    s0 = VoxelHeatSolver(layers, f, f, top="insulated", bottom="fixed", sides="insulated")
    r0 = s0.run(Q, 1e-3, 1.0001e-3, steps_pulse=400, steps_after=2)
    assert r0.dT[i_end, 0].mean() == pytest.approx(expect - q * R_b, rel=5e-3)


def test_interface_resistance_matches_axisymmetric_transient():
    """Laterally uniform heating, 1D transient with interface resistances: voxel vs StackHeatSolver(tbr_below)."""
    R1, R2 = 1e-8, 3e-8
    cap, gst, sp = Material("cap", 1.2, 2200, 740), Material("gst", 0.5, 6150, 220), Material("sp", 1.2, 2200, 740)
    # distinct names so each interface gets its own resistance
    layers = [PatternedLayer("cap", 20, cap, 2), PatternedLayer("gst", 20, gst, 3), PatternedLayer("sp", 270, sp, 8)]
    f = uniform_faces(0.5, 0.25)
    s = VoxelHeatSolver(layers, f, f, top="insulated", bottom="fixed", sides="insulated",
                        tbr={("cap", "gst"): R1, ("gst", "sp"): R2})
    Q = np.zeros((s.nz, s.nx, s.ny))
    Q[s.rows_of_layer(1)] = 1e-3 * np.outer(s.dx, s.dy) / 3
    res = s.run(Q, 10e-9, 60e-9, steps_pulse=200, steps_after=400)
    ax = StackHeatSolver([ThermalLayer("cap", 20, 1.2, 2200, 740, 2, tbr_below=R1),
                          ThermalLayer("gst", 20, 0.5, 6150, 220, 3, tbr_below=R2),
                          ThermalLayer("sp", 270, 1.2, 2200, 740, 8)], np.array([0.0, 1e-3]),
                         top="insulated", bottom="fixed", outer="insulated")
    q_row = np.zeros(ax.nz); q_row[ax.rows_of_layer(1)] = 1e-3 / 3   # J/m^2 per row
    out = ax.run(q_row[:, None] * ax.V / ax.dz[:, None], 10e-9, 60e-9, n_out=300)   # J per cell (uniform)
    T_vox = res.region_average(s.layer_of_row == 1, (-0.5, 0.5), (-0.5, 0.5))
    T_ax = out.layer_average(1, 0.0)
    assert np.interp(10e-9, res.t, T_vox) == pytest.approx(np.interp(10e-9, out.t, T_ax), rel=1e-2)
    assert np.interp(40e-9, res.t, T_vox) == pytest.approx(np.interp(40e-9, out.t, T_ax), rel=2e-2)
