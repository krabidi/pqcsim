import math

import numpy as np
import pytest

from pqcsim.arrays.crossbar import PCMArray, exponential_kernel
from pqcsim.electronics.readout import ADC, K_B, Photodetector, Q_E

# ~140 J/m^2 incident -> T_peak ~ 600 K in the default GeTe stack (SET regime)
SET = dict(fluence=140.0, lam_nm=1550.0, t_pulse=100e-9)


def test_isolated_write_changes_only_target(cell):
    arr = PCMArray(cell, 3, 3, pitch_um=2.0)
    res = arr.write(1, 1, **SET)
    assert 450 < res.T_peak < 670
    assert res.x > 0 and arr.X[1, 1] == res.x
    arr.X[1, 1] = 0.0
    assert np.all(arr.X == 0.0)


def test_optical_tail_disturb_matches_single_cell(cell):
    w = 1.5
    arr = PCMArray(cell, 3, 3, pitch_um=1.0, spot_radius_um=w)
    arr.write(1, 1, fluence=170.0, lam_nm=1550.0, t_pulse=100e-9)
    F_tail = 170.0 * math.exp(-2 * 1.0 ** 2 / w ** 2)
    expected = cell.apply_pulse(0.0, F_tail, 1550.0, 100e-9).x
    assert arr.X[0, 1] == pytest.approx(expected, rel=1e-12)
    # four nearest neighbours are equivalent, diagonal neighbours are disturbed less
    assert arr.X[0, 1] == arr.X[1, 0] == arr.X[2, 1] == arr.X[1, 2]
    assert arr.X[0, 1] > 0
    assert arr.X[0, 0] < arr.X[0, 1]      # diagonal neighbours (r = sqrt(2) pitch) stay below T_cryst_min


def test_thermal_kernel_adds_conducted_energy(cell):
    arr = PCMArray(cell, 1, 2, pitch_um=2.0, thermal_kernel=exponential_kernel(2.0))
    res = arr.write(0, 0, **SET)
    expected = cell.apply_pulse(0.0, 0.0, 1550.0, 100e-9, E_abs=math.exp(-1) * res.E_abs).x
    assert arr.X[0, 1] == pytest.approx(expected, rel=1e-12)


def test_mvm_is_linear_weighted_sum(cell):
    arr = PCMArray(cell, 2, 3, pitch_um=5.0)
    arr.X = np.array([[0.0, 0.5, 1.0], [0.2, 0.9, 0.4]])
    P = np.array([1e-3, 2e-3, 0.5e-3])
    assert arr.mvm(P) == pytest.approx(arr.weights() @ P)
    assert arr.mvm(2 * P) == pytest.approx(2 * arr.mvm(P))


def test_photodetector_noise_statistics():
    pd = Photodetector(responsivity=0.8, dark_current=1e-8, bandwidth=1e9, R_f=1e3)
    P = 1e-4
    V = pd.voltage(np.full(200000, P), rng=np.random.default_rng(0))
    sigma_I = math.sqrt(2 * Q_E * (0.8 * P + 1e-8) * 1e9 + 4 * K_B * 300 * 1e9 / 1e3)
    assert V.mean() == pytest.approx((0.8 * P + 1e-8) * 1e3, rel=1e-3)
    assert V.std() == pytest.approx(sigma_I * 1e3, rel=1e-2)


def test_adc_quantization_and_clipping():
    adc = ADC(bits=4, v_fullscale=1.5)
    V = np.linspace(0, 1.5, 1001)
    assert np.max(np.abs(adc.quantize(V) - V)) <= adc.lsb / 2 + 1e-15
    assert adc.codes([-1.0, 9.0]).tolist() == [0, 15]


@pytest.fixture(scope="module")
def heat_response():
    from pqcsim.thermal.diffusion import StackHeatSolver, ThermalLayer, radial_grid
    from pqcsim.thermal.response import PulseHeatResponse
    layers = [ThermalLayer("cap", 20, 1.4, 2200, 740, 2), ThermalLayer("pcm", 20, 0.5, 6140, 220, 3),
              ThermalLayer("spacer", 270, 1.4, 2200, 740, 8), ThermalLayer("Au", 100, 300, 19300, 129, 2),
              ThermalLayer("sub", 2000, 1.4, 2200, 740, 8)]
    solver = StackHeatSolver(layers, radial_grid(30, 0.05, 1.06), top="insulated", bottom="fixed", outer="fixed")
    return PulseHeatResponse(solver, active_layer=1, spot_radius_um=1.0, t_pulse=100e-9, t_end=2e-6,
                             heated_layers=[1, 3])          # PCM and Au mirror absorb


def test_diffusion_array_target_uses_solver_history(cell, heat_response):
    from pqcsim.thermal.response import optical_row_fractions
    arr = PCMArray(cell, 3, 3, pitch_um=1.5, heat_response=heat_response)
    F = 150.0
    res = arr.write(1, 1, F, 1550.0, 100e-9)
    opt = cell.optics(0.0, 1550.0, "set")
    frac = optical_row_fractions(1550.0, 1.0, cell.indices(0.0, 1550.0, "set"), cell.thicknesses_nm, opt,
                                 heat_response.solver)
    assert frac.sum() == pytest.approx(opt.A_total, abs=1e-12)     # optics -> heat: all absorbed power
    # direct solver run with the same depth-resolved source (independent of the row-basis superposition)
    solver = heat_response.solver
    direct = solver.run(solver.row_source(F * frac, 1.0), 100e-9, 2e-6)
    T = cell.T0 + direct.layer_average(1, 0.0)
    assert res.T_peak == pytest.approx(T.max(), rel=1e-5)
    assert res.x == pytest.approx(cell.apply_temperature_history(0.0, direct.t, T).x, rel=1e-4, abs=1e-9)


def test_heat_content_equals_absorbed_optical_energy(cell):
    from pqcsim.thermal.diffusion import StackHeatSolver, ThermalLayer, radial_grid
    from pqcsim.thermal.response import optical_row_fractions
    layers = [ThermalLayer("cap", 20, 1.4, 2200, 740, 2), ThermalLayer("pcm", 20, 0.5, 6140, 220, 4),
              ThermalLayer("spacer", 270, 1.4, 2200, 740, 6), ThermalLayer("Au", 100, 300, 19300, 129, 3)]
    s = StackHeatSolver(layers, radial_grid(20, 0.1, 1.08), "insulated", "insulated", "insulated")
    opt = cell.optics(0.4, 1550.0)
    frac = optical_row_fractions(1550.0, 1.0, cell.indices(0.4, 1550.0), cell.thicknesses_nm, opt, s)
    res = s.run(s.row_source(80.0 * frac, 1.0), 50e-9, 300e-9)
    E_in = 80.0 * opt.A_total * 0.5 * np.pi * (1e-6) ** 2 * (1 - np.exp(-2 * 20 ** 2))
    assert res.heat_content()[-1] == pytest.approx(E_in, rel=1e-5)


def test_diffusion_array_disturb_symmetric_and_decays(cell, heat_response):
    # 0.5 um pitch, 600 J/m^2: target ~630 K (no melt), nearest neighbour ~500 K (> T_cryst_min = 450 K)
    arr = PCMArray(cell, 5, 5, pitch_um=0.5, heat_response=heat_response)
    arr.write(2, 2, 600.0, 1550.0, 100e-9)
    X = arr.X
    assert X[1, 2] == X[3, 2] == X[2, 1] == X[2, 3]                # nearest neighbours (r = 1 pitch)
    assert X[1, 1] == X[1, 3] == X[3, 1] == X[3, 3]                # diagonals (r = sqrt 2 pitch)
    assert X[2, 2] >= X[1, 2] >= X[1, 1] >= X[0, 2] >= 0.0
    assert X[1, 2] > 0                                             # heat really reaches the neighbour
    assert X[0, 0] == 0.0                                          # r = 1.41 um stays below T_cryst_min


def test_heat_response_rejects_double_counting(cell, heat_response):
    with pytest.raises(ValueError):
        PCMArray(cell, 2, 2, 1.0, spot_radius_um=1.0, heat_response=heat_response)
