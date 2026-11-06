import numpy as np
import pytest

from pqcsim.materials import MIXING_RULES, PCMMaterial, mix_index

AM, CR = 3.64 + 0.48j, 4.48 + 1.45j


@pytest.mark.parametrize("rule", MIXING_RULES)
def test_mixing_endpoints_and_passivity(rule):
    assert mix_index(AM, CR, 0.0, rule) == pytest.approx(AM, abs=1e-12)
    assert mix_index(AM, CR, 1.0, rule) == pytest.approx(CR, abs=1e-12)
    for x in np.linspace(0, 1, 21):
        n = mix_index(AM, CR, x, rule)
        assert n.imag >= 0 and n.real > 0


def test_lorentz_lorenz_definition():
    x = 0.37
    n = mix_index(AM, CR, x, "lorentz_lorenz")
    f = lambda e: (e - 1) / (e + 2)
    assert f(n ** 2) == pytest.approx(x * f(CR ** 2) + (1 - x) * f(AM ** 2), abs=1e-12)


def test_tabulated_interpolation_and_range():
    m = PCMMaterial("t", {"mode": "tabulated", "lam_nm": [1400, 1700], "n_am": [3, 4], "k_am": [0, 0.3],
                          "n_cr": [5, 5], "k_cr": [1, 2]})
    assert m.nk_state("am", 1550) == pytest.approx(3.5 + 0.15j)
    with pytest.raises(ValueError):
        m.nk_state("am", 1800)


def test_loads_photomem_json(gete):
    assert gete.nk_state("cr", 1550) == pytest.approx(4.48 + 1.45j)
    assert gete.kinetics.Ea_eV == 2.2 and gete.thermal.tau_th == 2e-8
