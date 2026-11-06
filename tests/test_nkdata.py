import numpy as np
import pytest

from pqcsim.optics.nkdata import load_nk


def test_sellmeier_reference_values():
    # Malitson fused silica: n(1550 nm) = 1.44402; Luke Si3N4: n(1550 nm) = 1.9963
    assert load_nk("SiO2_Malitson1965").nk(1550.0).real == pytest.approx(1.44402, abs=2e-5)
    assert load_nk("Si3N4_Luke2015").nk(1550.0).real == pytest.approx(1.9963, abs=2e-4)
    assert load_nk("SiO2_Malitson1965").nk(587.6).real == pytest.approx(1.45846, abs=2e-5)   # d-line


def test_tabulated_gst_and_range_checks():
    am, cr = load_nk("GST225_Frantz2023_amorphous"), load_nk("GST225_Frantz2023_crystalline")
    assert am.nk(1550.0) == pytest.approx(4.073 + 0.009j, abs=2e-3)
    assert cr.nk(1550.0) == pytest.approx(6.464 + 0.918j, abs=2e-3)
    assert "Frantz" in cr.references
    with pytest.raises(ValueError):
        am.nk(200.0)


def test_metals_have_large_extinction():
    for name in ("Au_Johnson1972", "Pt_Werner2009"):
        n = load_nk(name).nk(1550.0)
        assert n.imag > 5 and n.real > 0
