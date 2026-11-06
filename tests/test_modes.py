"""Waveguide mode solvers: exact slab vs 2D FD, perturbation theory, symmetry."""
import numpy as np
import pytest

from pqcsim.optics.modes import graded_axis, quasi_te_modes, slab_te_neff

LAM = 1550.0
K0 = 2 * np.pi / (LAM * 1e-9)


def test_slab_matches_closed_form_symmetric_slab():
    """Symmetric TE slab: tan(q d/2) = g / q (fundamental even mode), solved independently by bisection."""
    from scipy.optimize import brentq
    n1, n2, d = 3.48, 1.444, 220.0
    ne = slab_te_neff(LAM, [n1], [d], n2, n2, 2.8)
    f = lambda n: np.tan(K0 * np.sqrt(n1 ** 2 - n ** 2) * d * 1e-9 / 2) - np.sqrt(n ** 2 - n2 ** 2) / np.sqrt(n1 ** 2 - n ** 2)
    ref = brentq(f, n2 + 1e-6, n1 - 1e-6)
    assert ne.real == pytest.approx(ref, abs=1e-10) and abs(ne.imag) < 1e-12


def _uniform_slab_case(hy):
    n_gst, W, h = 4.0 + 0.05j, 4000.0, 50.0
    y = graded_axis([-2500, 0, 335, 345, 2500], [2 * hy, hy, hy / 10, 2 * hy])
    x = np.linspace(-W / 2, W / 2, int(W / h) + 1)

    def index(X, Y):
        n = np.full(X.shape, 1.0 + 0j)
        n[Y < 0] = 1.444
        n[(Y >= 0) & (Y < 335)] = 1.99
        n[(Y >= 335) & (Y < 345)] = n_gst
        return n
    m = quasi_te_modes(LAM, x, y, index, 1.8)[0]
    lateral = (2 / (h * 1e-9) ** 2) * (1 - np.cos(np.pi * h / W))      # discrete eigenvalue of sin(pi x / W)
    return np.sqrt(m.neff ** 2 + lateral / K0 ** 2), slab_te_neff(LAM, [n_gst, 1.99], [10, 335], 1.0, 1.444, 1.8)


def test_2d_solver_reduces_to_exact_slab_and_converges():
    e10, ref = _uniform_slab_case(10)
    e5, _ = _uniform_slab_case(5)
    assert abs(e5.real - ref.real) < 5e-4
    assert abs(e5.imag - ref.imag) / ref.imag < 0.01            # loss within 1 %
    assert abs(e5 - ref) < abs(e10 - ref) * 1.5                 # not diverging under refinement


def test_weak_absorber_loss_is_linear_in_k():
    """First-order perturbation: Im(n_eff) proportional to the absorber's k for k << n."""
    vals = [slab_te_neff(LAM, [4.0 + 1j * k, 1.99], [10, 335], 1.0, 1.444, 1.8).imag for k in (1e-3, 2e-3, 4e-3)]
    assert vals[1] / vals[0] == pytest.approx(2.0, rel=1e-3)
    assert vals[2] / vals[0] == pytest.approx(4.0, rel=2e-3)


def test_graded_axis_contains_interfaces_and_grades_smoothly():
    y = graded_axis([-1000, 0, 335, 345, 1000], [40, 8, 1, 40])
    for b in (-1000, 0, 335, 345, 1000):
        assert np.min(np.abs(y - b)) < 1e-9
    assert np.all(np.diff(y) > 0)
    assert np.max(np.diff(y)[(y[:-1] >= 335) & (y[1:] <= 345)]) <= 1.0 + 1e-9


def test_rib_loss_is_bounded_by_fully_covered_slab():
    """A film covering only the rib top cannot absorb more than the same film covering a full slab."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("rios", Path(__file__).resolve().parent.parent / "validation" / "rios2015_gst_waveguide.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    n_cr = 6.46 + 0.9j
    rib = R.quasi_te_modes(1560.0, R.x, R.y, R.structure(n_cr, 1.8), 1.75)[0]
    slab = slab_te_neff(1560.0, [1.8, n_cr, 1.99], [10, 10, 335], 1.0, 1.444, 1.8)
    assert 0 < rib.neff.imag <= slab.imag


def test_reproduces_zheng2018_measured_crystalline_loss():
    """Zheng et al., OME 8, 1551 (2018): measured alpha_cGST = 7.6 +- 1.0 dB/um (paper's own ellipsometry)."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("zheng", Path(__file__).resolve().parent.parent / "validation" / "zheng2018_gst_on_si.py")
    Z = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Z)
    m = quasi_te_modes(1550.0, Z.x, Z.y, Z.structure(Z.C_GST, 1.9 + 0.002j, "ridge"), 2.7)[0]
    assert abs(m.loss_dB_per_um(1550.0) - 7.6) <= 1.0
    assert m.neff.imag == pytest.approx(0.22, rel=0.05)          # paper's FEM: Im(n_eff) = 0.22
