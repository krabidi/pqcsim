"""RCWA: exact TMM limit, gap independence, energy conservation, symmetry, diffraction, convergence."""
import numpy as np
import pytest

from pqcsim.optics.rcwa import RCWA, RCWALayer, pixel_eps
from pqcsim.optics.tmm import tmm

LAM = 1550.0


@pytest.mark.parametrize("eps_gap", [None, 2.5, 30.0])
def test_uniform_layers_reproduce_tmm_exactly(eps_gap):
    nl = [1.45, 4.48 + 1.45j, 1.45, 0.55 + 11.5j]
    d = [20.0, 20.0, 270.0, 30.0]
    ref = tmm(LAM, 1.0, nl, d, 1.45)
    r = RCWA(LAM, (1.0, 1.0), orders=3, eps_gap=eps_gap).solve([RCWALayer.uniform(t, n) for t, n in zip(d, nl)])
    assert r.R == pytest.approx(ref.R, abs=1e-12)
    assert r.T == pytest.approx(ref.T, abs=1e-12)
    assert r.A_layers == pytest.approx(ref.A_layers, abs=1e-12)


def test_lossless_grating_conserves_energy_and_diffracts():
    eps = pixel_eps(3.48, 1.45, 2.0, 1.0, 128)          # period 2 um > lambda/n: orders propagate
    r = RCWA(LAM, (2.0, 2.0), orders=5).solve([RCWALayer(300, eps)])
    assert r.R + r.T == pytest.approx(1.0, abs=1e-12)
    assert r.A_layers == pytest.approx(0.0, abs=1e-12)
    centre = r.T_orders[5, 5]
    assert r.T - centre > 0.01                           # real diffraction into higher orders
    sub = RCWA(LAM, (1.0, 1.0), orders=5).solve([RCWALayer(300, pixel_eps(3.48, 1.45, 1.0, 0.5, 128))])
    assert sub.T == pytest.approx(sub.T_orders[5, 5], abs=1e-12)   # period < lambda / n_sub: zeroth order only


def test_square_pixels_are_polarisation_independent_and_absorb_only_in_pcm():
    eps = pixel_eps(4.48 + 1.45j, 1.45, 1.0, 0.6, 128)
    solver = RCWA(LAM, (1.0, 1.0), orders=4)
    layers = [RCWALayer.uniform(20, 1.45), RCWALayer(20, eps), RCWALayer.uniform(270, 1.45)]
    a, b = solver.solve(layers, (1, 0)), solver.solve(layers, (0, 1))
    assert a.R == pytest.approx(b.R, abs=1e-10) and a.A_layers == pytest.approx(b.A_layers, abs=1e-10)
    assert a.R + a.T + a.A_layers.sum() == pytest.approx(1.0, abs=1e-12)
    assert a.A_layers[0] == pytest.approx(0, abs=1e-12) and a.A_layers[2] == pytest.approx(0, abs=1e-12)
    assert a.A_layers[1] > 0


def test_split_layer_rows_sum_to_layer():
    eps = pixel_eps(4.48 + 1.45j, 1.45, 1.0, 0.6, 128)
    solver = RCWA(LAM, (1.0, 1.0), orders=3)
    whole = solver.solve([RCWALayer(30, eps), RCWALayer.uniform(100, 0.55 + 11.5j)])
    split = solver.solve([RCWALayer(10, eps)] * 3 + [RCWALayer.uniform(100, 0.55 + 11.5j)])
    assert split.A_layers[:3].sum() == pytest.approx(whole.A_layers[0], abs=1e-12)
    assert split.R == pytest.approx(whole.R, abs=1e-12)


def test_fill_factor_limits():
    """Pixel filling the whole cell = uniform film; vanishing pixel = no film."""
    full = RCWA(LAM, (1.0, 1.0), orders=3).solve([RCWALayer(20, pixel_eps(4.48 + 1.45j, 1.45, 1.0, 1.01, 64))])
    ref = tmm(LAM, 1.0, [4.48 + 1.45j], [20.0], 1.45)
    assert full.R == pytest.approx(ref.R, abs=1e-10)
    none = RCWA(LAM, (1.0, 1.0), orders=3).solve([RCWALayer(20, pixel_eps(4.48 + 1.45j, 1.45, 1.0, 0.0, 64))])
    assert none.A_layers[0] == pytest.approx(0.0, abs=1e-12)


def test_li_factorization_converges_faster_to_the_same_limit():
    eps = pixel_eps(4.48 + 1.45j, 1.45, 1.0, 0.6, 128)
    layers = [RCWALayer.uniform(20, 1.45), RCWALayer(20, eps), RCWALayer.uniform(270, 1.45)]
    A = {f: [RCWA(LAM, (1.0, 1.0), orders=M, factorization=f).solve(layers, (1, 0)).A_layers[1] for M in (3, 5, 7)]
         for f in ("li", "laurent")}
    d_li = abs(A["li"][2] - A["li"][1])
    d_la = abs(A["laurent"][2] - A["laurent"][1])
    assert d_li < d_la                                        # Li changes less between truncations
    assert abs(A["li"][2] - A["li"][1]) < abs(A["li"][1] - A["li"][0])
    # Laurent approaches the Li value from the other side as orders grow
    assert abs(A["laurent"][2] - A["li"][2]) < abs(A["laurent"][0] - A["li"][2])


def test_li_rule_exact_for_uniform_and_one_dimensional_patterns():
    """For eps independent of y, Li's E_yy equals the plain convolution (Laurent along y is exact)."""
    r = RCWA(LAM, (1.0, 1.0), orders=3)
    x = (np.arange(64) + 0.5) / 64
    eps = np.where(np.abs(x - 0.5)[:, None] < 0.2, 9.0 + 1j, 2.1) * np.ones((1, 64))
    assert r._li_matrix(eps, 1) == pytest.approx(r._conv(eps), abs=1e-12)
    inv_rule = np.linalg.inv(r._conv(1.0 / eps))
    assert r._li_matrix(eps, 0) == pytest.approx(inv_rule, abs=1e-10)   # across x walls: inverse rule


# ----------------------------------------------------------------------------- oblique incidence and fields
@pytest.mark.parametrize("theta_deg", [10, 35, 60])
@pytest.mark.parametrize("n2", [1.5, 3.48, 2.0 + 0.8j])
def test_oblique_single_interface_matches_fresnel(theta_deg, n2):
    th = np.radians(theta_deg)
    kx = np.sin(th)
    c1 = np.cos(th)
    c2 = np.sqrt(1 - (kx / n2) ** 2 + 0j)
    rs = (c1 - n2 * c2) / (c1 + n2 * c2)
    rp = (n2 * c1 - c2) / (n2 * c1 + c2)
    r = RCWA(LAM, (1.0, 1.0), orders=1, n_inc=1.0, n_sub=n2, k_parallel=(kx, 0.0))
    Rs = r.solve([], pol=(0.0, 1.0)).R
    Rp = r.solve([], pol=(1.0, 0.0)).R
    assert Rs == pytest.approx(abs(rs) ** 2, abs=1e-10)
    assert Rp == pytest.approx(abs(rp) ** 2, abs=1e-10)


def test_oblique_lossless_grating_conserves_energy():
    eps = pixel_eps(3.48, 1.45, 1.2, 0.6, 128)
    r = RCWA(LAM, (1.2, 1.2), orders=4, k_parallel=(0.31, -0.17))
    for pol in [(1.0, 0.0), (0.0, 1.0), (0.6, 0.8j)]:
        res = r.solve([RCWALayer(250, eps)], pol=pol)
        assert res.R + res.T == pytest.approx(1.0, abs=1e-11)


def _field_absorption(eps, n_abs, orders, patterned):
    from pqcsim._compat import trapezoid
    r = RCWA(LAM, (1.0, 1.0), orders=orders, k_parallel=(0.2, 0.1))
    layers = [RCWALayer.uniform(20, 1.45), RCWALayer(30, eps), RCWALayer.uniform(200, 1.45)]
    i0 = int(np.where((r.p == 0) & (r.q == 0))[0][0])
    e_src = np.zeros(2 * r.N, complex)
    e_src[i0] = 1.0
    ref = r.solve(layers, e_src=e_src)
    g = np.asarray(eps).shape[0] if patterned else 40
    xs = (np.arange(g) / g + 0.5) % 1.0 - 0.5                           # the eps grid points
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    z = np.linspace(0, 30, 21)
    E, inc = r.layer_fields(layers, 1, e_src, z, X.ravel(), Y.ravel())
    if patterned:
        im = np.imag(eps)[np.rint((X.ravel() % 1) * g).astype(int) % g, np.rint((Y.ravel() % 1) * g).astype(int) % g]
    else:
        im = np.full(X.size, (n_abs ** 2).imag)
    k0 = 2 * np.pi / (LAM * 1e-9)
    A = k0 * trapezoid(((np.abs(E) ** 2).sum(axis=1) * im[None, :]).mean(axis=1), z * 1e-9) / inc
    return A, ref.A_layers[1]


def test_internal_field_absorption_matches_flux_uniform():
    """k0 Im(eps) ∫|E|^2 dV over the cell equals the flux-difference absorption (oblique incidence)."""
    n_abs = 4.48 + 1.45j
    A, A_flux = _field_absorption(n_abs ** 2, n_abs, 4, False)
    assert A == pytest.approx(A_flux, rel=2e-3)


def test_internal_field_absorption_converges_for_pixels():
    """Patterned layer: the field reconstruction (D-based normal components, Li-consistent) converges to the
    flux absorption as the number of orders grows (-5.2 % at 3 orders, -2.3 % at 8 orders)."""
    n_abs = 4.48 + 1.45j
    eps = pixel_eps(n_abs, 1.45, 1.0, 0.6, 96)
    errs = []
    for M in (3, 5, 7):
        A, A_flux = _field_absorption(eps, n_abs, M, True)
        errs.append(abs(A / A_flux - 1))
    assert errs[0] > errs[1] > errs[2] and errs[2] < 0.04
