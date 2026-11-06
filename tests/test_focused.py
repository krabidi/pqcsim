"""Focused-beam Bloch superposition on pixel arrays."""
import numpy as np
import pytest

from pqcsim.optics.focused import focused_pixel_absorption, naive_pixel_absorption
from pqcsim.optics.rcwa import RCWA, RCWALayer, pixel_eps

LAM, P, SIDE = 780.0, 0.8, 0.48
N_PIX = 6.3 + 2.0j                     # a strongly absorbing pixel


def layers():
    return [RCWALayer.uniform(20, 1.45), RCWALayer(20, pixel_eps(N_PIX, 1.45, P, SIDE, 96)),
            RCWALayer.uniform(270, 1.45)]


def test_shares_converge_in_bz_sampling_and_are_symmetric():
    sites = [(0, 0), (1, 0), (0, 1), (-1, 0), (1, 1)]
    a = focused_pixel_absorption(LAM, P, layers(), 1, N_PIX, SIDE, 0.5, sites, n_bz=4, orders=3, n_xy=6, n_z=3)
    b = focused_pixel_absorption(LAM, P, layers(), 1, N_PIX, SIDE, 0.5, sites, n_bz=6, orders=3, n_xy=6, n_z=3)
    assert b[(0, 0)] == pytest.approx(a[(0, 0)], rel=0.03)
    assert b[(1, 0)] == pytest.approx(b[(-1, 0)], rel=1e-6)                 # x-polarised beam: +-x symmetric
    assert b[(0, 0)] > b[(1, 0)] > b[(1, 1)] > 0


def test_exact_total_matches_plane_wave_for_wide_spectrum_limit():
    """Supercell total absorbed (flux) for a focused beam lies within the angular spread of plane-wave values."""
    sites = [(0, 0)]
    _, tot, _ = focused_pixel_absorption(LAM, P, layers(), 1, N_PIX, SIDE, 0.5, sites, n_bz=4, orders=3,
                                         n_xy=4, n_z=2, return_total=True)
    A0 = RCWA(LAM, (P, P), orders=3).solve(layers(), pol=(1, 0)).A_layers[1]
    assert 0.5 * A0 < tot < 1.5 * A0


def test_naive_model_geometry():
    nv = naive_pixel_absorption(0.4, 0.36, 0.48, 0.8, 1e6, [(0, 0)])
    assert nv[(0, 0)] == pytest.approx(0.4 / 0.36 * (0.48 / 1e6) ** 2 * 2 / np.pi, rel=1e-3)   # tiny pixel, huge beam


def test_focused_reflectance_limits():
    """Uniform stack: a wide beam reflects like a plane wave at normal incidence; a narrow collection NA
    collects less; with no NA limit every propagating order counts (R <= 1)."""
    from pqcsim.optics.focused import focused_reflectance
    from pqcsim.optics.rcwa import RCWA, RCWALayer
    from pqcsim.optics.tmm import tmm
    n_gst, n_au = 4.5 + 1.2j, 0.15 + 4.9j
    layers = [RCWALayer.uniform(20, 1.45), RCWALayer.uniform(20, n_gst), RCWALayer.uniform(270, 1.45),
              RCWALayer.uniform(100, n_au)]
    R0 = tmm(780.0, 1.0, [1.45, n_gst, 1.45, n_au], [20, 20, 270, 100], 1.45).R
    # the k grid (spacing lam / (period n_bz)) must resolve the beam's angular spectrum (~ 2 / (k0 w))
    R_wide = focused_reflectance(780.0, 6.0, layers, 6.0, n_bz=10, orders=0)
    assert R_wide == pytest.approx(R0, abs=1.5e-3)
    R_all = focused_reflectance(780.0, 1.0, layers, 0.5, n_bz=6, orders=2)
    R_na = focused_reflectance(780.0, 1.0, layers, 0.5, n_bz=6, orders=2, na_collect=0.3)
    assert 0 < R_na < R_all <= 1.0


def _angular_spectrum_R(lam_nm, n_layers, d_nm, n_sub, w_um, na, n_grid=1201):
    """Independent reference: one objective of NA na illuminates and collects; x-polarised tangential Gaussian
    spectrum; s/p decomposition with an oblique characteristic-matrix TMM; fine midpoint grid over the pupil."""
    k0 = 2 * np.pi / (lam_nm * 1e-9)

    def R(kap, pol):
        kap = kap.astype(complex)
        n_all = [1.0] + [complex(n) for n in n_layers] + [complex(n_sub)]
        kz = [np.sqrt(n ** 2 - kap ** 2) for n in n_all]
        Y = [k if pol == "s" else n ** 2 / k for k, n in zip(kz, n_all)]
        m11, m12, m21, m22 = (np.ones_like(kap), np.zeros_like(kap), np.zeros_like(kap), np.ones_like(kap))
        for j, d in enumerate(d_nm, start=1):
            dl = k0 * kz[j] * d * 1e-9
            c, s = np.cos(dl), np.sin(dl)
            m11, m12, m21, m22 = (m11 * c + m12 * (-1j * Y[j] * s), m11 * (-1j * s / Y[j]) + m12 * c,
                                  m21 * c + m22 * (-1j * Y[j] * s), m21 * (-1j * s / Y[j]) + m22 * c)
        B, C = m11 + m12 * Y[-1], m21 + m22 * Y[-1]
        return np.abs((Y[0] * B - C) / (Y[0] * B + C)) ** 2

    g = (np.arange(n_grid) + 0.5) / n_grid * 2 * na - na
    KX, KY = np.meshgrid(g, g)
    kap = np.hypot(KX, KY)
    m = kap < na
    kap, phi = kap[m], np.arctan2(KY, KX)[m]
    a2 = np.exp(-(kap * k0) ** 2 * (w_um * 1e-6) ** 2 / 2)
    ct = np.sqrt(1 - kap ** 2)
    fs, fp = a2 * np.sin(phi) ** 2 * ct, a2 * np.cos(phi) ** 2 / ct      # flux of the s and p parts of E_t = x
    return float((fs * R(kap, "s") + fp * R(kap, "p")).sum() / (fs + fp).sum())


@pytest.mark.parametrize("w_um", [1.0, 0.55])
def test_focused_reflectance_same_pupil_matches_angular_spectrum(w_um):
    """Uniform film, one objective (na_illum = na_collect): the Bloch sum equals the independent angular-spectrum
    TMM integral already on a coarse k grid (k-step 0.12 k0)."""
    from pqcsim.optics.focused import focused_reflectance
    lam, n_gst, n_au = 1550.0, 6.1 + 0.15j, 0.52 + 10.7j
    nl, d = [1.45, n_gst, 1.45, n_au], [20, 20, 270, 100]
    layers = [RCWALayer.uniform(t, n) for t, n in zip(d, nl)]
    na = min(lam * 1e-3 / (np.pi * w_um), 0.95)
    ref = _angular_spectrum_R(lam, nl, d, 1.45, w_um, na)
    R = focused_reflectance(lam, 2.5, layers, w_um, n_bz=5, orders=2, na_collect=na, na_illum=na)
    assert R == pytest.approx(ref, rel=3e-3)
