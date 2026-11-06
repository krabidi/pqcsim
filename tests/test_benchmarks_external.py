"""Benchmarks against established, independently developed codes.

Skipped automatically when a reference package is not installed:
  tmm        (S. Byrnes)               thin-film transfer matrices
  thewalrus  (Xanadu)                  hafnians, permanents, Gaussian-state photon statistics
  perceval   (Quandela)                linear-optical multiphoton simulation (SLOS)
  fipy       (NIST)                    finite-volume PDE solver, used for transient heat conduction
"""
import itertools
import math

import numpy as np
import pytest
from scipy.stats import unitary_group

from pqcsim.circuits.decompose import unitary_elements
from pqcsim.circuits.network import LinearNetwork, Loss
from pqcsim.optics.tmm import tmm as pq_tmm
from pqcsim.quantum.gaussian import GaussianState
from pqcsim.quantum.lossy import simulate_fock


# ----------------------------------------------------------------------------- thin films vs tmm
@pytest.mark.parametrize("seed", range(6))
def test_tmm_matches_byrnes_tmm(seed):
    tmm_pkg = pytest.importorskip("tmm")
    rng = np.random.default_rng(seed)
    N = int(rng.integers(1, 6))
    layers = [complex(rng.uniform(1.3, 4.5), rng.choice([0.0, rng.uniform(0, 3)])) for _ in range(N)]
    d = list(rng.uniform(5, 300, N))
    lam = float(rng.uniform(800, 1700))
    n_sub = complex(rng.uniform(1.0, 3.5), rng.choice([0.0, rng.uniform(0, 0.5)]))
    ours = pq_tmm(lam, 1.0, layers, d, n_sub)
    ref = tmm_pkg.coh_tmm("s", [1.0, *layers, n_sub], [np.inf, *d, np.inf], 0.0, lam)
    assert ours.R == pytest.approx(ref["R"], abs=1e-12)
    assert ours.T == pytest.approx(ref["T"], abs=1e-12)
    assert ours.r == pytest.approx(ref["r"], abs=1e-12)
    absorp = tmm_pkg.absorp_in_each_layer(ref)          # [incident, layers..., substrate]
    assert ours.A_layers == pytest.approx(np.array(absorp[1:-1]), abs=1e-12)


# ----------------------------------------------------------------------------- Gaussian vs thewalrus
def test_hafnian_and_permanent_match_thewalrus():
    tw = pytest.importorskip("thewalrus")
    import photonics_core as core
    from pqcsim.quantum.gaussian import hafnian
    from pqcsim.quantum.reference import permanent
    rng = np.random.default_rng(1)
    for n in (2, 4, 6, 8, 10):
        B = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        B = B + B.T
        assert hafnian(B) == pytest.approx(tw.hafnian(B), rel=1e-10)
        assert core.hafnian(B) == pytest.approx(tw.hafnian(B), rel=1e-10)
    for n in (1, 3, 5, 8):
        A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        assert permanent(A) == pytest.approx(tw.perm(A), rel=1e-10)
        assert core.permanent(A) == pytest.approx(tw.perm(A), rel=1e-10)


def _random_gaussian_state(seed):
    rng = np.random.default_rng(seed)
    s = GaussianState(3)
    s.two_mode_squeeze(0, 1, rng.uniform(0.2, 0.6), rng.uniform(0, 2 * np.pi))
    s.squeeze(2, rng.uniform(0.1, 0.5), rng.uniform(0, 2 * np.pi))
    U = unitary_group.rvs(3, random_state=seed)
    net = LinearNetwork(3, unitary_elements(U) + [Loss(0, 0.7), Loss(2, 0.45)])
    return s.apply_network(net)


@pytest.mark.parametrize("seed", range(3))
def test_gaussian_photon_statistics_match_thewalrus(seed):
    twq = pytest.importorskip("thewalrus.quantum")
    s = _random_gaussian_state(seed)
    cutoff = 4
    ref = twq.probabilities(s.mean, s.cov, cutoff, hbar=1)     # thewalrus: hbar = 1 -> vacuum cov I/2
    for pattern in itertools.product(range(cutoff), repeat=3):
        if sum(pattern) <= 5:
            assert s.fock_probability(list(pattern)) == pytest.approx(ref[pattern], abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_threshold_clicks_match_thewalrus(seed):
    tw = pytest.importorskip("thewalrus")
    s = _random_gaussian_state(seed)
    s.displace(1, 0.3 - 0.2j)
    for clicks in itertools.product([0, 1], repeat=3):
        ref = tw.threshold_detection_prob(s.mean, s.cov, np.array(clicks), hbar=1)
        assert s.click_probability(list(clicks)) == pytest.approx(float(np.real(ref)), abs=1e-10)


# ----------------------------------------------------------------------------- Fock vs Perceval
@pytest.mark.parametrize("seed,n_in", [(0, (1, 1, 0)), (1, (1, 1, 1)), (2, (2, 0, 1)), (3, (1, 0, 1, 1))])
def test_fock_probabilities_match_perceval(seed, n_in):
    pcvl = pytest.importorskip("perceval")
    m = len(n_in)
    U = unitary_group.rvs(m, random_state=seed)
    ours = simulate_fock(LinearNetwork(m, unitary_elements(U)), list(n_in))
    backend = pcvl.SLOSBackend()
    backend.set_circuit(pcvl.Unitary(pcvl.Matrix(U)))
    backend.set_input_state(pcvl.BasicState(list(n_in)))
    N = sum(n_in)
    for out in itertools.product(range(N + 1), repeat=m):
        if sum(out) == N:
            ref = backend.probability(pcvl.BasicState(list(out)))
            assert ours.get(out, 0.0) == pytest.approx(ref, abs=1e-12)


# ----------------------------------------------------------------------------- heat vs FiPy
def test_axisymmetric_heat_matches_fipy():
    """Gaussian-spot pulse in a PCM film on SiO2 over a heat sink, solved by FiPy on its own grid."""
    fipy = pytest.importorskip("fipy")
    from pqcsim.thermal.diffusion import StackHeatSolver, ThermalLayer, radial_grid
    w, E0, tp, t_end = 0.8e-6, 40.0, 50e-9, 300e-9
    d_pcm, d_ox, R = 20e-9, 300e-9, 4e-6
    ours = StackHeatSolver([ThermalLayer("pcm", 20, 0.5, 6140, 220, 4), ThermalLayer("ox", 300, 1.4, 2200, 740, 15)],
                           radial_grid(4, 0.02, 1.03), top="insulated", bottom="fixed", outer="fixed")
    res = ours.run(ours.gaussian_source(0, E0, w * 1e6), tp, t_end, rtol=1e-8)
    T_ours = res.layer_average(0, 0.0)

    nr, dz = 200, np.r_[np.full(4, d_pcm / 4), np.full(30, d_ox / 30)]
    mesh = fipy.CylindricalGrid2D(dr=R / nr, dz=dz[::-1], nr=nr, nz=dz.size)   # z up: oxide first, film on top
    r, z = (np.asarray(v) for v in mesh.cellCenters)
    in_pcm = z > d_ox
    k = fipy.CellVariable(mesh=mesh, value=np.where(in_pcm, 0.5, 1.4))
    rc = fipy.CellVariable(mesh=mesh, value=np.where(in_pcm, 6140 * 220, 2200 * 740))
    # absorbed power density: Gaussian spot spread uniformly through the film during the pulse
    q = fipy.CellVariable(mesh=mesh, value=np.where(in_pcm, E0 * np.exp(-2 * r ** 2 / w ** 2) / d_pcm / tp, 0.0))
    kf = k.harmonicFaceValue                    # series resistance across material interfaces
    centre = (r < R / nr) & in_pcm
    weights = np.asarray(rc.value) * np.asarray(mesh.cellVolumes)
    t_out = np.linspace(0, t_end, 61)

    def run_fipy(n_pulse):
        """FiPy's implicit Euler on a uniform step; returns the centre temperature on t_out."""
        T = fipy.CellVariable(mesh=mesh, value=0.0)
        T.constrain(0.0, mesh.facesBottom)
        T.constrain(0.0, mesh.facesRight)
        eq_on = fipy.TransientTerm(coeff=rc) == fipy.DiffusionTerm(coeff=kf) + q
        eq_off = fipy.TransientTerm(coeff=rc) == fipy.DiffusionTerm(coeff=kf)
        dt = tp / n_pulse
        ts, Ts = [0.0], [0.0]
        for step in range(int(round(t_end / dt))):
            (eq_on if step < n_pulse else eq_off).solve(var=T, dt=dt)
            ts.append((step + 1) * dt)
            Ts.append(float((np.asarray(T.value) * weights)[centre].sum() / weights[centre].sum()))
        return np.interp(t_out, ts, Ts)

    # FiPy's Euler error is first order in dt: Richardson extrapolation 2 T(dt/2) - T(dt) is second order
    T_ref = 2 * run_fipy(100) - run_fipy(50)
    T_ours_out = np.interp(t_out, res.t, T_ours)
    assert np.max(np.abs(T_ours_out - T_ref)) < 0.01 * T_ref.max()
    assert T_ours.max() == pytest.approx(T_ref.max(), rel=0.005)


# ----------------------------------------------------------------------------- RCWA vs grcwa
def test_rcwa_matches_grcwa_for_pcm_pixel_array():
    grcwa = pytest.importorskip("grcwa")
    from pqcsim.optics.rcwa import RCWA, RCWALayer, pixel_eps
    grcwa.set_backend("numpy")
    lam, P, side, nx = 1.55, 2.0, 1.0, 200
    eps = pixel_eps(4.48 + 1.45j, 1.45, P, side, nx)
    ours = RCWA(lam * 1e3, (P, P), orders=5, factorization="laurent").solve(      # same rule as grcwa
        [RCWALayer.uniform(20, 1.45), RCWALayer(20, eps), RCWALayer.uniform(270, 1.45)], pol=(1.0, 0.0))
    obj = grcwa.obj(121, [P, 0], [0, P], 1 / lam, 0.0, 0.0, verbose=0)
    for thick, e in ((0.1, 1.0), (0.020, 1.45 ** 2)):
        obj.Add_LayerUniform(thick, e)
    obj.Add_LayerGrid(0.020, nx, nx)
    obj.Add_LayerUniform(0.270, 1.45 ** 2)
    obj.Add_LayerUniform(0.1, 1.45 ** 2)
    obj.Init_Setup()
    obj.GridLayer_geteps(eps.flatten())
    obj.MakeExcitationPlanewave(1, 0, 0, 0, order=0)
    R, T = obj.RT_Solve(normalize=1)
    # same physics, different order truncation (square vs circular): agree to ~1e-3
    assert ours.R == pytest.approx(R, abs=1e-3)
    assert ours.T == pytest.approx(T, abs=1e-3)
    assert ours.A_layers.sum() == pytest.approx(1 - R - T, abs=1e-3)
