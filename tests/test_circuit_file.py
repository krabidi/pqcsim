import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from pqcsim.circuits.network import BeamSplitter, LinearNetwork, Loss, PhaseShift
from pqcsim.io.circuit_file import load_circuit
from pqcsim.quantum.distinguishable import gaussian_overlap

ROOT = Path(__file__).resolve().parent.parent


def write(tmp_path, spec, name="c.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(spec))
    return str(p)


def test_yaml_network_matches_python_construction(tmp_path):
    spec = {"modes": 2, "elements": [{"type": "beamsplitter", "modes": [0, 1], "theta": 0.3, "phi": 0.2},
                                     {"type": "phase", "mode": 1, "phi": 1.1},
                                     {"type": "loss", "mode": 0, "loss_dB": 3.0},
                                     {"type": "coupler", "modes": [0, 1], "split": 0.5}]}
    c = load_circuit(write(tmp_path, spec))
    ref = LinearNetwork(2, [BeamSplitter(0, 1, 0.3, 0.2), PhaseShift(1, 1.1), Loss(0, 10 ** -0.3),
                            BeamSplitter(0, 1, math.pi / 4)])
    assert c.network.matrix() == pytest.approx(ref.matrix(), abs=1e-12)


def test_matrix_and_ring_elements_round_trip(tmp_path):
    from scipy.stats import unitary_group
    from pqcsim.circuits.components import ring_add_drop
    U = unitary_group.rvs(3, random_state=4) * 0.9
    spec = {"modes": 3, "wavelength_nm": 1550.3,
            "elements": [{"type": "matrix", "modes": [0, 1, 2], "real": U.real.tolist(), "imag": U.imag.tolist()},
                         {"type": "ring_add_drop", "modes": [0, 2], "length_um": 50, "n_eff": 2.4, "n_g": 4.2,
                          "theta1": 0.3, "theta2": 0.3, "loss_dB_per_cm": 3}]}
    c = load_circuit(write(tmp_path, spec))
    R = np.eye(3, dtype=complex)
    R[np.ix_([0, 2], [0, 2])] = ring_add_drop(1550.3, 50, 2.4, 4.2, 0.3, 0.3, 3)
    assert c.network.matrix() == pytest.approx(R @ U, abs=1e-11)


def test_fock_source_with_detectors(tmp_path):
    spec = {"modes": 2, "elements": [{"type": "coupler", "modes": [0, 1], "split": 0.5}],
            "source": {"type": "fock", "occupation": [1, 1]},
            "detectors": [{"threshold": True}, {"threshold": True}]}
    p = load_circuit(write(tmp_path, spec)).run()
    assert p == pytest.approx({(1, 0): 0.5, (0, 1): 0.5})


def test_example_file_runs_and_matches_hand_calculation():
    c = load_circuit(str(ROOT / "examples" / "circuits" / "hom_pcm.yaml"))
    p = c.run()
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-9)
    # arms are diagonal before the 50:50 coupler: P_cc = eta0 eta1 eta_d^2 (1 - |S|^2) / 2 + O(dark)
    T_arms = LinearNetwork(2, c.network.elements[:-1]).matrix()
    eta0, eta1 = abs(T_arms[0, 0]) ** 2, abs(T_arms[1, 1]) ** 2
    S2 = abs(gaussian_overlap(0.2e-12, 2.0e12)) ** 2
    expected = eta0 * eta1 * 0.85 ** 2 * (1 - S2) / 2
    assert p[(1, 1)] == pytest.approx(expected, abs=5e-5)


def test_gaussian_source_clicks(tmp_path):
    spec = {"modes": 2, "elements": [], "source": {"type": "gaussian", "two_mode_squeezers": [{"modes": [0, 1], "mean_pairs": 0.1}]},
            "detectors": [{"efficiency": 1.0, "threshold": True}, {"efficiency": 1.0, "threshold": True}]}
    p = load_circuit(write(tmp_path, spec)).run()
    assert p[(1, 1)] == pytest.approx(0.1 / 1.1, abs=1e-12)        # P(pairs >= 1) = nbar / (1 + nbar)
    assert p[(1, 0)] == pytest.approx(0.0, abs=1e-12)


def test_errors_are_clear(tmp_path):
    with pytest.raises(ValueError, match="unknown element"):
        load_circuit(write(tmp_path, {"modes": 1, "elements": [{"type": "lens", "mode": 0}]}))
    with pytest.raises(ValueError, match="missing"):
        load_circuit(write(tmp_path, {"modes": 1, "elements": [{"type": "phase", "mode": 0}]}))
