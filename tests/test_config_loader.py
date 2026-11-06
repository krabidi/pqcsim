from pathlib import Path

from config_loader import load_circuit_from_yaml

CONFIG = Path(__file__).resolve().parent.parent / "spdc_bs_config.yaml"


def test_yaml_circuit_runs():
    circuit, detectors = load_circuit_from_yaml(CONFIG)

    photons = circuit.run()

    assert len(photons) == 2
    assert all(p.mode.endswith("_bsplit") for p in photons)
    assert len(detectors) == 1
    assert detectors[0].count() == 2
