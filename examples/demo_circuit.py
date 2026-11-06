import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]

from config_loader import load_circuit_from_yaml


def run_simulation(config_file):
    circuit, detectors = load_circuit_from_yaml(config_file)
    print("Running circuit simulation...")
    photons = circuit.run()

    print("\nFinal photon states:")
    for p in photons:
        print(f"  Mode: {p.mode}, Pol: {p.polarization}")

    print("\nDetector counts:")
    for i, d in enumerate(detectors):
        print(f"  Detector {i+1}: {d.count()} photons")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else ROOT / "spdc_bs_config.yaml"
    run_simulation(config)
