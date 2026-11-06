import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]

from config_loader import load_circuit_from_yaml

circuit, _ = load_circuit_from_yaml(ROOT / "spdc_bs_config.yaml")
photons = circuit.run()

print(f"\n[ Number of photons: {len(photons)} ]")
for p in photons:
    print(f"Mode: {p.mode}, Pol: {p.polarization}, Pair ID: {p.pair_id}, Time: {p.timestamp}")
