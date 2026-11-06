"""Hong-Ou-Mandel interference: two photons on a beamsplitter of varying reflectivity."""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]

import photonics_core as pc

print("Input |1,1> on a beamsplitter with transmission T = cos^2(theta)\n")
print(f"{'T':>6}  {'P(1,1)':>8}  {'P(2,0)':>8}  {'P(0,2)':>8}")
for k in range(11):
    theta = k * math.pi / 20
    out = pc.OpticalCircuit(2).beamsplitter(0, 1, theta).run(pc.FockState([1, 1]))
    print(f"{math.cos(theta) ** 2:6.2f}  {out.probability([1, 1]):8.4f}  "
          f"{out.probability([2, 0]):8.4f}  {out.probability([0, 2]):8.4f}")

out = pc.OpticalCircuit(2).beamsplitter(0, 1).run(pc.FockState([1, 1]))
print(f"\nAt 50:50 the photons always leave together (bunching):\n  {out}")
print("1000 detection samples:", {o: n for o in out.probabilities()
                                  for n in [out.sample(1000, seed=3).count(o)]})
