"""Device -> array -> circuit (classical and quantum) with one PCM material description."""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]

import numpy as np

from pqcsim.arrays.crossbar import PCMArray
from pqcsim.circuits.network import BeamSplitter, LinearNetwork, pcm_segment
from pqcsim.devices.pcm import PCMCell
from pqcsim.electronics.readout import ADC, Photodetector
from pqcsim.materials import load_materials
from pqcsim.quantum.lossy import simulate_fock

mats = {m.name: m for m in load_materials(str(ROOT / "data" / "materials.json"))}
gete = mats["GeTe"]

# 1) Device: photomem stack, SET pulse trains (100 ns pulses at 1550 nm)
cell = PCMCell(gete, [1.45, "active", 1.45, 0.55 + 11.5j], [20.0, 20.0, 270.0, 100.0], readout="1-R")
print("1) Device: GeTe cell, SET pulse trains, 100 ns pulses")
for F in (108.0, 115.0):
    x, levels = 0.0, []
    for _ in range(8):
        out = cell.apply_pulse(x, F, 1550.0, 100e-9)
        x = out.x
        levels.append(f"{x:.3f}")
    print(f"   F = {F:5.1f} J/m^2: x = {' '.join(levels)}   (last T_peak {out.T_peak:.0f} K)")
print("   -> crystallization raises A_active, so heating per pulse grows: gradual at 108, runaway at 115")

# 2) Array: 4x4 crossbar, program cell (i, j) with i + j pulses, optical-tail crosstalk, noisy readout
arr = PCMArray(cell, 4, 4, pitch_um=1.2, spot_radius_um=1.0)
target = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        before = arr.X.copy()
        for _ in range(i + j):
            arr.write(i, j, 108.0, 1550.0, 100e-9)
        target[i, j] = arr.X[i, j] - before[i, j]
print("\n2) Array: 4x4, cell (i, j) gets i + j pulses at 108 J/m^2, 1.2 um pitch, 1.0 um spot radius")
print("   final x:\n" + "\n".join("     " + " ".join(f"{v:.4f}" for v in row) for row in arr.X))
print(f"   largest change caused by neighbours' writes (crosstalk): {np.abs(arr.X - target).max():.2e}")
P_in = np.full(4, 1e-4)                                    # 0.1 mW per column
pd, adc = Photodetector(responsivity=1.0, bandwidth=1e9, R_f=2e3), ADC(bits=8, v_fullscale=1.0)
y = arr.mvm(P_in)
codes = adc.codes(pd.voltage(y, rng=np.random.default_rng(0)))
print(f"   MVM output power (uW): {np.round(y * 1e6, 2)}   ADC codes: {codes}")

# 3) Circuit: the same GeTe as a waveguide PCM phase shifter in a Mach-Zehnder, classical and quantum
print("\n3) Circuit: 2 um GeTe patch (Gamma = 0.05) in one MZI arm, input |1,1>")
print("     x   |t|^2   phase(rad)  P_classical(out0)  P(1,1)   P(2,0)+P(0,2)  P(photon lost)")
for xs in (0.0, 0.25, 0.5, 1.0):
    seg = pcm_segment(0, gete, xs, 2.0, 1550.0, confinement=0.05, n_ref=1.45)
    net = LinearNetwork(2, [BeamSplitter(0, 1), *seg, BeamSplitter(0, 1)])
    t = net.matrix()
    p = simulate_fock(net, [1, 1])
    Pc = abs(net.propagate_fields([1, 0])[0]) ** 2
    lost = sum(v for k, v in p.items() if sum(k) < 2)
    print(f"   {xs:4.2f}  {seg[1].eta:.3f}   {seg[0].phi:8.3f}    {Pc:12.4f}    {p.get((1, 1), 0):.4f}   "
          f"{p.get((2, 0), 0) + p.get((0, 2), 0):.4f}        {lost:.4f}")
