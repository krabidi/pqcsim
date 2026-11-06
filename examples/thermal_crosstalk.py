"""Lumped RC vs full heat diffusion for a PCM cell, and write disturb vs array pitch."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]

import numpy as np

from pqcsim.arrays.crossbar import PCMArray
from pqcsim.devices.pcm import PCMCell
from pqcsim.materials import load_materials
from pqcsim.thermal.diffusion import StackHeatSolver, ThermalLayer, radial_grid
from pqcsim.thermal.response import PulseHeatResponse

gete = {m.name: m for m in load_materials(str(ROOT / "data" / "materials.json"))}["GeTe"]
cell = PCMCell(gete, [1.45, "active", 1.45, 0.55 + 11.5j], [20.0, 20.0, 270.0, 100.0], readout="1-R")

# Same stack for heat flow (illustrative thermal properties; replace with measured values)
layers = [ThermalLayer("SiO2 cap", 20, 1.4, 2200, 740, 2), ThermalLayer("GeTe", 20, 0.5, 6140, 220, 3),
          ThermalLayer("SiO2 spacer", 270, 1.4, 2200, 740, 8), ThermalLayer("Au", 100, 300, 19300, 129, 2),
          ThermalLayer("SiO2 substrate", 2000, 1.4, 2200, 740, 8)]
solver = StackHeatSolver(layers, radial_grid(30, 0.05, 1.06), top="insulated", bottom="fixed", outer="fixed")
hr = PulseHeatResponse(solver, active_layer=1, spot_radius_um=1.0, t_pulse=100e-9, t_end=2e-6,
                       heated_layers=[1, 3])                  # heat where light is absorbed: GeTe and Au

from pqcsim.thermal.response import optical_row_fractions
opt = cell.optics(0.0, 1550.0, "set")
frac = optical_row_fractions(1550.0, 1.0, cell.indices(0.0, 1550.0, "set"), cell.thicknesses_nm, opt, solver)
print("Peak temperature of an amorphous cell, 100 ns pulse, 1 um spot radius")
print("  F (J/m^2)   lumped RC (tau=20 ns)   heat diffusion   diffusion at r = 0.5 / 1.0 um")
for F in (150, 300, 600):
    lumped = cell.apply_pulse(0.0, F, 1550.0, 100e-9).T_peak
    diff = [300 + hr.rise(r, F * frac).max() for r in (0.0, 0.5, 1.0)]
    print(f"  {F:9.0f}   {lumped:18.0f} K   {diff[0]:12.0f} K   {diff[1]:6.0f} / {diff[2]:.0f} K")

print("\nWrite disturb on the nearest neighbour, one 600 J/m^2 pulse on the centre of a 3x3 array")
for pitch in (0.4, 0.5, 0.6, 0.8, 1.0):
    arr = PCMArray(cell, 3, 3, pitch_um=pitch, heat_response=hr)
    res = arr.write(1, 1, 600.0, 1550.0, 100e-9)
    print(f"  pitch {pitch:.1f} um: target x = {res.x:.4f}, neighbour x = {arr.X[0, 1]:.2e}, diagonal x = {arr.X[0, 0]:.2e}")
