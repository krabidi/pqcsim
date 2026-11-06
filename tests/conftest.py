import sys
from pathlib import Path

import pytest

# Make the compiled module (build/), config_loader.py and pqcsim (repo root) importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]


@pytest.fixture
def gete():
    from pqcsim.materials import load_materials
    return {m.name: m for m in load_materials(str(ROOT / "data" / "materials.json"))}["GeTe"]


@pytest.fixture
def cell(gete):
    """Default photomem stack: air | SiO2 20 | PCM 20 | SiO2 270 | Au 100 | SiO2."""
    from pqcsim.devices.pcm import PCMCell
    return PCMCell(gete, [1.45, "active", 1.45, 0.55 + 11.5j], [20.0, 20.0, 270.0, 100.0],
                   n_inc=1.0, n_sub=1.45, readout="1-R")


@pytest.fixture
def E_for_peak(cell):
    """Absorbed fluence (J/m^2) that gives a peak temperature T for a pulse of length tp."""
    import math
    th = cell.material.thermal
    C = th.rho * th.cp * cell.d_active_nm * 1e-9

    def f(T, tp):
        return (T - cell.T0) * C * (tp / th.tau_th) / -math.expm1(-tp / th.tau_th)
    return f
