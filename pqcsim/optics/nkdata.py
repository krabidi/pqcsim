"""Measured optical constants in the refractiveindex.info YAML format.

Supported data types: 'tabulated nk', 'tabulated n', 'tabulated k', 'formula 1' (Sellmeier,
n^2 - 1 = C0 + sum_i B_i lam^2 / (lam^2 - C_i^2)) and 'formula 2' (n^2 - 1 = C0 + sum B_i lam^2 / (lam^2 - C_i)).
Wavelengths are in micrometres in the files and in nm in this API. Interpolation is linear and no
extrapolation outside the measured range is done (an error is raised instead).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "nk"


@dataclass
class NKData:
    name: str
    references: str
    tables: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    formulas: List[Tuple[str, np.ndarray, Tuple[float, float]]] = field(default_factory=list)

    def _range(self):
        lo, hi = [], []
        for _, t in self.tables:
            lo.append(t[0, 0]); hi.append(t[-1, 0])
        for _, _, r in self.formulas:
            lo.append(r[0]); hi.append(r[1])
        return max(lo), min(hi)

    @property
    def range_nm(self):
        lo, hi = self._range()
        return lo * 1e3, hi * 1e3

    def nk(self, lam_nm: float) -> complex:
        lam = lam_nm * 1e-3
        lo, hi = self._range()
        if not lo - 1e-12 <= lam <= hi + 1e-12:
            raise ValueError(f"{self.name}: {lam_nm} nm outside measured range {lo*1e3:.0f}-{hi*1e3:.0f} nm")
        n, k = None, 0.0
        for kind, coeff, _ in self.formulas:
            c = coeff
            if kind == "formula 1":
                s = c[0] + sum(c[i] * lam ** 2 / (lam ** 2 - c[i + 1] ** 2) for i in range(1, len(c) - 1, 2))
            else:
                s = c[0] + sum(c[i] * lam ** 2 / (lam ** 2 - c[i + 1]) for i in range(1, len(c) - 1, 2))
            n = float(np.sqrt(1 + s))
        for kind, t in self.tables:
            if kind == "tabulated nk":
                n = float(np.interp(lam, t[:, 0], t[:, 1]))
                k = float(np.interp(lam, t[:, 0], t[:, 2]))
            elif kind == "tabulated n":
                n = float(np.interp(lam, t[:, 0], t[:, 1]))
            elif kind == "tabulated k":
                k = float(np.interp(lam, t[:, 0], t[:, 1]))
        if n is None:
            raise ValueError(f"{self.name}: no refractive index data")
        return complex(n, k)


def load_nk(path_or_name: str) -> NKData:
    """Load a YAML file (path, or a file name inside data/nk)."""
    p = Path(path_or_name)
    if not p.exists():
        p = DATA_DIR / (path_or_name if path_or_name.endswith(".yml") else path_or_name + ".yml")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    out = NKData(name=p.stem, references=" ".join(str(d.get("REFERENCES", "")).split()))
    for block in d["DATA"]:
        kind = block["type"]
        if kind.startswith("tabulated"):
            rows = [[float(v) for v in line.split()] for line in block["data"].strip().splitlines()]
            out.tables.append((kind, np.array(rows)))
        elif kind in ("formula 1", "formula 2"):
            rng = tuple(float(v) for v in str(block["wavelength_range"]).split())
            out.formulas.append((kind, np.array([float(v) for v in str(block["coefficients"]).split()]), rng))
        else:
            raise NotImplementedError(f"{p.name}: data type {kind!r} not supported")
    return out
