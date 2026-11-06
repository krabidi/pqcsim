"""Uncertainty propagation and global sensitivity for device predictions.

Parameters are independent with uniform, log-uniform or normal distributions. Monte Carlo uses a
scrambled Latin hypercube (scipy.stats.qmc); variance-based Sobol indices (first order S1 and total ST)
use scipy.stats.sobol_indices (Saltelli/Jansen estimators). Validated on the Ishigami function in tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np
from scipy import stats
from scipy.stats import qmc


@dataclass(frozen=True)
class Param:
    name: str
    low: float
    high: float
    kind: str = "uniform"          # 'uniform', 'loguniform', or 'normal' (low = mean, high = std)

    def dist(self):
        if self.kind == "uniform":
            return stats.uniform(self.low, self.high - self.low)
        if self.kind == "loguniform":
            return stats.loguniform(self.low, self.high)
        if self.kind == "normal":
            return stats.norm(self.low, self.high)
        raise ValueError(f"unknown distribution {self.kind!r}")

    @property
    def nominal(self) -> float:
        if self.kind == "loguniform":
            return float(np.sqrt(self.low * self.high))
        return float(self.low if self.kind == "normal" else 0.5 * (self.low + self.high))


def latin_hypercube(params: Sequence[Param], n: int, seed: int = 0) -> np.ndarray:
    """(n, d) samples, each column distributed as its parameter."""
    u = qmc.LatinHypercube(d=len(params), seed=seed).random(n)
    return np.column_stack([p.dist().ppf(u[:, j]) for j, p in enumerate(params)])


def propagate(model: Callable[[Dict[str, float]], Dict[str, float]], params: Sequence[Param], n: int,
              seed: int = 0) -> Dict[str, np.ndarray]:
    """Evaluate model on n LHS samples; returns arrays of each output plus the inputs."""
    X = latin_hypercube(params, n, seed)
    outs: Dict[str, List[float]] = {}
    for row in X:
        res = model({p.name: v for p, v in zip(params, row)})
        for k, v in res.items():
            outs.setdefault(k, []).append(v)
    result = {k: np.asarray(v) for k, v in outs.items()}
    result.update({p.name: X[:, j] for j, p in enumerate(params)})
    return result


def summary(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, float)
    q = np.percentile(v, [5, 50, 95])
    return {"p05": q[0], "median": q[1], "p95": q[2], "mean": v.mean(), "std": v.std(ddof=1)}


def sobol(model_vec: Callable[[np.ndarray], np.ndarray], params: Sequence[Param], n: int, seed: int = 0):
    """First-order and total Sobol indices. model_vec takes an array (d, m) and returns (s, m) or (m,).
    n must be a power of two; cost is n (d + 2) model evaluations."""
    res = stats.sobol_indices(func=model_vec, n=n, dists=[p.dist() for p in params],
                              rng=np.random.default_rng(seed))
    return np.asarray(res.first_order), np.asarray(res.total_order)
