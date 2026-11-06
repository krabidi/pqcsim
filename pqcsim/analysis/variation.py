"""Fabrication variation: Monte Carlo over perturbed circuits, and yield."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

import numpy as np

from ..circuits.network import BeamSplitter, LinearNetwork, Loss, PhaseShift


def perturb_network(net: LinearNetwork, rng: np.random.Generator, sigma_theta: float = 0.0,
                    sigma_phase: float = 0.0, sigma_loss_dB: float = 0.0) -> LinearNetwork:
    """Copy of `net` with Gaussian errors on coupler angles, phases and losses (losses kept passive)."""
    out = LinearNetwork(net.n_modes)
    for el in net.elements:
        if isinstance(el, BeamSplitter) and sigma_theta:
            el = replace(el, theta=el.theta + rng.normal(0, sigma_theta))
        elif isinstance(el, PhaseShift) and sigma_phase:
            el = replace(el, phi=el.phi + rng.normal(0, sigma_phase))
        elif isinstance(el, Loss) and sigma_loss_dB:
            dB = -10 * np.log10(max(el.eta, 1e-300)) + rng.normal(0, sigma_loss_dB)
            el = replace(el, eta=float(10 ** (-max(dB, 0.0) / 10)))
        out.add(el)
    return out


def monte_carlo(sample: Callable[[np.random.Generator], float], n: int, seed: int = 0) -> np.ndarray:
    """Evaluate a metric on n random realisations; sample(rng) builds one device and returns the metric."""
    rng = np.random.default_rng(seed)
    return np.array([sample(rng) for _ in range(n)])


def yield_fraction(values: np.ndarray, threshold: float, higher_is_better: bool = True):
    """Fraction passing the spec, with its binomial standard error."""
    v = np.asarray(values)
    p = float(np.mean(v >= threshold if higher_is_better else v <= threshold))
    return p, float(np.sqrt(p * (1 - p) / v.size))
