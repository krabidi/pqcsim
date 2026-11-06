"""Exact Fock-state simulation of lossy linear networks.

Each Loss(mode, eta) element is replaced by a beamsplitter with cos^2(theta) = eta that couples the
mode to a fresh vacuum ancilla mode; tracing the ancillas out gives the exact mixed output state
(Stinespring dilation of the pure-loss channel). Ancillas are distinct for every lossy element,
so no unphysical interference between lost photons can occur.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Sequence, Tuple

from .._core import core
from ..circuits.network import BeamSplitter, Loss, LinearNetwork, PhaseShift


def simulate_fock(network: LinearNetwork, occupation: Sequence[int],
                  tol: float = 1e-15) -> Dict[Tuple[int, ...], float]:
    """Photon-number detection probabilities on the network's modes for a Fock-state input."""
    if len(occupation) != network.n_modes:
        raise ValueError("occupation length must equal the number of network modes")
    n = network.n_modes
    lossy = [el for el in network.elements if isinstance(el, Loss) and el.eta < 1.0]
    circuit = core.OpticalCircuit(n + len(lossy))
    ancilla = n
    for el in network.elements:
        if isinstance(el, BeamSplitter):
            circuit.beamsplitter(el.i, el.j, el.theta, el.phi)
        elif isinstance(el, PhaseShift):
            circuit.phase_shift(el.mode, el.phi)
        elif isinstance(el, Loss):
            if el.eta < 1.0:
                # Real mixing (phi = -pi/2 gives [[c, -s], [s, c]]); keeps the system amplitude sqrt(eta).
                circuit.beamsplitter(el.mode, ancilla, math.acos(math.sqrt(el.eta)), -math.pi / 2)
                ancilla += 1
        else:
            raise TypeError(f"unknown element {el!r}")
    state = circuit.run(core.FockState(list(occupation) + [0] * len(lossy)))
    probs: Dict[Tuple[int, ...], float] = defaultdict(float)
    for occ, p in state.probabilities().items():
        probs[occ[:n]] += p
    return {k: v for k, v in probs.items() if v > tol}
