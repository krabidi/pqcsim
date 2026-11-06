"""Single-photon detector models applied to photon-number distributions and Gaussian states.

Per detector (independent, one time gate):
  efficiency eta    binomial thinning  P(k | n) = C(n, k) eta^k (1 - eta)^(n - k)  (identical to a loss)
  dark counts       Poisson counts with mean mu_dark added to the photocounts (classical noise)
  threshold         reports min(count, 1) (click / no click); PNR reports the count
Timing (continuous-wave pairs, Gaussian jitter): for a coincidence window of full width tau_c centred
on the true delay, with combined timing jitter sigma = sqrt(sigma_1^2 + sigma_2^2),
  fraction of true coincidences inside the window   f = erf(tau_c / (2 sqrt 2 sigma))
  accidental coincidence rate                         R_acc = R_1 R_2 tau_c   (uncorrelated singles)
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from .gaussian import GaussianState

Dist = Dict[Tuple[int, ...], float]


@dataclass(frozen=True)
class Detector:
    efficiency: float = 1.0
    dark_mean: float = 0.0          # mean dark counts per gate
    threshold: bool = False         # True: click/no-click; False: photon-number resolving
    max_dark: int = 6               # truncation of the dark-count Poisson sum

    def __post_init__(self):
        if not 0 <= self.efficiency <= 1 or self.dark_mean < 0:
            raise ValueError("efficiency must be in [0, 1] and dark_mean >= 0")

    def response(self, n: int) -> Dict[int, float]:
        """Distribution of the reported outcome for n incident photons."""
        out: Dict[int, float] = defaultdict(float)
        e = self.efficiency
        for k in range(n + 1):
            pk = math.comb(n, k) * e ** k * (1 - e) ** (n - k)
            if pk == 0:
                continue
            for d in range(self.max_dark + 1 if self.dark_mean > 0 else 1):
                pd = math.exp(-self.dark_mean) * self.dark_mean ** d / math.factorial(d)
                c = k + d
                out[min(c, 1) if self.threshold else c] += pk * pd
        return dict(out)


def detect(dist: Dist, detectors: Sequence[Detector]) -> Dist:
    """Apply independent detector models to a photon-number distribution over modes."""
    out: Dist = defaultdict(float)
    for pattern, p in dist.items():
        if len(pattern) != len(detectors):
            raise ValueError("one detector per mode is required")
        responses = [d.response(n).items() for d, n in zip(detectors, pattern)]
        for combo in itertools.product(*responses):
            q = p
            for _, pr in combo:
                q *= pr
            out[tuple(o for o, _ in combo)] += q
    return dict(out)


def gaussian_clicks(state: GaussianState, clicks: Sequence[int], efficiency: Sequence[float],
                    dark_mean: Sequence[float]) -> float:
    """Exact threshold-detector click probability for a Gaussian state with efficiency and dark counts.
    No-click on a set V requires vacuum after the efficiency loss and no dark count:
        P(no click on V) = P_vac,eta(V) prod_{v in V} e^{-mu_v}."""
    from ..circuits.network import LinearNetwork, Loss
    lossy = GaussianState(state.n)
    lossy.cov, lossy.mean = state.cov.copy(), state.mean.copy()
    lossy.apply_network(LinearNetwork(state.n, [Loss(m, efficiency[m]) for m in range(state.n)]))
    dark = [m for m, c in enumerate(clicks) if c == 0]
    lit = [m for m, c in enumerate(clicks) if c == 1]
    total = 0.0
    for k in range(len(lit) + 1):
        for E in itertools.combinations(lit, k):
            V = dark + list(E)
            total += (-1) ** k * lossy.vacuum_probability(V) * math.exp(-sum(dark_mean[v] for v in V))
    return total


def coincidence_window_fraction(window_s: float, jitter1_s: float, jitter2_s: float) -> float:
    sigma = math.hypot(jitter1_s, jitter2_s)
    if sigma == 0:
        return 1.0
    return math.erf(window_s / (2 * math.sqrt(2) * sigma))


def accidental_rate(rate1_hz: float, rate2_hz: float, window_s: float) -> float:
    return rate1_hz * rate2_hz * window_s


def coincidence_to_accidental(true_rate_hz: float, rate1_hz: float, rate2_hz: float, window_s: float,
                              jitter1_s: float = 0.0, jitter2_s: float = 0.0) -> float:
    """CAR = (detected true coincidences + accidentals) / accidentals."""
    acc = accidental_rate(rate1_hz, rate2_hz, window_s)
    return (true_rate_hz * coincidence_window_fraction(window_s, jitter1_s, jitter2_s) + acc) / acc
