import math

import numpy as np
import pytest

from pqcsim.analysis.uncertainty import Param, latin_hypercube, propagate, sobol, summary


def test_sobol_matches_ishigami_analytic_indices():
    a, b = 7.0, 0.1
    params = [Param(f"x{i}", -math.pi, math.pi) for i in range(3)]

    def ishigami(X):
        return np.sin(X[0]) + a * np.sin(X[1]) ** 2 + b * X[2] ** 4 * np.sin(X[0])

    S1, ST = sobol(ishigami, params, n=2 ** 14, seed=1)
    V = a ** 2 / 8 + b * math.pi ** 4 / 5 + b ** 2 * math.pi ** 8 / 18 + 0.5
    V1 = 0.5 * (1 + b * math.pi ** 4 / 5) ** 2
    V2 = a ** 2 / 8
    V13 = b ** 2 * math.pi ** 8 * (1 / 18 - 1 / 50)
    assert S1 == pytest.approx([V1 / V, V2 / V, 0.0], abs=0.02)
    assert ST == pytest.approx([(V1 + V13) / V, V2 / V, V13 / V], abs=0.02)


def test_latin_hypercube_marginals_and_propagation():
    params = [Param("u", 2.0, 4.0), Param("g", 1e-9, 1e-7, "loguniform"), Param("n", 5.0, 0.5, "normal")]
    X = latin_hypercube(params, 4000, seed=3)
    assert X[:, 0].min() >= 2.0 and X[:, 0].max() <= 4.0 and X[:, 0].mean() == pytest.approx(3.0, abs=0.01)
    assert np.log10(X[:, 1]).mean() == pytest.approx(-8.0, abs=0.01)
    assert X[:, 2].std() == pytest.approx(0.5, rel=0.03)
    out = propagate(lambda p: {"y": p["u"] * 2}, params, 200, seed=1)
    s = summary(out["y"])
    assert 4.0 <= s["p05"] < s["median"] < s["p95"] <= 8.0
