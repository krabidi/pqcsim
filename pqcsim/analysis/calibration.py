"""Fitting model parameters to measurements (non-linear least squares with uncertainties)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares


@dataclass
class FitResult:
    params: np.ndarray
    stderr: np.ndarray
    covariance: np.ndarray
    chi2_reduced: float


def fit(model: Callable[[np.ndarray, np.ndarray], np.ndarray], x: np.ndarray, y: np.ndarray,
        p0: Sequence[float], sigma: Optional[np.ndarray] = None, bounds=(-np.inf, np.inf)) -> FitResult:
    """Weighted least squares. Covariance = (J^T W J)^{-1}, scaled by the reduced chi^2 when sigma is
    not given (unknown noise level)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones_like(y) if sigma is None else 1.0 / np.asarray(sigma, float)
    res = least_squares(lambda p: (model(x, p) - y) * w, np.asarray(p0, float), bounds=bounds,
                        x_scale="jac", xtol=1e-14, ftol=1e-14, gtol=1e-14)
    dof = max(y.size - res.x.size, 1)
    chi2r = float(np.sum(res.fun ** 2) / dof)
    J = res.jac
    cov = np.linalg.pinv(J.T @ J)
    if sigma is None:
        cov *= chi2r
    return FitResult(res.x, np.sqrt(np.diag(cov)), cov, chi2r)


def fit_crosstalk_matrix(P: np.ndarray, phi: np.ndarray, D: Sequence[float]) -> np.ndarray:
    """Least-squares K from measured phases phi[m, i] for heater power settings P[m, j]:
    phi = P K^T D, linear in K, so each row K_i is an ordinary least-squares problem."""
    P, phi = np.asarray(P, float), np.asarray(phi, float)
    D = np.asarray(D, float)
    K = np.linalg.lstsq(P, phi / D[None, :], rcond=None)[0].T
    return K
