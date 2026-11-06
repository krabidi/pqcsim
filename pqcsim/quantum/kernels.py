"""Fast permanent / hafnian: C++ (photonics_core) when available, Python reference otherwise.
The Python implementations in reference.py / gaussian.py stay as independent checks."""
import numpy as np

try:
    from .._core import core as _core
    _HAVE_CORE = hasattr(_core, "permanent") and hasattr(_core, "hafnian")
except ImportError:          # pragma: no cover
    _HAVE_CORE = False


def permanent(a) -> complex:
    a = np.asarray(a, complex)
    if a.shape[0] == 0:
        return 1.0
    if _HAVE_CORE:
        return complex(_core.permanent(a))
    from .reference import permanent as py_perm
    return py_perm(a)


def hafnian(a) -> complex:
    a = np.asarray(a, complex)
    if a.shape[0] == 0:
        return 1.0
    if _HAVE_CORE and a.shape[0] <= 26:
        return complex(_core.hafnian(a))
    from .gaussian import hafnian as py_haf
    return py_haf(a)
