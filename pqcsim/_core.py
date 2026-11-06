"""Locate and import the compiled C++ module (built into <repo>/build by CMake)."""
import sys
from pathlib import Path

try:
    import photonics_core as core  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
    try:
        import photonics_core as core  # noqa: F401
    except ImportError as exc:
        raise ImportError("photonics_core not built; run: cmake -S . -B build && cmake --build build") from exc
