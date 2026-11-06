"""Builds the C++ Fock engine (photonics_core) with the vendored pybind11 headers."""
from pathlib import Path

from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent
pybind_include = ROOT / "extern" / "pybind11" / "include"
if not pybind_include.exists():
    raise RuntimeError("pybind11 submodule missing: run `git submodule update --init`")

setup(ext_modules=[Extension(
    "photonics_core",
    sources=["bindings/pybindings.cpp"],
    include_dirs=[str(pybind_include)],
    language="c++",
    extra_compile_args=["-std=c++17", "-O3", "-fvisibility=hidden"],
)])
