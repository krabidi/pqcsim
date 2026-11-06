cmake_minimum_required(VERSION 3.14)
project(PhotonicQC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_FINDPYTHON ON)

# Tell CMake to use pybind11 from subdirectory
add_subdirectory(extern/pybind11)

# Compile the Python module (produces photonics_core.<python-ext>.so in build/)
pybind11_add_module(photonics_core bindings/pybindings.cpp)
