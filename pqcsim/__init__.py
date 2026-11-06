"""pqcsim: device-to-circuit simulation of photonic and optoelectronic chips.

Layers (each validated against analytic results or an independent calculation in tests/):
  materials           PCM optical constants, effective-medium mixing, thermal and kinetic data
  optics.tmm          thin-film optics with exact per-layer absorption
  devices.pcm         optically programmed PCM memristor cell (optics + thermal + kinetics)
  arrays.crossbar     PCM arrays: programming with optical/thermal crosstalk, optical MVM
  circuits.network    classical lossy linear waveguide networks (incl. PCM segments)
  quantum.lossy       exact Fock-state simulation of the same networks, losses included
  electronics.readout photodetector/TIA noise and ADC quantization
"""
__version__ = "0.1.0"
