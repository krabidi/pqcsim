"""YAML/JSON circuit description: one file for the circuit, the light source and the detectors.

    modes: 2
    wavelength_nm: 1550
    materials: data/materials.json          # optional, for pcm_segment
    elements:
      - {type: waveguide, mode: 0, length_um: 50, n_eff: 2.4, loss_dB_per_cm: 2}
      - {type: coupler, modes: [0, 1], split: 0.5}           # or length_um + dn_super
      - {type: phase, mode: 0, phi: 1.2}
      - {type: pcm_segment, mode: 1, material: GeTe, x: 0.3, length_um: 2, confinement: 0.05, n_ref: 1.45}
    source:  {type: fock, occupation: [1, 1]}
    detectors: [{efficiency: 0.9, dark_mean: 1e-4, threshold: true}, {efficiency: 0.9}]

Element types: beamsplitter (modes, theta, phi), coupler (modes, split | length_um + dn_super
[+ ddn_dlam_per_nm, lam0_nm]), phase (mode, phi), loss (mode, eta | loss_dB), waveguide (mode, length_um,
n_eff, loss_dB_per_cm), ring_allpass (mode, length_um, n_eff, n_g, theta, loss_dB_per_cm),
ring_add_drop (modes [in/through, add/drop], length_um, n_eff, n_g, theta1, theta2, loss_dB_per_cm),
matrix (modes, real, imag), touchstone (file, modes, in_ports, out_ports, index), pcm_segment.
Source types: fock (occupation), distinguishable (input_modes, gram or delays_s + sigma_omega),
gaussian (two_mode_squeezers: [{modes, r | mean_pairs}], squeezers: [{mode, r}]).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..circuits import components as comp
from ..circuits.network import BeamSplitter, LinearNetwork, Loss, PhaseShift, pcm_segment, waveguide
from ..materials import load_materials
from ..quantum.detectors import Detector, detect, gaussian_clicks


def _numbers(obj):
    """YAML 1.1 reads e.g. 2.0e12 (no exponent sign) as a string; convert numeric strings to floats."""
    if isinstance(obj, dict):
        return {k: _numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numbers(v) for v in obj]
    if isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return obj
    return obj


def _read(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith(".json"):
        return _numbers(json.loads(text))
    import yaml
    return _numbers(yaml.safe_load(text))


def _need(spec, *keys):
    missing = [k for k in keys if k not in spec]
    if missing:
        raise ValueError(f"element {spec.get('type')!r} is missing {missing}")


@dataclass
class CircuitFile:
    network: LinearNetwork
    wavelength_nm: float
    source: Optional[Dict[str, Any]] = None
    detectors: List[Detector] = field(default_factory=list)

    def run(self) -> Dict[tuple, float]:
        """Detection probabilities for the file's source (and detectors, if given)."""
        if self.source is None:
            raise ValueError("the circuit file has no source")
        kind = self.source.get("type")
        n = self.network.n_modes
        if kind == "gaussian":
            from ..quantum.gaussian import GaussianState, squeezing_for_mean_pairs
            st = GaussianState(n)
            for sq in self.source.get("two_mode_squeezers", []):
                r = sq["r"] if "r" in sq else squeezing_for_mean_pairs(sq["mean_pairs"])
                st.two_mode_squeeze(sq["modes"][0], sq["modes"][1], r, sq.get("phi", 0.0))
            for sq in self.source.get("squeezers", []):
                st.squeeze(sq["mode"], sq["r"], sq.get("phi", 0.0))
            st.apply_network(self.network)
            dets = self.detectors or [Detector(threshold=True)] * n
            if not all(d.threshold for d in dets):
                cutoff = int(self.source.get("photon_cutoff", 4))
                dist = {p: st.fock_probability(list(p)) for p in np.ndindex(*([cutoff + 1] * n))
                        if sum(p) <= cutoff}
                return detect(dist, dets) if self.detectors else dist
            import itertools
            return {c: gaussian_clicks(st, c, [d.efficiency for d in dets], [d.dark_mean for d in dets])
                    for c in itertools.product([0, 1], repeat=n)}
        if kind == "fock":
            from ..quantum.lossy import simulate_fock
            dist = simulate_fock(self.network, self.source["occupation"])
        elif kind == "distinguishable":
            from ..quantum.distinguishable import gaussian_overlap, simulate_partial_distinguishability
            modes = self.source["input_modes"]
            if "gram" in self.source:
                S = np.asarray(self.source["gram"], complex)
            else:
                d = [float(v) for v in self.source["delays_s"]]
                sig = float(self.source["sigma_omega"])
                S = np.array([[gaussian_overlap(d[j] - d[i], sig) for j in range(len(d))] for i in range(len(d))])
            dist = simulate_partial_distinguishability(self.network, modes, S)
        else:
            raise ValueError(f"unknown source type {kind!r}")
        return detect(dist, self.detectors) if self.detectors else dist


def load_circuit(path: str) -> CircuitFile:
    spec = _read(path)
    base = os.path.dirname(os.path.abspath(path))
    n = int(spec["modes"])
    for e in spec.get("elements", []):
        for key in ("mode",):
            if key in e:
                e[key] = int(e[key])
        if "modes" in e:
            e["modes"] = [int(m) for m in e["modes"]]
    if spec.get("source") and "input_modes" in spec["source"]:
        spec["source"]["input_modes"] = [int(m) for m in spec["source"]["input_modes"]]
    if spec.get("source") and "occupation" in spec["source"]:
        spec["source"]["occupation"] = [int(m) for m in spec["source"]["occupation"]]
    lam = float(spec.get("wavelength_nm", 1550.0))
    mats = {}
    if "materials" in spec:
        mpath = spec["materials"] if os.path.isabs(spec["materials"]) else os.path.join(base, spec["materials"])
        mats = {m.name: m for m in load_materials(mpath)}
    net = LinearNetwork(n)
    for e in spec.get("elements", []):
        t = e.get("type")
        if t == "beamsplitter":
            _need(e, "modes")
            net.add(BeamSplitter(e["modes"][0], e["modes"][1], e.get("theta", math.pi / 4), e.get("phi", 0.0)))
        elif t == "coupler":
            _need(e, "modes")
            i, j = e["modes"]
            if "split" in e:
                net.add(BeamSplitter(i, j, math.asin(math.sqrt(e["split"]))))
            else:
                _need(e, "length_um", "dn_super")
                net.add(comp.directional_coupler(i, j, e["length_um"], lam, e["dn_super"],
                                                 e.get("ddn_dlam_per_nm", 0.0), e.get("lam0_nm", 1550.0)))
        elif t == "phase":
            _need(e, "mode", "phi")
            net.add(PhaseShift(e["mode"], e["phi"]))
        elif t == "loss":
            _need(e, "mode")
            eta = e["eta"] if "eta" in e else 10 ** (-e["loss_dB"] / 10)
            net.add(Loss(e["mode"], eta))
        elif t == "waveguide":
            _need(e, "mode", "length_um", "n_eff")
            net.extend(waveguide(e["mode"], e["length_um"], e["n_eff"], lam, e.get("loss_dB_per_cm", 0.0)))
        elif t == "ring_allpass":
            _need(e, "mode", "length_um", "n_eff", "n_g", "theta")
            tr = comp.ring_allpass(lam, e["length_um"], e["n_eff"], e["n_g"], e["theta"],
                                   e.get("loss_dB_per_cm", 0.0), e.get("lam0_nm", 1550.0))
            net.extend(comp.matrix_elements(np.array([[tr]]), [e["mode"]]))
        elif t == "ring_add_drop":
            _need(e, "modes", "length_um", "n_eff", "n_g", "theta1", "theta2")
            T = comp.ring_add_drop(lam, e["length_um"], e["n_eff"], e["n_g"], e["theta1"], e["theta2"],
                                   e.get("loss_dB_per_cm", 0.0), e.get("lam0_nm", 1550.0))
            net.extend(comp.matrix_elements(T, e["modes"]))
        elif t == "matrix":
            _need(e, "modes", "real")
            T = np.asarray(e["real"], float) + 1j * np.asarray(e.get("imag", np.zeros_like(e["real"])), float)
            net.extend(comp.matrix_elements(T, e["modes"]))
        elif t == "touchstone":
            _need(e, "file", "modes", "in_ports", "out_ports")
            f = e["file"] if os.path.isabs(e["file"]) else os.path.join(base, e["file"])
            _, S = comp.load_touchstone(f)
            T = comp.transfer_from_sparams(S[int(e.get("index", 0))], e["in_ports"], e["out_ports"])
            net.extend(comp.matrix_elements(T, e["modes"]))
        elif t == "pcm_segment":
            _need(e, "mode", "material", "x", "length_um", "confinement", "n_ref")
            if e["material"] not in mats:
                raise ValueError(f"material {e['material']!r} not found (add a 'materials' file)")
            net.extend(pcm_segment(e["mode"], mats[e["material"]], e["x"], e["length_um"], lam,
                                   e["confinement"], complex(e["n_ref"])))
        else:
            raise ValueError(f"unknown element type {t!r}")
    dets = [Detector(float(d.get("efficiency", 1.0)), float(d.get("dark_mean", 0.0)), bool(d.get("threshold", False)))
            for d in spec.get("detectors", [])]
    if dets and len(dets) != n:
        raise ValueError("give one detector per mode (or none)")
    return CircuitFile(net, lam, spec.get("source"), dets)
