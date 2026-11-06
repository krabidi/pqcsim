# pqcsim physics reference

Every model below states its equations, its approximations, and the tests that validate it
(`tests/`). Quote the approximations together with any result.

## Global conventions

| Quantity | Convention |
|---|---|
| Time dependence | e^{-iωt} |
| Complex index | ñ = n + ik, k ≥ 0 absorbing; ε = ñ² |
| Transfer matrices | U[out][in]; fields E_out = T E_in; passive ⇔ largest singular value ≤ 1 |
| Beamsplitter | [[cos θ, i e^{-iφ} sin θ], [i e^{iφ} sin θ, cos θ]]; θ = π/4 is 50:50 |
| PCM state | crystalline fraction x ∈ [0, 1] (0 amorphous, 1 crystalline) |
| Gaussian states | ħ = 1, x = (a+a†)/√2, p = (a−a†)/(i√2), ordering (x₁..xₙ, p₁..pₙ), vacuum covariance I/2 |
| Units | SI internally; lengths in nm/µm where the argument name says so |

## Materials (`pqcsim.materials`)
- Optical constants per phase: constant, tabulated (linear interpolation, no extrapolation) or per channel.
- Mixing of the two phases: Lorentz–Lorenz (default), linear in ε, or linear in ñ.
- Kinetics: first-order JMAK, k(T) = k₀ e^{−Eₐ/k_BT} for T_cryst_min ≤ T < T_melt.
  - Optional Wilson–Frenkel driving force [1 − e^{−ΔG/k_BT}] with ΔG = ΔH_f(T_m−T)/T_m.
- *Tests:* mixing endpoints and passivity, Lorentz–Lorenz definition, tabulated range.

## Thin-film optics (`pqcsim.optics.tmm`)
- Normal-incidence transfer-matrix method.
- Layer absorption uses the exact Poynting flux, S = n′(|a|²−|b|²) + 2k Im(b a*).
- Depth profile: closed-form integral of k₀ Im(ε)|E|² over sub-slabs.
- *Tests:* Fresnel, quarter-wave AR, Bragg mirror, half-wave layer, Beer–Lambert, reciprocity,
  R+T+ΣA = 1, field integral, depth profile vs numerical integration.

## PCM cell (`pqcsim.devices.pcm`)
- **Thermal:** lumped RC per area (C = ρc_p d), or any temperature history T(t) from the heat solvers.
- **SET:** x′ = 1 − (1−x) exp(−∫k(T(t))dt), integrated over heating and cooling.
- **RESET:** phenomenological erase at the temperature peak, x(1−u^γ). Crystallization is applied before the erase, and recrystallization during the quench after it.
- **Approximations:**
  - absorption evaluated at the pre-pulse state
  - no latent heat
  - constant ρ, c_p, k
  - phenomenological RESET
- *Tests:*
  - adiabatic and steady-state limits
  - kinetics against an independent ODE (SET) and a temperature-domain integral (quench)
  - temperature-history route against the analytic RC result
  - legacy formulas

## Periodic pixel optics: RCWA (`pqcsim.optics.rcwa`)
- **Method:** 2D-periodic rigorous coupled-wave analysis at normal incidence, e^{-iωt}, normalized 𝐇 = iZ₀H.
  - Laurent-rule convolution matrices.
  - Redheffer star products of S-matrices referenced to a zero-thickness gap.
  - Per-layer absorption from interior Poynting fluxes.
- **Used for** pixelated arrays with `PCMArray(..., optics="rcwa")`.
  - Pixel-layer absorption lies entirely inside the pixels (lossless fill).
  - Uniform layers absorb uniformly over the cell.
- **Approximations:**
  - plane-wave illumination of an infinite array (the spot should be several periods wide)
  - the Laurent rule converges slowly for absorbing pixels (~1% between 121 and 225 orders), so check convergence
- **Why it matters:** for amorphous GeTe pixels at 1550 nm, RCWA pixel absorption is 1.3–2.0× the local-TMM value (pitch 0.4–2 µm). Light is funnelled into the high-index pixels, which a column-by-column 1D model cannot capture.
- *Tests:*
  - uniform layers reproduce TMM to 1e-12, independent of the gap permittivity
  - lossless energy conservation to 1e-12
  - diffraction only above the cutoff
  - polarization independence of square pixels
  - fill-factor limits
  - split-row consistency
  - grcwa benchmark within 1e-3

## Heat conduction (`pqcsim.thermal`)
- **Axisymmetric (r, z) finite volumes** (continuous films)
  - adaptive BDF time integration
  - exact Gaussian-spot source per annulus
- **3D Cartesian voxels** (patterned arrays)
  - harmonic-mean face conductances
  - BDF2 with restarts at source switches (energy-exact) and variable-step BDF2
- **Heat responses**
  - superposition over z rows (and pixel/fill column types), so any optical depth profile is applied exactly
  - local 1D TMM per column type
- *Tests:*
  - energy conservation
  - 2D Gaussian spreading (analytic)
  - steady parabola and series resistance
  - RC limit and linearity
  - voxel vs axisymmetric cross-check (≤ 2% of peak)
  - array symmetry
  - superposition vs direct runs

## Phase change and nonlinear heat (`pqcsim.thermal.nonlinear`, `pqcsim.devices.resolved`)
- **Enthalpy method:** H(T) = ρc_p(T−T₀) + ρL f(T), with a logistic liquid fraction of width w.
  - Integrating H keeps energy conservation exact through melting.
  - T(H) is recovered by safeguarded Newton (closed form where L = 0).
- **Properties:** k = k_s + f(k_l − k_s), optionally × (1 + β(T−T₀)).
- **Time stepping:** variable-step BDF2 with a *full* Newton iteration per step (analytic sparse Jacobian, damped), BDF1 restarts at source switches, and step control on ΔT per step.
  - scipy's Radau/BDF use a frozen Jacobian and stall when a cell crosses the melting interval.
- **Resolved programming:**
  - per-column TMM with the active layer split into rows, each with its own x
  - enthalpy heat flow
  - **melt-quench RESET** per cell: T_peak ≥ T_melt → amorphous, then JMAK recrystallization during the quench; no phenomenological erase
- **Approximations:**
  - optics evaluated at the pre-pulse x field (no change within the pulse)
  - local TMM per column
  - melting smoothed over w (results converge as w → 0)
  - no change of n or k with temperature
- *Tests:*
  - linear limit: 1e-6 of peak (Radau) and 1% (Newton-BDF2)
  - energy conservation through melt/refreeze
  - one-phase Stefan front: 1.8% → 1.3% → 0.9% with refinement
  - Kirchhoff k(T) steady state
  - analytic Jacobian vs finite differences (≤ 3e-6)
  - absorbed-energy bookkeeping
  - nonlinear vs linear-superposition route (0.12% of rise)
  - melt-quench behaviour

## Arrays (`pqcsim.arrays.crossbar`)
- **Write disturb**, from one of:
  - a Gaussian optical tail plus a conduction kernel (simple model)
  - full heat-diffusion histories
- **Readout:** incoherent optical MVM, y = W P.
- *Tests:* single-cell equivalence, symmetry, decay with distance, solver consistency, linearity.

## Linear optics (`pqcsim.circuits`)
- **Network:** element list → transfer matrix; coherent fields or incoherent powers.
- **Decomposition** of passive matrices into couplers, phases and losses:
  - 2×2 unitaries
  - Clements N×N unitaries (N(N−1)/2 nearest-neighbor couplers)
  - SVD for passive matrices
- **Components:**
  - dispersive directional coupler (coupled-mode theory)
  - all-pass and add-drop rings with group index
  - PCM waveguide segment (first-order δn_eff = Γ(ñ_PCM − n_ref))
  - Touchstone S-parameters
- *Tests:*
  - exact reconstruction
  - MZI fringes and extinction range
  - critical coupling and exact FSR = λ_mλ_{m+1}/(n_g L)
  - add-drop unitarity and reciprocity
  - Touchstone RI/MA/DB round trips

## Quantum (`pqcsim.quantum`)
- **Fock engine (C++):** exact two-mode unitaries on photon-number superpositions.
- **Loss:** a beamsplitter to a fresh vacuum ancilla per lossy element, then traced out (exact).
- **Partial distinguishability:** Gram-matrix permanent formula, with Halmos dilation for loss.
  - Gaussian wavepacket overlaps: S = e^{−Δω²/8σ²} e^{−σ²τ²/2}.
- **Gaussian states:**
  - squeezers and two-mode squeezers (SPDC/SFWM, single-mode approximation)
  - lossy networks: σ → XσXᵀ + (I−XXᵀ)/2
  - hafnian photon-number probabilities (zero mean)
  - vacuum and click probabilities, including displacement
- **Detectors:**
  - binomial efficiency
  - Poisson dark counts
  - threshold vs PNR
  - coincidence window and accidentals
- *Tests:*
  - HOM with loss and distinguishability
  - binomial loss and ⟨n⟩ = Σ|T|²n
  - Halmos-dilation permanent reference
  - internal-mode cross-simulation
  - thermal, squeezed and TMSV statistics, heralding
  - two-route click agreement, Monte Carlo timing

## Electronics and analysis
- **Readout:** shot, dark and thermal noise of a photodetector with TIA; ADC quantization.
- **Control:**
  - thermo-optic heaters: φ = (2π/λ)(dn/dT) L R_th V²/R
  - DAC quantization and noise
  - heater-crosstalk compensation, D K P = φ (mod 2π) with P ≥ 0
  - feed-forward delay-line loss
- **Variation and calibration:**
  - Monte Carlo yield with binomial error bars
  - weighted least-squares fits with covariance
  - crosstalk-matrix extraction
- *Tests:*
  - Monte Carlo noise statistics
  - analytic MZI leakage mean under coupler errors
  - fit coverage and χ²
  - exact crosstalk recovery

## Benchmarks against external codes (`tests/test_benchmarks_external.py`)

Run with `pip install ".[bench]"`; skipped automatically when a package is missing.

| pqcsim | Reference code | Agreement |
|---|---|---|
| `optics.tmm` (R, T, r, per-layer absorption) | `tmm` (S. Byrnes) | ≤ 1e-12 on random absorbing stacks |
| permanent, hafnian (Python and C++) | The Walrus (Xanadu) | ≤ 1e-10 relative |
| Gaussian photon-number probabilities (lossy, multimode) | The Walrus `probabilities` | ≤ 1e-12 |
| threshold-click probabilities (with displacement) | The Walrus `threshold_detection_prob` | ≤ 1e-10 |
| Fock multiphoton probabilities | Perceval SLOS (Quandela) | ≤ 1e-12 |
| axisymmetric transient heat conduction | FiPy (NIST), Richardson-extrapolated | < 1 % of peak at all times; peak within 0.5 % |
| RCWA of a PCM pixel array (R, T, A) | grcwa | ≤ 1e-3 (different order truncation) |

The heat benchmark: FiPy's implicit Euler converges to pqcsim as its step shrinks (first-order
error, 0.32 K -> 0.08 K for a 4x smaller step at 10 ns), so the comparison uses 2 T(dt/2) - T(dt).

## Known limits
- Exact multiphoton methods scale exponentially, as permanents or hafnians.
  - Partial distinguishability is meant for ≲ 6 photons.
  - The Fock engine is limited by state-space size.
- Ring and coupler models are monochromatic: no multimode spectral or temporal dynamics.
- Squeezing uses a single temporal mode, so there's no spectral purity model yet.
- PCM waveguide segments are first-order perturbative; use a mode solver for strong confinement.
- The phase-change heat solver is pure Python (one sparse solve per Newton iteration). After warm-started
  enthalpy inversion and one inversion per Newton iteration: ~7 s for 72 cells through melt and refreeze
  (was 37 s). Fine for device studies; large sweeps would need a compiled kernel.
