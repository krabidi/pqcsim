# Validation against experiments

Verification (equations solved correctly) is covered by `tests/` and the external benchmarks in
`docs/physics.md`. This file records comparisons with **measurements and published statements**,
including the ones that did not validate the model. A comparison counts as a validation only when every
input comes from an independent source, not from fitting to the number being compared.

Measured optical constants used here are in `data/nk/` (refractiveindex.info, CC0), each with its reference.

## 1. Crystal growth of melt-quenched AgInSbTe: Salinga et al., Nat. Commun. 4, 2371 (2013), open access

**Model:** `pqcsim.devices.growth`. Inputs are exactly as stated in the paper:
- eq. 1 constants
- ΔH_m = 173 meV/atom, T_m = 808 K
- MYEGA fit m = 135, T_g = 445 K, η∞ = 1.22×10⁻³ Pa·s
- glass growth: 2.7 eV, with 3.4 m/s at 553 K

Checked in `tests/test_growth.py`.

| Statement in the paper | Ours | |
|---|---|---|
| 0.9 < ΔG/k_BT < 1.6 over 418–553 K | 0.93–1.58 | ✓ |
| η(T_g) = 10¹² Pa·s, fragility 135 | 10¹² Pa·s, 135.00 | ✓ (definition) |
| Measured growth: ~100 nm/s at 418 K | 38 nm/s from the 2.7 eV Arrhenius | ✓ within the paper's "around" (2.6×) |
| Arrhenius extrapolation reaches 1000 m/s "at around 625 K" | 615 K | ✓ |
| Maximum growth velocity 10–100 m/s (two-pulse measurement) | 97 m/s at 716 K | ✓ |
| u(553 K) = 3.4 m/s (measured) | 1.56 m/s on the fitted liquid branch | ~2× (the paper's MYEGA curve is a fit to 11 points) |

**Verdict: reproduced.** The model and the published parameters give the measured maximum velocity and
the stated extrapolations. A first implementation used η = min(η_glass, η_liquid). That gave
u_max = 1.9×10⁸ m/s, which is unphysical, because a fragile liquid flattens at high T. It was replaced by
the physical glass-to-liquid junction at the curves' closest approach (T_x = 532 K).

## 2. GST on silicon waveguides: Zheng et al., Opt. Mater. Express 8, 1551 (2018), open access

**Script:** `validation/zheng2018_gst_on_si.py`. Inputs:
- the paper's own ellipsometry: aGST 3.89+0.02i, cGST 6.63+1.09i (fcc, RTA 200 °C)
- geometry: 500 × 220 nm Si ridge, 190 nm etch, 20 nm GST, 11 nm ITO (k < 0.002)
- Si: Li 1980; SiO₂: Malitson 1965
- ITO's real index isn't stated, so it is scanned over 1.7–2.0; the coverage (ridge top only, or also the slab) is varied

| Quantity | Paper measured | Paper FEM | Ours |
|---|---|---|---|
| α, crystalline GST | **7.6 ± 1.0 dB/µm** | 7.7 dB/µm | **7.37–7.61 dB/µm** (grid-converged to 0.7%) |
| Im(n_eff), crystalline | | 0.22 | 0.209–0.216 |
| α, amorphous GST | 0.27 ± 0.04 dB/µm | 0.08 dB/µm | 0.043–0.048 dB/µm |

**Verdict: crystalline loss reproduced within the measurement error, from independent inputs.**
- The amorphous measurement is several times above both simulations; the paper itself attributes this to effects outside the model.
- Our Re(n_eff) is 0.07–0.15 below the paper's FEM. The quasi-TE solver converges to the reference 2.44 for a bare 500 × 220 strip, so the difference most likely comes from film coverage (sidewalls), which isn't specified. The measured quantity, loss, is unaffected.

## 3. Crystal growth of supercooled-liquid GST: Orava et al., Nat. Mater. 11, 279 (2012) and SI

**Model:** `devices.growth.orava2012_gst()`. Inputs as published:
- Cohen–Grest fit B = 121 K, C = 1.71 K, T₀ = 427 K (SI p. 7)
- T_m = 900 K, ΔH_m = 12.13 kJ/mol, Thompson–Spaepen driving force
- absolute scale U_kin(T_m) = 13.5 m/s (SI eqs. S7–S8)

| Statement | Ours | |
|---|---|---|
| Maximum of U at 0.76 T_m | 0.757 T_m (681 K) | ✓ |
| U_max (Fig. 5) just above 3 m/s | 2.98 m/s | ✓ |
| U_kin(T_m) = 13.5 m/s from Stokes–Einstein (a = 0.3 nm, η = 1.1–1.2 mPa·s) | 12.2–13.3 m/s | ✓ within 1.5 % |

**Verdict: reproduced.**

**Cross-check, Jeyasingh et al. 2014 (SI):**
- Their device has a ≈ 70 nm amorphous region that recrystallizes at 500 K in about 10 µs, i.e. ≈ 3.5 mm/s.
- The growth-dominated AIST model gives 4.2 µs, within a factor of 2.4. The GST supercooled-liquid model is about 150× faster.
- Order-of-magnitude consistency only: the SI does not name the material.

## 4. GST cell on a Si₃N₄ waveguide: Ríos et al., Nature Photonics 9, 725 (2015)

**Script:** `validation/rios2015_gst_waveguide.py --measured`. Inputs, all measured:
- GST: Frantz et al. 2023 (ellipsometry)
- Si₃N₄: Luke 2015
- SiO₂: Malitson 1965
- ITO: Minenkov 2024, two depositions, plus a transparent bound
- geometry: from the paper (confirmed by Ríos et al., Sci. Adv. 2019, arXiv:1801.06228)

| Quantity | Measured | Predicted |
|---|---|---|
| Fully crystalline cell loss at 1560 nm (5 µm) | 7.14 dB | 13.8–14.9 dB (over the ITO datasets) |
| Readout change for full amorphization at 1570 nm | 21 % (single pulse, partial) | 95–96 % |

**Numerics are robust:**
- The 2D rib loss (14.96 dB) sits just below the exact upper bound of a fully covered slab (15.36 dB), computed with an independent solver.
- Grid refinement changes the result by about 4%.
- With measured constants, a GST film on Si gives reflectivities within 0.01–0.03 of the values simulated in arXiv:1911.03536.

**Verdict: not validated. The model predicts about 2× the loss.**

Evidence obtained since (the paper's SI and the downloaded articles):
- **Phase is not the cause.** The device GST is cubic (fcc), "lattice spacings 3.03 Å and 2.15 Å", and the Frantz reference films are also fcc.
- **The model is sound with the right film data.** The same solver reproduces Zheng 2018 (section 2) when the films' own ellipsometry is used.
- **The device's films absorb less.** The SI reports the measured absorption of the device's crystalline GST films as α = 1.1 dB/µm, and the measured cells give 0.97 dB (1 µm) and 7.14 dB (5 µm). Thicker films with ellipsometry-measured fcc constants predict about 3 dB/µm.
- **Most likely explanation:** the 10 nm films absorb less than thicker films (thickness-dependent optical constants of ultrathin PCM films, as Cheng et al. 2021 report for Sb), possibly together with a partly crystalline conditioned Level 0.
- **What it would take:** crystalline k ≈ 0.4 instead of 0.9 would match. That's an inference, not a measurement; ellipsometry of 10 nm GST/ITO films would settle it.

## 4b. End-to-end thermal response of the Ríos 2015 cell (SI Fig. S3b): reproduced

**Script:** `validation/rios2015_thermal.py`. The full chain, nothing fitted:
- **optics:** absorption decays along the 5 µm cell at its measured 7.14 dB; across the guide it follows the computed mode's |E|²
- **heat:** 3D voxel heat flow in the full rib/film/oxide stack, with the paper's own thermal properties (SI Table S1)
- **pulse:** 90 ns, 540 pJ, of which 80.7 % is absorbed (the paper's figure)

| t (ns) | 0 | 25 | 50 | 90 | 100 | 150 | 200 | 250 |
|---|---|---|---|---|---|---|---|---|
| measured ΔT (K), read from Fig. S3b, ±2 K | 0 | 21 | 31 | 38 | 29 | 19 | 13 | 10 |
| model ΔT (K) | 0 | 21.2 | 30.6 | 41.2 | 30.2 | 18.7 | 13.9 | 10.9 |

**Verdict: reproduced.** Heating, peak (+8 %) and cooling all agree within the reading accuracy.
- Assumption: the SI does not state that device's cell length; it is taken as 5 µm, the paper's main device.

**Switching energies (SI Table S2) cannot serve as validation.**
- The measured transient (38 K for 540 pJ over 90 ns) implies about 8600 pJ to reach melting at that pulse length.
- Table S2 reports about 300 pJ at 100 ns. Both are the paper's own numbers, so this is an internal inconsistency of about 25×, independent of any model.
- Table S2 is also below the adiabatic minimum to melt the GST+ITO volume using the paper's Table S1 properties (83 pJ for the 1 µm cell versus 13.4 pJ reported).
- Our model (`validation/rios2015_switching.py`) predicts 254–5711 pJ, consistent with the thermal measurement, not with Table S2.

## 4c. Free-space laser crystallisation of as-deposited GST films: Sun et al., Sci. Rep. 6, 28246 (2016)

**Script:** `validation/sun2016_laser_crystallization.py`.
- **Experiment:** ~90 nm GST on 500 nm SiO2/Si, single 248 nm, 20 ns top-hat pulse (1D heat flow).
- **Optics:** the paper's absorption coefficient (k = 1.58); n bracketed over 1.2–2.2.
- **Kinetics:** growth-only, as-deposited GST (Orava 2012).

| Threshold (mJ/cm²) | measured | model (growth only) |
|---|---|---|
| onset (reflectivity change) | 15 | 2.2–2.4 (1 nm growth) |
| XRD-detectable | 36 | — |
| full crystallisation | 67–130 | 3.5–3.8 (25 nm growth) |
| melting onset | not reported | 7.0–7.4 |

**Verdict: not reproducible with standard GST properties; the dataset conflicts with basic heat diffusion.**

With nucleation added (classical nucleation theory, `pqcsim.devices.nucleation`, parameters of Orava & Greer, Acta Mater. 139, 226 (2017)):
- The predicted onset (1 % crystallised at the film top) rises to 4.0 mJ/cm², still below the measured 15.
- Above 7.1 mJ/cm² the top of the film melts and re-amorphises on cooling. The paper reports a fully crystalline film up to 130 mJ/cm².

**Model-independent check.** For a constant absorbed flux q over a pulse of length t, a semi-infinite solid's surface heats by ΔT = 2q√(t/πkρc).
- With about 70 % absorbed, k = 0.2 W/m/K and ρc = 1.35×10⁶ J/m³/K, 15 mJ/cm² over 20 ns gives ΔT ≈ 1600 K, far above melting.
- The full model agrees: 8.7 versus 10.8 K per J/m², the difference being the finite absorption depth.
- So the reported fluences cannot be the energy absorbed in a GST film with standard thermal properties.
- The probable cause is fluence calibration (energy measured upstream of the sample) or optics. This cannot be resolved from the paper.

**What was validated instead:**
- **The nucleation module against Orava & Greer's own results:** I_st peaks at T/T_m = 0.594 (paper ≈ 0.6); rates agree within 1–2 decades over 0.43–0.8 T_m; time lags span 10³ s near T_g to ~10⁻¹⁰ s.
- **Viscosity:** η(T_g) = 1.1×10¹² Pa·s.
- **JMAK:** reproduces the isothermal closed form including the transient delay.

## 4d. Femtosecond-laser switching of ultrathin GST on gold: Schlich et al., ACS Photon. 2, 178 (2015)

**Script:** `validation/schlich2015_fs_switching.py` (`--ratio` for the calibration-free test).

**Experiment:**
- Stack, from the TEM in Fig. 1a: 40 nm SiO2 / 18 nm GST / 5 nm Ti+TiN / 158 nm Au.
- 60 fs pulses at 800 nm, 270 µm spot (1D heat flow).
- Amorphisation: one pulse at 14.3 mJ/cm². Recrystallisation: 640 pulses at 4.0 mJ/cm².

**Model:**
- The pulse is absorbed before the lattice heats, so the optics are exact (TMM with measured n,k for GST, Ti and TiN).
- 1D enthalpy heat flow with latent heat. Melt-quenched growth kinetics (Sebastian 2014).

**Optics check:** modelled reflectance 0.09 (amorphous) against about 0.10 measured (Fig. 1c), and 0.26–0.36 (60–100 % crystalline) against about 0.27 measured.

| | measured | model |
|---|---|---|
| F_cryst / F_amorph (needs no fluence calibration) | 0.28 | window 0.17–0.42 (all parameter variants) |
| full-melt fluence (absolute) | 14.3 mJ/cm² quoted | 5.2–5.6 mJ/cm² |

**Verdict:**
- **Relative thresholds: reproduced.**
- **Absolute fluences: the quoted values are 2.6× (if peak) to 5× (if spot average) above what energy conservation allows.**
  - At the quoted 4.0 mJ/cm², the absorbed energy alone (72 % in 18 nm) would heat the amorphous film by about 1200 K before heat can move, so it would melt rather than crystallise.
  - The gap therefore lies in the quoted fluence (its definition or losses before the sample), or in energy leaving the GST within a picosecond (hot carriers into the metal). It does not lie in the heat-flow model.

## 4e. Femtosecond-laser pixel writing in 70 nm GST: Wang et al., Nat. Photon. 10, 60 (2016)

**Script:** `validation/wang2016_fs_writing.py`.

**Experiment:**
- Stack: 70 nm ZnS-SiO2 / 70 nm GST / 70 nm ZnS-SiO2 on glass.
- 85 fs pulses at 730 nm, NA 0.8.
- Crystallisation: up to 80 pulses at 140 mJ/cm². Erasure: one pulse at 445 mJ/cm².

**Absolute check:** taken as at-sample values, 140 mJ/cm² would heat the film adiabatically by about 11,700 K. The quoted fluences cannot be the delivered ones; they are most likely measured before the objective and SLM losses.

**Ratio test** (measured 0.31):

| Erase criterion | Predicted window | Measured ratio inside? |
|---|---|---|
| Whole 70 nm film molten | 0.09–0.21 | no |
| Top half molten | 0.22–0.49 | yes |
| Top 30 % molten | 0.33–0.71 | at the edge |

The light penetrates only 18 nm into crystalline GST, so the erase criterion is ambiguous. **Verdict: consistent, but not decisive.**

## 4f. What the free-space laser datasets mean for absolute fluences

Sun 2016 (6–20×), Schlich 2015 (2.6–5×) and Wang 2016 (about 20×) all quote fluences above what energy conservation allows for the absorbed light. In two of the three cases this is shown without any heat-flow model, from the adiabatic bound alone.
- Quoted laser fluences in this literature therefore cannot validate absolute thresholds.
- The model's absolute fluences are delivered at-sample values; their uncertainty is set by the input data (laser spot size, optical and thermal constants), about ±40 %.
- Ratio metrics are calibration-free, and are reproduced where testable (Schlich 2015).
- A single at-sample fluence calibration (e.g. measured power after the objective plus a knife-edge spot size) would close the gap.

## 5. Not reproducible from accessible sources

- **Cheng et al., Sci. Adv. 7, eabd7097 (2021):** the thin-Sb n, k spectra are only plotted, not tabulated, and the database has no thin-film Sb.
- **Jeyasingh et al., Nano Lett. 14, 3419 (2014):** growth velocities only in figures. The SI gives device dimensions used for the consistency check in section 3.

## Status

| Comparison | Result |
|---|---|
| Crystal growth: AIST (Salinga 2013), GST (Orava 2012) | quantitative reproduction of published values |
| GST-on-Si waveguide loss (Zheng 2018) | quantitative reproduction of a measurement, within its error |
| GST-on-SiN memory cell (Ríos 2015), optical loss | ~2× gap, traced to the device's 10 nm films absorbing less than thicker reference films |
| Ríos 2015 cell, end-to-end thermal transient | reproduced within ±2 K (peak +8 %) |
| Ríos 2015 switching energies | not usable: inconsistent with the paper's own thermal data (~25×) and below the adiabatic bound |
| Free-space laser crystallisation of as-deposited GST (Sun 2016) | not usable: reported fluences conflict with the analytic surface-heating bound for standard GST properties |
| Femtosecond switching window, 18 nm GST on Au (Schlich 2015) | relative thresholds reproduced (0.28 inside 0.17–0.42); quoted absolute fluences 2.6–5× above the energy bound |
| Femtosecond pixel writing, 70 nm GST (Wang 2016) | ratio consistent for a half-depth erase criterion; quoted fluences about 20× above the adiabatic bound |
| GST nucleation (Orava & Greer 2017) | CNT reproduced: peak position, rates within 1–2 decades, time lags |
| Thin-film optics | consistent with independent simulations; benchmarked against `tmm` |
