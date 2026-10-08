# BOREAS

**BOREAS** is a Python package for modeling **hydrodynamic mass loss** from exoplanet atmospheres, including **energy-limited (EL)** and **recombination-limited (RL)** regimes with **multi-species fractionation** among hydrogen (H), helium (He), oxygen (O), carbon (C), nitrogen (N), and sulfur (S).

The code couples a **molecular bolometric (IR) region** to a **fully dissociated atomic outflow**, tracking composition-dependent escape and diffusive separation self-consistently.

> **Package name:** boreas </br>
> **Import name:** boreas </br>
> **Requires:** python ≥ 3.11, numpy ≥ 1.26, scipy ≥ 1.12 </br>
> **Authors:** M. Valatsou, J. Owen, C. Dorn (2025)

---
## License and Usage Notice

This repository is made public for transparency and collaboration but is **not open source**.
Use of this code for research, publication, or derivative work requires explicit written permission from the authors.
Until the author publishes a paper covering the full code, any scientific use must include the author as a co-author or obtain written permission to waive that requirement. The code will become open-source upon publication of the future paper that covers the full code.
Please see the LICENSE file for full terms or contact Marilina Valatsou (mvalatsou@phys.ethz.ch) or Caroline Dorn (cdorn@phys.ethz.ch) to discuss collaboration or permission requests.

## Installation

```bash
# clone repo
git clone https://github.com/ExoInteriors/BOREAS.git
cd BOREAS
# create environment
python -m venv .venv        # or python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
# upgrade pip and install in editable (development) mode
python -m pip install --upgrade pip
python -m pip install -e .
```

This installs BOREAS as an editable package (pip install -e .), so any code edits take effect immediately.

## Quick start (run an example)

### Examples live in examples/configs/. Use the runner:

```bash
# default example, runs json_planets.toml: TRAPPIST-1 b with a pure H2O atmosphere
python examples/run_single_planet.py

# explicit config (relative or absolute path)
python examples/run_single_planet.py --config examples/configs/json_planets.toml

# extra prints, including input params such as mass, radius, Teq, FXUV
python examples/run_single_planet.py -v -c examples/configs/json_planets.toml
```

### Typical output

```bash
Done!
Config: /Users/mvalatsou/PhD/Repos/BOREAS/examples/configs/json_planets.toml
Planet: TRAPPIST-1 b
Regime: EL , RXUV[cm]: 735615412.5639001 , Mdot[g/s]: 502665040.61089957
Mdot_EL_target[g/s]: 502665040.61089957  (analytic EL rate at the EL RXUV; calibrates c_s, not reported as Mdot)
light_major: H , heavy_major: O
T_outflow[K]: 3815.873045427351 , mu_outflow: 3.2676802181478926
phi_H_num: 11540198791750.639 , phi_He_num 0.0 , phi_O_num 2055358407730.817 , phi_C_num 0.0 , phi_N_num 0.0 , phi_S_num 0.0
x_He 0.0 , x_O 0.356208492560815 , x_C 0.0 , x_N 0.0 , x_S 0.0
RS_cold[cm]: 150291609718.336 , escape_base: RXUV = 735615412.5639001 cm
R_homopause[cm]: 730696737.9838252 ( H2O , Kzz: 100000000.0 )
--- regime flags ---
core_powered?        : No  (cold sonic point inside RXUV)
homopause penetrated?: No  (RXUV below the homopause -> fractionation is an upper bound)
```

> Notebook users: relative paths resolve from the notebook’s working directory. Either cd to the repo root first, or build an absolute Path to the TOML.

## How to run your own planet

1. Copy an example file:
```bash
cp examples/configs/my_planet.toml user's_planet.toml
```
2. Edit my_planet.toml (see the full schema below).

3. Run it:
```bash
python examples/run_single_planet.py --config user's_planet.toml
# OR
python examples/run_single_planet.py -v -c user's_planet.toml
```

## Saving results

### The runner can write results to JSON and/or CSV:

```bash
# JSON (full structure)
python examples/run_single_planet.py -c examples/configs/my_planet.toml --json out/my_planet_results.json

# CSV (compact table of key outputs)
python examples/run_single_planet.py -c examples/configs/my_planet.toml --csv  out/my_planet_summary.csv
```

## Config file schema (TOML)

### A config describes one planet and the physics knobs. Example:
```bash
[planet]
name           = "TRAPPIST-1 b"       # look up mass(grams)/radius(cm)/Teq(K) in packaged data (planet_params.json)
FXUV_erg_cm2_s = "from_data"     # incident XUV energy flux at the planet's orbit, erg cm-2 s-1. Four ways to set it:
                                 #   no [xuv].spectrum_file below:  "from_data" -> look up the catalog value   |  a number -> use it directly
                                 #   with [xuv].spectrum_file set:  "from_spectrum" -> integrate the spectrum as given  |  a number -> rescale the spectrum (same shape) to that total flux

[composition]                    # atmospheric mass fractions (sum≈1); auto-normalized if enabled below
H2  = 0
He  = 0
H2O = 1
O2  = 0
CO2 = 0
CO  = 0
CH4 = 0
N2  = 0
NH3 = 0
H2S = 0
SO2 = 0
S2  = 0

[physics]
efficiency = 0.30                 # mass loss efficiency eta (η), dimensionless
albedo     = 0.30
beta       = 0.75                 # dayside redistribution factor, 0.5<b<1
emissivity = 1.0
use_homopause = true              # diagnostic flag only: report the homopause, change nothing
Kzz_cm2_s     = 1e8               # eddy diffusion coefficient, sets the homopause (Kzz = Dzz)

[xuv]                             # optional stellar XUV spectrum; replaces FXUV and E_photon (20 eV)
# spectrum_file     = "path/to/spectrum.txt"  # two columns (x, flux), "#" comments; .csv is comma-separated; relative to this file
spectrum_x_unit   = "angstrom"              # angstrom | nm | eV | keV; flux is erg cm^-2 s^-1 per this unit
spectrum_scale    = 1.0                     # multiplies flux, e.g. (d_ref / a)^2 to bring it to the planet's orbit
spectrum_E_max_eV = 2400.0                  # optional upper end of the XUV band (default: the whole spectrum above 13.6 eV)
chi_from_spectrum = false                   # true: XUV absorption from the spectrum instead of the sigmas below

[xuv.sigma_cm2]                   # atomic cross-sections sigma (σ) (cm^2) for the dissociated outflow at ~20 eV assuming neutral atoms (Verner+1996)
H = 1.89e-18
He = 7.43e-18                     # at its 24.59 eV edge: He cannot absorb at 20 eV (0 there)
O = 1.09e-17
C = 1.01e-17
N = 1.41e-17
S = 3.27e-17

[infrared.kappa_cm2_g]            # IR mass opacities kappa (κ) (cm^2 g^-1) for the bolometric region
H2  = 1e-2
He  = 3e-3                        # collision-induced only (H2-He); He-He is negligible
H2O = 1.0                         # IR (1–30 µm) Planck-mean-ish at ~1000 K, ~1 bar
O2  = 2e-2                        # weak in thermal IR except CIA/quadrupole effects
CO2 = 5e-1                        # moderate to strong in thermal IR
CO  = 1e-1                        # weaker as a band-limited mean
CH4 = 5e-1                        # moderate to strong in thermal IR
N2  = 1e-2                        # weak in thermal IR except CIA/quadrupole effects
NH3 = 5e-1                        # moderate to strong in thermal IR
H2S = 8e-1                        # moderate to strong in thermal IR
SO2 = 1.0                         # strong in thermal IR
S2  = 2e-1                        # probably low to moderate

[diffusion]
he_set = "literature"             # He pairs: "literature" (Mason & Marrero 1970; Cherubim+ 2024, 2025) or "chapman-enskog"

[diffusion.b]                     # b_ij(T) = A * T^gamma (cm^-1 s^-1); keys "HO", "H-O", "HHe" or "He-O", either order; overrides he_set
HO = { A=4.8e17, gamma=0.75 }     # Zahnle and Kasting 1986, O loss with background H
HC = { A=8.303e17, gamma=0.6561 } # Chapman-Enskog LJ 12-6, Svehla 1962, fit over 1-15 kK
HS = { A=5.261e17, gamma=0.6593 } # Chapman-Enskog LJ 12-6, Svehla 1962, fit over 1-15 kK
HN = { A=7.954e17, gamma=0.6561 } # Chapman-Enskog LJ 12-6, Svehla 1962, fit over 1-15 kK
OC = { A=2.523e17, gamma=0.6561 } # Chapman-Enskog LJ 12-6, Svehla 1962, fit over 1-15 kK
ON = { A=2.323e17, gamma=0.6562 } # Chapman-Enskog LJ 12-6, Svehla 1962, fit over 1-15 kK
OS = { A=1.212e17, gamma=0.6692 } # Chapman-Enskog LJ 12-6, Svehla 1962, fit over 1-15 kK
CN = { A=2.486e17, gamma=0.6561 } # Chapman-Enskog LJ 12-6, Svehla 1962, fit over 1-15 kK
CS = { A=1.478e17, gamma=0.6585 } # Chapman-Enskog LJ 12-6, Svehla 1962, fit over 1-15 kK
NS = { A=1.274e17, gamma=0.6643 } # Chapman-Enskog LJ 12-6, Svehla 1962, fit over 1-15 kK

[fractionation]
allow_dynamic_light_major = true  # let the code pick the "light major species" automatically
forced_light_major        = "H"   # used only if the above is false
tol                       = 1e-5
max_iter                  = 100

[advanced]                        # optional overrides
auto_normalize_X = true           # normalize composition if sum!=1
rl_policy        = "auto"         # "auto": switch to RL when recombination-limited; "never": always EL
```

### Notes & units
- FXUV: always pass the **standard incident flux** `FXUV = L_XUV / (4π a²)` [erg cm⁻² s⁻¹], i.e. the stellar XUV energy flux at the planet's orbit (plain inverse-square law).
  BOREAS implements the Owen & Schlichting (2024) Eq. 17 energy-limited rate:

      Ṁ_EL = η · F_XUV · π R³_XUV / (4 G M_p)

  The factor of 4 in the denominator — accounting for the planet intercepting flux over cross-section πR²_XUV but losing mass over the full sphere 4πR²_XUV — is built into BOREAS and gives the correct **global** mass-loss rate.
- Composition: mass fractions of molecules in the bolometric region; outflow is atomic (the code handles the bookkeeping).
  Valid keys are `H2, He, H2O, O2, CO2, CO, CH4, N2, NH3, H2S, SO2, S2`; any other key is an error (so `HE` cannot
  silently drop helium).
- σ_XUV: atomic photoabsorption cross-sections (cm²).
  Defaults have been calculated after Verner+1996.
  He is the exception to the 20 eV reference: its ionization edge is 24.59 eV, so at 20 eV it is strictly
  transparent and a He-dominated outflow would have nothing to absorb the XUV. The default takes He at its edge
  (7.43e-18 cm²), the closest energy at which it absorbs at all; set `He = 0` for the strict 20 eV picture.
- κ_IR: IR mass opacities (cm² g⁻¹) for the hydrostatic molecular layer.
  If `[infrared.kappa_cm2_g]` is omitted, BOREAS uses the same coarse 1-30 µm, Planck-mean-ish defaults shown in the example block above.
  He has no IR bands; its 3e-3 is the H2-He collision-induced share in an H2-rich gas, so a He-dominated gas is more
  transparent than that.
- b_ij(T): binary diffusion coefficients in cm⁻¹ s⁻¹; the model uses gram masses and k_B in erg/K consistently.
  The defaults use H-O from Zahnle & Kasting (1986) and the other nine pairs from Chapman-Enskog theory with a
  Lennard-Jones 12-6 potential, using Svehla (1962) atomic parameters. The exact Chapman-Enskog expression is not a
  power law; the stored `A`/`gamma` are least-squares fits to it over 1000-15000 K, accurate to better than 3%
  (worst case O-S).
  As a cross-check, this set reproduces the independent Zahnle & Kasting H-O value to within 30% across 300-15000 K.
  The older Banks & Kockarts rigid-sphere set (`gamma = 0.5` throughout) is retained as commented values in
  `src/boreas/parameters.py`; it is a factor ~2-3 lower at 3-10 kK and is not exposed as a preset in the example TOMLs.
  Note that above ~10 kK both routes are extrapolations beyond the range their underlying fits were established on.

- Mdot_EL_target: every solution also reports the analytic energy-limited rate above, evaluated at the EL
  R_XUV solution. It is the target that calibrates the outflow sound speed, not the reported `Mdot`, which always
  comes from the isothermal Parker wind. The two agree unless c_s hits the 1.2e6 cm/s (T ~ 10^4 K) cap, the
  Lyα-cooling thermostat; then `Mdot` < `Mdot_EL_target` and η no longer affects `Mdot`. To compare with a code
  that uses the EL formula directly, compare against `Mdot_EL_target`.
- rl_policy: `"auto"` (default) switches to the recombination-limited solution when the recombination time is
  shorter than the flow time; `"never"` always keeps the EL one. From Python, pass `rl_policy=` to
  `MassLoss.compute_mass_loss_parameters()` and `Fractionation.execute()` (the config runner does both).

## Stellar XUV spectrum (optional)

By default the star enters BOREAS through one number, `FXUV`, and one typical photon energy (20 eV) at which
all cross-sections are taken. If you have an XUV spectrum of the host star, BOREAS can use it instead. It then
derives the stellar quantities the model needs, each from 13.6 eV (912 Å) upward:

| quantity | from the spectrum | where it enters |
|---|---|---|
| energy flux `FXUV` | ∫ F_E dE | EL energy budget (`Mdot_EL_target`) |
| ionising photon flux | ∫ F_E / E dE | RL base density, recombination timescale (EL/RL switch) |
| XUV mass absorption χ (optional) | Verner+1996 cross-sections σ_s(E), weighted over F_E | XUV base R_XUV (EL branch) |

By default χ still comes from the `[xuv.sigma_cm2]` values. With `chi_from_spectrum = true` it comes from the
spectrum instead: since σ depends on photon energy, soft photons are stopped high up and hard ones penetrate
deeper, and BOREAS uses χ = 1/Σ, where Σ is the column (g cm⁻²) after which 1/e of the absorbable XUV energy is
still left, i.e. roughly where the bulk of the energy is deposited.

### Run the example

`examples/configs/json_planets.toml` (TRAPPIST-1 b) has the TRAPPIST-1 spectrum in its `[xuv]` block, commented
out. Uncomment it and set `FXUV_erg_cm2_s = "from_spectrum"` to run the planet with the spectrum.

### Spectrum files

A plain text file with two columns, x and flux; lines starting with `#` are ignored (a `.csv` works too):

```text
# wavelength[A]  flux[erg cm^-2 s^-1 A^-1]
15.0 4.240420e-03
16.0 6.107949e-03
...
```

- `spectrum_x_unit` says what x is: wavelength in `"angstrom"` or `"nm"`, or photon energy in `"eV"` or `"keV"`.
  The flux is in erg cm⁻² s⁻¹ per that same unit (per Å, per nm, per eV or per keV).
- Anything below 13.6 eV is ignored, so the file may extend into the FUV.
- `spectrum_file` is relative to the config file.

The flux has to be the one at the planet's orbit `a`. Published spectra are usually given at some other
distance `d_ref`; `spectrum_scale = (d_ref / a)²` converts them:

| spectrum given at | d_ref | spectrum_scale |
|---|---|---|
| 1 au | 1 au | `(1 au / a)²` |
| Earth (observed) | distance to the star | `(d_star / a)²` |
| stellar surface | R★ | `(R★ / a)²` |
| the planet | a | `1` (default) |

`[planet].FXUV_erg_cm2_s` then has a different meaning than usual, since it decides what to do with the
spectrum rather than setting the flux directly:

| `[xuv].spectrum_file` | `FXUV_erg_cm2_s` | flux used |
|---|---|---|
| not set | `"from_data"` | looked up in `planet_params.json` |
| not set | a number | that number |
| set | `"from_spectrum"` | the spectrum's own integrated flux, i.e. `spectrum_scale` as given |
| set | a number | the spectrum rescaled (same shape) so its integral equals that number; `spectrum_scale` then has no effect at all |

The two mismatched combinations (`spectrum_file` set with `"from_data"`, or no `spectrum_file` with
`"from_spectrum"`) should raise an error.

<!-- ### Where to get spectra

- [MUSCLES / Mega-MUSCLES](https://archive.stsci.edu/hlsp/muscles) (MAST): panchromatic spectra of M and K dwarf
  planet hosts, including TRAPPIST-1, as FITS files with wavelength in Å and flux in erg cm⁻² s⁻¹ Å⁻¹ at Earth.
- [X-exoplanets](https://sdc.cab.inta-csic.es/xexoplanets/jsp/homepage.jsp) (Sanz-Forcada et al. 2011): synthetic
  1–912 Å coronal spectra for known planet hosts.

The example `examples/spectra/trappist-1_xuv_1au.txt` is the 15–1000 Å part of the Mega-MUSCLES v25 TRAPPIST-1 SED
(Wilson et al. 2021, doi:10.17909/T9DG6F, CC BY 4.0), moved from Earth to 1 au. A FITS SED becomes such a file with:

```python
import numpy as np
from astropy.io import fits

d = fits.open("hlsp_muscles_..._const-res-sed.fits")[1].data
keep = d["WAVELENGTH"] <= 1000.0
np.savetxt("star_xuv.txt", np.column_stack([d["WAVELENGTH"][keep], d["FLUX"][keep]]))
``` -->

### From Python

```python
from boreas import ModelParams, XUVSpectrum

spec = XUVSpectrum.from_file("star_xuv.txt", x_unit="angstrom", scale=(d_ref / a)**2)
params = ModelParams()
params.set_xuv_spectrum(spec)                          # FXUV = integrated spectrum
params.set_xuv_spectrum(spec, normalize_to=FXUV_t)     # same shape, rescaled (e.g. along an evolution track)
params.set_xuv_spectrum(None)                          # back to a scalar FXUV
params.chi_from_spectrum = True                        # optional: χ from the spectrum
```

With a spectrum set, `params.FXUV` holds its integrated flux; to change the flux, use `normalize_to`.

## Regime flags: homopause and the cold sonic point

Each solution carries diagnostics for *which level actually throttles the escape*, so an
EL/RL number can be checked against the assumptions behind it. Nothing here changes the
computed `Mdot`; they are flags.

Both flags are reported twice. Under their own name they are real booleans, for code
(`core_powered`, `homopause_above_RXUV`); under the same name with a **`?`** appended they
are the strings `"Yes"` / `"No"` (`core_powered?`, `homopause_penetrated?`), so a results
CSV can be read without decoding anything. A row that never reached a solution carries
`"n/a"` on both rather than a blank cell that could be misread as `"No"`.

## Repo Layout

```bash
BOREAS/
├─ src/boreas/
│  ├─ __init__.py
│  ├─ parameters.py             # species tables, constants, composition, cross-sections, diffusion fits
│  ├─ mass_loss.py              # EL/RL solver, Parker wind normalization, RXUV search
│  ├─ fractionation.py          # Odert-style multi-species fractionation
│  ├─ spectrum.py               # optional stellar XUV spectrum input
│  ├─ config.py                 # TOML I/O and param application
│  └─ data/planet_params.json   # M, R, Teq, FXUV planet calatog
├─ examples/
│  ├─ configs/json_planets.toml # default: TRAPPIST-1 b, pure H2O (TRAPPIST-1 spectrum optional)
│  ├─ configs/my_planet.toml
│  └─ run_single_planet.py
├─ tests/
├─ pyproject.toml
└─ README.md
```
