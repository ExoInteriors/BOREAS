# BOREAS

**BOREAS** is a Python package for modeling **hydrodynamic mass loss** from exoplanet atmospheres,  
including **energy-limited (EL)** and **recombination-limited (RL)** regimes with
**multi-species fractionation** among hydrogen (H), oxygen (O), carbon (C), nitrogen (N), and sulfur (S).

The code couples a **molecular bolometric (IR) region** to a **fully dissociated atomic outflow**,  
tracking composition-dependent escape and diffusive separation self-consistently.

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

This installs BOREAS as an editable package (pip install -e .), 
so any code edits take effect immediately.

## Quick start (run an example)

### Examples live in examples/configs/. Use the runner:

```bash
# default example, runs json_planets.toml
# which has K2-18 b with pure H2/He envelope as default
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
Regime: EL , RXUV[cm]: 735074860.5223168 , Mdot[g/s]: 2006230940.4435651
light_major: H , heavy_major: O
T_outflow[K]: 7174.834055775216 , mu_outflow: 4.778164893161477
phi_H_num: 27642592522227.367 , phi_O_num 9306701767503.541 , phi_C_num 0.0 , phi_N_num 0.0 , phi_S_num 0.0
x_O 0.6733595454203535 , x_C 0.0 , x_N 0.0 , x_S 0.0
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
FXUV_erg_cm2_s = "from_data"     # "from_data" to read .json value, or input value. Incident XUV (energy flux), or stellar irradiance at the planet’s orbit, ergs cm-2 s-1

[composition]                    # atmospheric mass fractions (sum≈1); auto-normalized if enabled below
H2  = 0
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

[xuv.sigma_cm2]                   # atomic cross-sections sigma (σ) (cm^2) for the dissociated outflow at ~20 eV assuming neutral atoms (Verner+1996)
H = 1.89e-18
O = 1.09e-17
C = 1.01e-17
N = 1.41e-17
S = 3.27e-17

[infrared.kappa_cm2_g]            # IR mass opacities kappa (κ) (cm^2 g^-1) for the bolometric region
H2  = 1e-2
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

[diffusion.b]                     # b_ij(T) = A * T^gamma (cm^-1 s^-1); keys can be "HO" or "H-O"
HO = { A=4.8e17, gamma=0.75 }     # Zahnle and Kasting 1986, O loss with background H
HC = { A=1.577e18, gamma=0.5 }    # Banks+Kockarts 1973 fits
HS = { A=1.539e18, gamma=0.5 }    # Banks+Kockarts 1973 fits
HN = { A=1.569e18, gamma=0.5 }    # Banks+Kockarts 1973 fits
OC = { A=5.807e17, gamma=0.5 }    # Banks+Kockarts 1973 fits
ON = { A=5.566e17, gamma=0.5 }    # Banks+Kockarts 1973 fits
OS = { A=4.656e17, gamma=0.5 }    # Banks+Kockarts 1973 fits
CN = { A=5.981e17, gamma=0.5 }    # Banks+Kockarts 1973 fits
CS = { A=5.146e17, gamma=0.5 }    # Banks+Kockarts 1973 fits
NS = { A=4.872e17, gamma=0.5 }    # Banks+Kockarts 1973 fits

[fractionation]
allow_dynamic_light_major = true  # let the code pick the "light major species" automatically
forced_light_major        = "H"   # used only if the above is false
tol                       = 1e-5
max_iter                  = 100

[advanced]                        # optional overrides
auto_normalize_X = true           # normalize composition if sum!=1
```

### Notes & units
- FXUV: if you set a number, use stellar irradiance at orbit (erg cm⁻² s⁻¹) (incident flux). If you use "from_data", the value is read from packaged planet_params.json.
- EL normalization: the model uses the Owen/Schlichting convention with the factor of 4 (absorb over πR², lose over 4πR²). Do not pass a global-mean FXUV (already ÷4), as this is done internally.
- Composition: mass fractions of molecules in the bolometric region; outflow is atomic (the code handles the bookkeeping).
- σ_XUV: atomic photoabsorption cross-sections (cm²).
  Defaults have been calculated after Verner+1996.
- κ_IR: IR mass opacities (cm² g⁻¹) for the hydrostatic molecular layer.
  If `[infrared.kappa_cm2_g]` is omitted, BOREAS uses the same coarse 1-30 µm, Planck-mean-ish defaults shown in the example block above.
- b_ij(T): binary diffusion coefficients in cm⁻¹ s⁻¹; the model uses gram masses and k_B in erg/K consistently.
  The defaults use H-O from Zahnle & Kasting (1986) and the other pairs from Banks & Kockarts-style atomic neutral-neutral fits.
  An alternative Chapman-Enskog LJ 12-6 set is left as commented reference values in parameters.py (line 120), but is not exposed as a preset in the example TOMLs.

## Built-in planet data

Packaged under boreas.data/planet_params.json (mass [M⊕], radius [R⊕], Teq [K], (incident) FXUV [erg cm⁻² s⁻¹]). </br>
Use [planet].name = "<key>" to pull those numbers. You can open that JSON to see available keys.

## Running tests (optional but helpful)

```bash
python -m pip install pytest
python -m pytest -q
```

### This runs unit tests that lock in:
- grams vs amu usage in diffusion/fractionation formulas,
- diffusion- vs energy-limited branch behavior when heavy species “stall”,
- well-formed diffusion fits and bounded entrainment fractions.

## Repo Layout

```bash
BOREAS/
├─ src/boreas/
│  ├─ __init__.py
│  ├─ parameters.py             # constants, composition, cross-sections, diffusion fits
│  ├─ mass_loss.py              # EL/RL solver, Parker wind normalization, RXUV search
│  ├─ fractionation.py          # Odert-style multi-species fractionation
│  ├─ config.py                 # TOML I/O and param application
│  └─ data/planet_params.json   # M, R, Teq, FXUV planet calatog
├─ examples/                    # ship example TOMLs here if desired
│  ├─ configs/k2-18b.toml
│  ├─ configs/my_planet.toml
│  └─ run_single_planet.py
├─ tests/
│  ├─ test_choose_light_and_heavy_major.py
│  ├─ test_consistency_benchmark.py
│  └─ test_fractionation_units.py
├─ pyproject.toml
└─ README.md
```
