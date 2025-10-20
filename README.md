# BOREAS

Hydrodynamic mass loss (EL/RL) with multi-species fractionation (H–O–C–N–S).  
The bolometric (IR) region is **molecular**, while the escaping outflow is **fully dissociated atoms**.

> Package name: **boreas** · Import name: **boreas**  
> Requires **Python ≥ 3.9**

---

## Installation

### Option A — install from this repo

```bash
# from the repo root:
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### Option B — one-liner install from GitHub (link might be broken)

```bash
pip install "boreas @ git+https://github.com/ExoInteriors/BOREAS.git@proteus#"
```

## Quick start (run an example)

### Examples live in examples/configs/. Use the runner:

```bash
# default example (my_planet.toml)
python examples/run_single_planet.py

# explicit config (relative or absolute path)
python examples/run_single_planet.py --config examples/configs/my_planet.toml

# extra prints, including input params such as mass, radius, Teq, FXUV
python examples/run_single_planet.py -v -c examples/configs/my_planet.toml
```

### Typical output

```bash
Config: /.../examples/configs/my_planet.toml
Planet: my_planet
Regime: EL  RXUV[cm]: 1.23e+09  Mdot[g/s]: 4.56e+08
light_major: H  heavy_major: O
T_outflow[K]: 10000.0  mu_outflow: 1.02
```

> Notebook users: relative paths resolve from the notebook’s working directory. Either cd to the repo root first, or build an absolute Path to the TOML.


## How to run your own planet

1. Copy an example file:
```bash
cp examples/configs/my_planet.toml my_other_planet.toml
```
2. Edit my_planet.toml (see the full schema below).
3. Run it:
```bash
python examples/run_single_planet.py --config my_other_planet.toml
# OR
python examples/run_single_planet.py -v -c my_other_planet.toml
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
name           = "my_planet"   # use packaged properties (mass, radius, Teq)
FXUV_erg_cm2_s = "from_data"   # or a number (stellar irradiance at orbit; cm^-2 s^-1 * erg)

[composition]                    # atmospheric mass fractions (sum≈1); auto-normalized if enabled below
H2  = 0.10
H2O = 0.10
O2  = 0.1
CO2 = 0.1
CO  = 0.1
CH4 = 0.1
N2  = 0.1
NH3 = 0.1
H2S = 0.1
SO2 = 0.05
S2  = 0.05

[physics]
efficiency = 0.30                 # mass loss efficiency eta (η), dimensionless
albedo     = 0.30
beta       = 0.75                 # dayside redistribution factor, 0.5<b<1
emissivity = 1.0

[xuv.sigma_cm2]                   # atomic cross-sections sigma (σ) (cm^2) for the dissociated outflow
H = 1.89e-18                      # representative neutral-atom σ at ~25 eV, sigma(E) ≈ sigma(25 eV) * (E / 25 eV)^(-3)
O = 5.00e-18
C = 6.00e-18
N = 7.00e-18
S = 1.20e-17

[infrared.kappa_cm2_g]            # IR mass opacities kappa (κ) (cm^2 g^-1) for the bolometric region
H2  = 1.0e-2
H2O = 1.0                         # IR (1–30 µm) Planck-mean-ish at ~1000 K, ~1 bar
O2  = 2.0e-2
CO2 = 5.0e-1
CO  = 1.0e-1
CH4 = 5.0e-1
N2  = 1.0e-2
NH3 = 5.0e-1
H2S = 8.0e-1
SO2 = 1.0
S2  = 2.0e-1

[diffusion.b]                     # b_ij(T) = A * T^gamma (cm^-1 s^-1); keys can be "HO" or "H-O"
HO = { A=4.8e17, gamma=0.75 }     # Zahnle and Kasting 1986, O loss with background H
HC = { A=5.5e17, gamma=0.72 }     # H–C slightly faster than H–N/O
HN = { A=5.0e17, gamma=0.73 }
HS = { A=4.6e17, gamma=0.72 }     # H–S a touch slower (heavier partner)
OC = { A=9.0e16, gamma=0.77 }     # heavy–heavy all ~1e17 with small spread
ON = { A=9.5e16, gamma=0.78 }
OS = { A=8.5e16, gamma=0.78 }
CN = { A=8.5e16, gamma=0.76 }
CS = { A=7.8e16, gamma=0.76 }
NS = { A=8.0e16, gamma=0.77 }

[fractionation]
allow_dynamic_light_major = true  # let the code pick the "light major species" automatically
forced_light_major        = "H"   # used only if the above is false
tol                       = 1e-5
max_iter                  = 100

[advanced]                        # optional overrides
auto_normalize_X = true           # normalize composition if sum!=1
```

### Notes & units
- FXUV: if you set a number, use stellar irradiance at orbit (erg cm⁻² s⁻¹). If you use "from_data", the value is read from packaged planet_params.json.
- EL normalization: the model uses the Owen/Schlichting convention with the factor of 4 (absorb over πR², lose over 4πR²). If you pass a global-mean FXUV (already ÷4), set that numeric value directly.
- Composition: mass fractions of molecules in the bolometric region; outflow is atomic (the code handles the bookkeeping).
- σ_XUV: atomic photoabsorption cross-sections (cm²).
- κ_IR: IR mass opacities (cm² g⁻¹) for the hydrostatic molecular layer.
- b_ij(T): binary diffusion coefficients in cm⁻¹ s⁻¹; the model uses gram masses and k_B in erg/K consistently.

## Built-in planet data

### Packaged under boreas.data/planet_params.json (mass [M⊕], radius [R⊕], Teq [K], FXUV [erg cm⁻² s⁻¹]). 
### Use [planet].name = "<key>" to pull those numbers. You can open that JSON to see available keys.

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