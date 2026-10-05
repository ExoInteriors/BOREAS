# tests/test_consistency_benchmark.py

import math
from pathlib import Path

import numpy as np
import pytest

from boreas import ModelParams, MassLoss, Fractionation
from boreas.fractionation import FractionationPhysics
from boreas.parameters import ATOMS

from boreas.config import (
    load_config_toml,
    apply_params_from_config,
    build_inputs_from_config,
    fractionation_runtime_args,
)

# verifies:
# - mass conservation at RXUV: Σm_s*phi_s=Mdot/(4π*RXUV^2)
# - x bounds: each reported x_s∈[0,1] and (by construction) x_i≡1
# - f ratios: reported f_s match atomic count ratios N_s/N_i implied by the config
# - μ consistency: reported mmw_outflow equals the number-flux-weighted mean μ from phis
# - regime conventions: in RL, T_outflow≈1e4 K and cs≈1.2e6 cm/s
# - non-negativity: all number fluxes are ≥ 0
# - stall signal: if mode says j stalled, then phi_j≈0
 
# --- helpers ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples" / "configs"
DEFAULT_CFG = EXAMPLES_DIR / "my_planet.toml"

def approx_rel(a, b, rtol=1e-6, atol=0.0):
    """Relative approximate comparison that works with tiny denominators."""
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)

def atomic_counts_from_X(p):
    return FractionationPhysics.atomic_counts_from_X(p)

# --- tests -----------------------------------------------------------------

# None = composition from the config file
# He cases: He as j, He-rich with H only from water, He as i (no H)
COMPOSITIONS = {
    "config":      None,
    "H2-He-H2O":   {"H2": 0.74, "He": 0.25, "H2O": 0.01},
    "He-H2O":      {"He": 0.95, "H2O": 0.05},
    "He-CO2":      {"He": 0.90, "CO2": 0.10},
}

@pytest.mark.parametrize("composition", list(COMPOSITIONS.values()), ids=list(COMPOSITIONS))
@pytest.mark.parametrize("cfg_path", [DEFAULT_CFG])
def test_pipeline_consistency(cfg_path: Path, composition):
    if not cfg_path.exists():
        pytest.skip(f"Example config not found: {cfg_path}")

    # 1) load config and initialize modules
    params = ModelParams()
    cfg = load_config_toml(cfg_path)
    if composition is not None:
        cfg["composition"] = composition
    _fx_args = apply_params_from_config(cfg, params)

    mass_loss = MassLoss(params)
    fractionation = Fractionation(params)

    # 2) build inputs & run hydro + fractionation for a single planet
    mass, radius, teq = build_inputs_from_config(cfg, params)
    ml_results  = mass_loss.compute_mass_loss_parameters(mass, radius, teq)
    frac_kwargs = fractionation_runtime_args(cfg)
    f_results   = fractionation.execute(ml_results, mass_loss, **frac_kwargs)
    r0 = f_results[0]

    # quick sanity
    assert r0["regime"] in ("EL", "RL")
    assert r0["light_major_i"] in ATOMS
    # heavy_major_j can be None
    assert r0.get("heavy_major_j", None) in (None, *ATOMS)

    # 3) mass conservation at RXUV: sum(m_i * phi_i) == Mdot / (4π R^2)
    RXUV = float(r0["RXUV"])
    Mdot = float(r0["Mdot"])
    Fmass_expected = Mdot / (4.0 * math.pi * RXUV**2)

    reg = params.species_registry()
    m = {s: reg[s]["m"] for s in ATOMS}
    phi = {s: float(r0.get(f"phi_{s}_num", 0.0)) for s in ATOMS}
    Fmass_from_phi = sum(m[s] * phi.get(s, 0.0) for s in m.keys())

    assert approx_rel(Fmass_from_phi, Fmass_expected, rtol=1e-6, atol=0.0), (
        f"Mass flux mismatch: from φ = {Fmass_from_phi:.6e}, from Mdot = {Fmass_expected:.6e}"
    )

    # 4) x’s are physical and x_i == 1
    i = r0["light_major_i"]
    x = {s: r0[f"x_{s}"] for s in ATOMS if s != "H"}
    for k, xv in x.items():
        assert 0.0 <= xv <= 1.0, f"x_{k} out of bounds: {xv}"
    # x_i ≡ 1 isn't reported, so only the bounds above are checked

    # 5) f (base mixing ratios) matches atomic count ratios relative to i
    #    f_s = N_s / N_i (no f_H column: H is always i when present)
    N = atomic_counts_from_X(params)
    Ni = max(N[i], 1e-300)
    f_expected = {s: (N[s] / Ni) for s in ATOMS}
    f_reported = {s: float(r0[f"f_{s}"]) for s in ATOMS if s != "H"}
    if i != "H":
        assert f_expected["H"] == 0.0
        assert f_reported[i] == 1.0

    for s, fv in f_reported.items():
        # absent species -> f ~ 0
        if f_expected[s] == 0.0:
            assert abs(fv) < 1e-12
        else:
            assert approx_rel(fv, f_expected[s], rtol=1e-6, atol=0.0), (
                f"f_{s} mismatch: got {fv:.6e}, expected {f_expected[s]:.6e} (i={i})"
            )

    # 6) μ consistency: mmw_outflow equals flux-weighted mean from φ’s
    mu_from_phi = sum(reg[s]["A"] * phi[s] for s in ATOMS)
    denom = max(sum(phi.values()), 1e-300)
    mu_from_phi /= denom

    mu_reported = float(r0["mmw_outflow"])
    assert approx_rel(mu_reported, mu_from_phi, rtol=1e-6, atol=0.0), (
        f"μ mismatch: reported {mu_reported:.6e}, from φ {mu_from_phi:.6e}"
    )

    # 7) regime conventions
    if r0["regime"] == "RL":
        # pin T_outflow ~ 1e4 K in RL
        assert math.isclose(float(r0["T_outflow"]), 1.0e4, rel_tol=0.05, abs_tol=0.0)
        # and hydro uses cs ~ 1.2e6 cm/s in RL closure
        assert math.isclose(float(r0["cs"]), 1.2e6, rel_tol=0.05, abs_tol=0.0)

    # 8) non-negativity of number fluxes
    for s in phi:
        assert phi[s] >= 0.0, f"Negative number flux for {s}: {phi[s]}"

    # 9) if heavy major j stalled, mode string should reflect that
    mode = r0.get("fractionation_mode", "")
    j = r0.get("heavy_major_j", None)
    if "j stalled" in mode:
        assert j is not None
        assert float(r0[f"phi_{j}_num"]) <= 1e-20