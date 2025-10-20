# tests/test_consistency_benchmark.py

import math
from pathlib import Path

import numpy as np
import pytest

from boreas import ModelParams, MassLoss, Fractionation
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
# - diffusion-limited signal: if mode says diffusion-limited / j stalled, then phi_j≈0
 
# --- helpers ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples" / "configs"
DEFAULT_CFG = EXAMPLES_DIR / "my_planet.toml"

def approx_rel(a, b, rtol=1e-6, atol=0.0):
    """Relative approximate comparison that works with tiny denominators."""
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)

def atomic_counts_from_X(p):
    """Reproduce the f-ratios check from your FractionationPhysics.atomic_counts_from_X."""
    (X_H2, X_H2O, X_O2, X_CO2, X_CO, X_CH4, X_N2, X_NH3, X_H2S, X_SO2, X_S2) = p.get_X_tuple()

    def part(X, mmw):
        return X / mmw if X > 0 else 0.0

    N_H2  = part(X_H2,  p.mmw_H2_outflow)
    N_H2O = part(X_H2O, p.mmw_H2O_outflow)
    N_O2  = part(X_O2,  p.mmw_O2_outflow)
    N_CO2 = part(X_CO2, p.mmw_CO2_outflow)
    N_CO  = part(X_CO,  p.mmw_CO_outflow)
    N_CH4 = part(X_CH4, p.mmw_CH4_outflow)
    N_N2  = part(X_N2,  p.mmw_N2_outflow)
    N_NH3 = part(X_NH3, p.mmw_NH3_outflow)
    N_H2S = part(X_H2S, p.mmw_H2S_outflow)
    N_SO2 = part(X_SO2, p.mmw_SO2_outflow)
    N_S2  = part(X_S2,  p.mmw_S2_outflow)

    N = dict(
        H = 2*N_H2 + 2*N_H2O + 4*N_CH4 + 3*N_NH3 + 2*N_H2S,
        O = 1*N_H2O + 2*N_O2  + 2*N_CO2 + 1*N_CO  + 2*N_SO2,
        C = 1*N_CO2 + 1*N_CO  + 1*N_CH4,
        N = 2*N_N2  + 1*N_NH3,
        S = 1*N_H2S + 1*N_SO2 + 2*N_S2,
    )
    return N

# --- tests -----------------------------------------------------------------

@pytest.mark.parametrize("cfg_path", [DEFAULT_CFG])
def test_pipeline_consistency(cfg_path: Path):
    if not cfg_path.exists():
        pytest.skip(f"Example config not found: {cfg_path}")

    # 1) load config and initialize modules
    params = ModelParams()
    cfg = load_config_toml(cfg_path)
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
    assert r0["light_major_i"] in ("H", "C", "N", "O", "S")
    # heavy_major_j can be None
    assert r0.get("heavy_major_j", None) in (None, "H", "C", "N", "O", "S")

    # 3) mass conservation at RXUV: sum(m_i * phi_i) == Mdot / (4π R^2)
    RXUV = float(r0["RXUV"])
    Mdot = float(r0["Mdot"])
    Fmass_expected = Mdot / (4.0 * math.pi * RXUV**2)

    m = {"H": params.m_H, "C": params.m_C, "N": params.m_N, "O": params.m_O, "S": params.m_S}
    phi = {
        "H": float(r0.get("phi_H_num", 0.0)),
        "O": float(r0.get("phi_O_num", 0.0)),
        "C": float(r0.get("phi_C_num", 0.0)),
        "N": float(r0.get("phi_N_num", 0.0)),
        "S": float(r0.get("phi_S_num", 0.0)),
    }
    Fmass_from_phi = sum(m[s] * phi.get(s, 0.0) for s in m.keys())

    assert approx_rel(Fmass_from_phi, Fmass_expected, rtol=1e-6, atol=0.0), (
        f"Mass flux mismatch: from φ = {Fmass_from_phi:.6e}, from Mdot = {Fmass_expected:.6e}"
    )

    # 4) x’s are physical and x_i == 1
    i = r0["light_major_i"]
    x = {"O": r0["x_O"], "C": r0["x_C"], "N": r0["x_N"], "S": r0["x_S"]}
    for k, xv in x.items():
        assert 0.0 <= xv <= 1.0, f"x_{k} out of bounds: {xv}"
    # by definition x_i ≡ 1 (not stored separately); ensure implied via construction:
    # if any species is the light major, its 'x' isn't reported and should be conceptually 1.
    # at least enforce that no reported x exceeds 1 and none are negative (done above).

    # 5) f (base mixing ratios) matches atomic count ratios relative to i
    #    f_s = N_s / N_i
    N = atomic_counts_from_X(params)
    Ni = max(N[i], 1e-300)
    f_expected = {s: (N[s] / Ni) for s in "HCNOS"}
    # reported f_*
    f_reported = {
        "H": 1.0,  # by definition relative to i (if i==H this equals 1; if i!=H this is N_H/N_i)
        "C": float(r0["f_C"]),
        "N": float(r0["f_N"]),
        "O": float(r0["f_O"]),
        "S": float(r0["f_S"]),
    }

    # H is not explicitly stored; if i != H, the expected ratio is N_H/N_i; if i==H, that is 1.
    if i != "H":
        assert approx_rel(f_reported.get("H", 1.0), f_expected["H"], rtol=1e-6, atol=0.0)

    for s in ("C", "N", "O", "S"):
        # If a species is truly absent, f_expected could be 0; accept tiny absolute error there.
        if f_expected[s] == 0.0:
            assert abs(f_reported[s]) < 1e-12
        else:
            assert approx_rel(f_reported[s], f_expected[s], rtol=1e-6, atol=0.0), (
                f"f_{s} mismatch: got {f_reported[s]:.6e}, expected {f_expected[s]:.6e} (i={i})"
            )

    # 6) μ consistency: mmw_outflow equals flux-weighted mean from φ’s
    mu_from_phi = (
        params.am_h * phi["H"]
        + params.am_o * phi["O"]
        + params.am_c * phi["C"]
        + params.am_n * phi["N"]
        + params.am_s * phi["S"]
    )
    denom = max(sum(phi.values()), 1e-300)
    mu_from_phi /= denom

    mu_reported = float(r0["mmw_outflow"])
    assert approx_rel(mu_reported, mu_from_phi, rtol=1e-6, atol=0.0), (
        f"μ mismatch: reported {mu_reported:.6e}, from φ {mu_from_phi:.6e}"
    )

    # 7) regime conventions
    if r0["regime"] == "RL":
        # You pin T_outflow ~ 1e4 K in RL
        assert math.isclose(float(r0["T_outflow"]), 1.0e4, rel_tol=0.05, abs_tol=0.0)
        # And your hydro uses cs ~ 1.2e6 cm/s in RL closure
        assert math.isclose(float(r0["cs"]), 1.2e6, rel_tol=0.05, abs_tol=0.0)

    # 8) non-negativity of number fluxes
    for s in phi:
        assert phi[s] >= 0.0, f"Negative number flux for {s}: {phi[s]}"

    # 9) if heavy major j stalled, mode string should reflect that; else not diffusion-limited
    mode = r0.get("fractionation_mode", "")
    j = r0.get("heavy_major_j", None)
    if "diffusion-limited" in mode or "j stalled" in mode:
        # when j stalls, all heavier species than i should have zero flux (or near zero).
        if j is not None:
            pj = {"H": "phi_H_num", "C": "phi_C_num", "N": "phi_N_num", "O": "phi_O_num", "S": "phi_S_num"}[j]
            assert float(r0[pj]) <= 1e-20