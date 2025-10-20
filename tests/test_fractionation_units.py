# tests/test_fractionation_units.py

import math
import numpy as np
import pytest

from boreas.parameters import ModelParams
from boreas.fractionation import FractionationPhysics, GeneralizedFractionation

# verifies:
# - diffusion coefficients (b_ij):
#     - symmetry check: b_ij(T) == b_ji(T)
#     - magnitude sanity: b_HO(T=1e4 K) ~ 1e20–1e21 cm^-1 s^-1 (detects unit typos)
# - diffusion-limited regime:
#     - heavy major j stalls → φ_i ≈ F_crit = g(m_j−m_i)b_ij / [k_B T (1+f_j)]
#     - confirms correct unit usage (grams, k_B in erg/K, etc.)
# - energy-limited (j-stalled) regime:
#     - small mass flux → φ_i = F_mass / m_i (Fi_EL supply)
#     - verifies consistent mass–flux conversion
# - entrainment fractions x_s:
#     - all x_s within [0,1] for moderate flux
#     - indirectly confirms x-update equations are dimensionless and clamped
# overall:
#     - protects against unit errors (amu↔grams)
#     - ensures correct regime branching and physical bounds in fractionation core


def _params_H2_H2O():
    p = ModelParams()
    # 90% H2, 10% H2O by mass (no auto-normalize so it's exact)
    p.set_composition({"H2": 0.90, "H2O": 0.10}, auto_normalize=False)
    return p

def test_b_pair_symmetry_and_scale():
    """
    To check bij(T) is symmetric (bHO=bOHb) and roughly the right size at 10^4 K. 
    Also catches typos in diffusion fits.
    """
    p = _params_H2_H2O()
    T = 1.0e4
    # symmetry
    b1 = p.b_pair("H", "O", T)
    b2 = p.b_pair("O", "H", T)
    assert math.isclose(b1, b2, rel_tol=1e-12)

    # spot-check magnitude (guards unit typos for b_ij)
    # HO default: A=4.8e17, gamma=0.75 -> ~4.8e20 cm^-1 s^-1 at 1e4 K
    assert 1e20 <= b1 <= 1e21

def test_diffusion_limited_branch_matches_Fcrit():
    """
    High mass-flux case: heavy major j should stall and phi_i == F_crit.
    This tightly couples grams (masses), k_B (erg/K), and b_ij (cm^-1 s^-1).
    If amu gets swapped for grams anywhere, this test will fail by ~1e24.
    """
    p = _params_H2_H2O()
    gen = GeneralizedFractionation(p)

    # K2-18 b-ish numbers
    M = 8.92 * p.mearth
    R = 2.37 * p.rearth
    RXUV = 1.2 * R
    g = p.G * M / RXUV**2
    T = 1.0e4 # K (H-controlled outflow typical of RL/H branch)

    # Compute Fcrit and a mass-flux that puts us in the diffusion-limited window
    i, j, f = FractionationPhysics.choose_light_and_heavy_major(p, RXUV, T, M)
    assert i == "H" and j == "O"

    m = p.species_registry()
    b_ij = p.b_pair(i, j, T)
    Fcrit = g * (m[j]["m"] - m[i]["m"]) * b_ij / (p.k_b * T * (1.0 + f[j]))

    # Initial-iteration denominator (x_s ≈ 1): grams per escaping i-particle
    denom0 = m[i]["m"] + sum(m[s]["m"] * f[s] for s in f if s != i)

    # Pick Fmass so that: m_i*Fcrit < Fmass < denom0*Fcrit
    eps = 0.05 # 5% above the lower bound; requires denom0/m_i > 1.05 (true here)
    assert denom0 > (1.0 + eps) * m[i]["m"]
    Fmass = (1.0 + eps) * m[i]["m"] * Fcrit

    res = gen.compute_fluxes(Fmass, RXUV, T, M)

    assert res["i"] == "H"
    assert res["j"] == "O"
    assert "diffusion-limited" in res["mode"]

    phi_i = res["phi"]["H"]
    assert math.isclose(phi_i, Fcrit, rel_tol=2e-2)

def test_energy_limited_j_stalled_branch_matches_FiEL():
    """
    Small mass-flux case with j still 'stalled' label in the code path:
    phi_i should fall back to the EL supply Fi_EL = Fmass / m_i.
    This also implicitly checks that m_i is in grams.
    """
    p = _params_H2_H2O()
    gen = GeneralizedFractionation(p)

    M = 8.92 * p.mearth
    R = 2.37 * p.rearth
    RXUV = 1.2 * R
    T = 1.0e4
    
    i, j, f = FractionationPhysics.choose_light_and_heavy_major(p, RXUV, T, M)
    m = p.species_registry()
    g = p.G * M / RXUV**2
    b_ij = p.b_pair(i, j, T)
    Fcrit = g * (m[j]["m"] - m[i]["m"]) * b_ij / (p.k_b * T * (1.0 + f[j]))

    # Pick Fmass so that EL supply is below the diffusion cap
    Fmass = 0.90 * m[i]["m"] * Fcrit # => Fi_EL = 0.90*Fcrit

    res = gen.compute_fluxes(Fmass, RXUV, T, M)

    assert res["i"] == "H" and res["j"] == "O"
    assert "energy-limited" in res["mode"]
    Fi_EL = Fmass / m[i]["m"]
    assert math.isclose(res["phi"]["H"], Fi_EL, rel_tol=2e-2)

def test_x_updates_dimensionless_and_bounded():
    """
    For a moderate flux, all entrainment fractions x_s must be in [0,1].
    This indirectly checks the x-update terms are dimensionless.
    """
    p = _params_H2_H2O()
    gen = GeneralizedFractionation(p)

    M = 8.92 * p.mearth
    R = 2.37 * p.rearth
    RXUV = 1.2 * R
    T = 1.0e4

    Fmass = 1e-9 # g cm^-2 s^-1 (moderate)
    res = gen.compute_fluxes(Fmass, RXUV, T, M)

    for s, x in res["x"].items():
        assert 0.0 <= x <= 1.0, f"x_{s} not bounded: {x}"
