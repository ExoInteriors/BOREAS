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

def test_every_pair_resolves_from_the_fits_table():
    """
    b_pair() must find every H/C/N/O/S pair in diffusion_fits itself, not via the legacy
    b_XY methods or the geometric-mean fallback.

    The table is written light-species-first ("HC", "OC", "ON"), which is not alphabetical,
    so an alphabetical lookup missed H-C, C-O and N-O. Those three were silently served by
    the legacy methods, whose constants happen to be identical -- meaning the table looked
    complete while three pairs would have degraded to a crude fallback the moment the
    legacy block was deleted.
    """
    p = _params_H2_H2O()
    T = 5.0e3
    species = ("H", "C", "N", "O", "S")

    for a in species:
        for b in species:
            if a >= b:
                continue
            assert (a + b) in p.diffusion_fits or (b + a) in p.diffusion_fits, (
                f"pair {a}-{b} is not in diffusion_fits under either spelling"
            )
            # and the lookup is order-independent
            assert p.b_pair(a, b, T) == p.b_pair(b, a, T)

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

def _params_H2O_CO2():
    p = ModelParams()
    # 50/50 H2O-CO2 by mass: O wins the abundance contest, so j = O and C is a *minor*
    # that is lighter than the heavy major (m_C = 12 < m_O = 16).
    p.set_composition({"H2O": 0.50, "CO2": 0.50}, auto_normalize=False)
    return p

def test_light_minor_still_escapes_when_heavy_major_stalls():
    """
    Regression: the heavy major j is chosen by abundance, not by mass, so minors can be
    lighter than j. When j stalls, those lighter minors must still be entrained -- the
    drag term in Eq. 5 is monotonic in mass, so anything lighter than a species that is
    marginally stalled is easier to drag, not harder.

    Guards against zeroing every minor in the stalled branch, which silently returned
    phi_C = 0 for every C/O-bearing atmosphere.
    """
    p = _params_H2O_CO2()
    gen = GeneralizedFractionation(p)

    M = 8.92 * p.mearth
    R = 2.37 * p.rearth
    RXUV = 1.2 * R
    T = 5.0e3
    g = p.G * M / RXUV**2

    i, j, f = FractionationPhysics.choose_light_and_heavy_major(p, RXUV, T, M)
    assert i == "H" and j == "O"
    assert f["C"] > 0.0

    m = p.species_registry()
    b_ij = p.b_pair(i, j, T)
    Fcrit = g * (m[j]["m"] - m[i]["m"]) * b_ij / (p.k_b * T * (1.0 + f[j]))

    # mass flux low enough that O stalls (below the supply needed to drag it)
    Fmass = 1.05 * m[i]["m"] * Fcrit
    res = gen.compute_fluxes(Fmass, RXUV, T, M)

    assert "j stalled" in res["mode"]
    assert res["phi"]["O"] == 0.0 and res["x"]["O"] == 0.0
    # carbon is lighter than the stalled oxygen, so it must be partially dragged
    assert res["phi"]["C"] > 0.0
    assert 0.0 < res["x"]["C"] <= 1.0

def test_stalled_branch_never_escapes_more_mass_than_supplied():
    """
    The stalled branch used to return before the mass-flux guard. Escaping mass must
    never exceed the energy budget, and may fall below it only when the diffusion cap
    on phi_i actually binds.
    """
    p = _params_H2O_CO2()
    gen = GeneralizedFractionation(p)

    M = 8.92 * p.mearth
    R = 2.37 * p.rearth
    RXUV = 1.2 * R
    T = 5.0e3
    g = p.G * M / RXUV**2

    i, j, f = FractionationPhysics.choose_light_and_heavy_major(p, RXUV, T, M)
    m = p.species_registry()
    Fcrit = g * (m[j]["m"] - m[i]["m"]) * p.b_pair(i, j, T) / (p.k_b * T * (1.0 + f[j]))

    for scale in (0.5, 0.9, 1.05, 5.0, 50.0):
        Fmass = scale * m[i]["m"] * Fcrit
        res = gen.compute_fluxes(Fmass, RXUV, T, M)

        Fout = sum(m[s]["m"] * res["phi"][s] for s in ("H", "C", "N", "O", "S"))
        assert math.isclose(Fout, res["Fmass_out"], rel_tol=1e-12)
        assert Fout <= Fmass * (1.0 + 1e-6), f"scale={scale}: escapes more mass than supplied"
        if "diffusion-limited" not in res["mode"]:
            assert math.isclose(Fout, Fmass, rel_tol=1e-6), f"scale={scale}: budget not used up"
