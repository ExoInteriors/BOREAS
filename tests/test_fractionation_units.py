# tests/test_fractionation_units.py

import math
import numpy as np
import pytest

from boreas.parameters import ModelParams, ATOMS, pair_key, split_pair
from boreas.fractionation import FractionationPhysics, GeneralizedFractionation

# verifies:
# - b_ij symmetric, b_HO(1e4 K) ~ 1e20-1e21 cm^-1 s^-1 (catches unit typos)
# - every pair of ATOMS is in diffusion_fits under its canonical key
# - j escapes only for F_mass > m_i F_crit, F_crit = g(m_j−m_i)b_ij / [k_B T (1+f_j)]
# - j stalled: φ_i = F_mass / m_i
# - all x_s in [0,1]
# - minors lighter than j still escape when j stalls, mass budget always used up
# - He: diffusion sets + overrides, He as j, He as i


def _params_H2_H2O():
    p = ModelParams()
    # 90% H2, 10% H2O by mass (no auto-normalize so it's exact)
    p.set_composition({"H2": 0.90, "H2O": 0.10}, auto_normalize=False)
    return p

def test_b_pair_symmetry_and_scale():
    """b_ij symmetric and the right order of magnitude at 1e4 K."""
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
    Every pair must be in diffusion_fits under one canonical key, so nothing falls back
    to the geometric mean (mixed spellings like "HC"/"OC" used to break lookups).
    """
    p = _params_H2_H2O()
    T = 5.0e3

    for k in p.diffusion_fits:
        assert k == pair_key(*split_pair(k)), f"diffusion_fits key {k!r} is not canonical"

    for n, a in enumerate(ATOMS):
        for b in ATOMS[n + 1:]:
            assert pair_key(a, b) in p.diffusion_fits, f"pair {a}-{b} missing from diffusion_fits"
            # and the lookup is order-independent
            assert p.b_pair(a, b, T) == p.b_pair(b, a, T)
    assert not p._warned_pairs, "a pair fell through to the geometric-mean fallback"

def test_heavy_major_crossover_sits_at_m_i_Fcrit():
    """
    j escapes once F_mass > m_i*F_crit. Above that, Eq. 4 + mass budget give
        phi_i = (F_mass + m_j f_j F_crit) / (m_i + m_j f_j),   x_j = 1 - F_crit/phi_i
    Mixes grams, erg/K and cm^-1 s^-1, so an amu/gram mix-up fails by ~1e24.
    """
    p = _params_H2_H2O()
    gen = GeneralizedFractionation(p)

    # K2-18 b-ish numbers
    M = 8.92 * p.mearth
    R = 2.37 * p.rearth
    RXUV = 1.2 * R
    g = p.G * M / RXUV**2
    T = 1.0e4 # K (H-controlled outflow typical of RL/H branch)

    i, j, f = FractionationPhysics.choose_light_and_heavy_major(p, RXUV, T, M)
    assert i == "H" and j == "O"

    m = p.species_registry()
    b_ij = p.b_pair(i, j, T)
    Fcrit = g * (m[j]["m"] - m[i]["m"]) * b_ij / (p.k_b * T * (1.0 + f[j]))
    mi, mjfj = m[i]["m"], m[j]["m"] * f[j]

    # just below the crossover: j stays behind, i takes the whole budget
    res = gen.compute_fluxes(0.99 * mi * Fcrit, RXUV, T, M)
    assert res["mode"] == "energy-limited (j stalled)"
    assert res["x"][j] == 0.0
    assert math.isclose(res["phi"][i], 0.99 * Fcrit, rel_tol=1e-9)

    # just above: j is dragged
    Fmass = 1.05 * mi * Fcrit
    res = gen.compute_fluxes(Fmass, RXUV, T, M)
    assert res["mode"] == "energy-limited"
    phi_i = (Fmass + mjfj * Fcrit) / (mi + mjfj)
    assert math.isclose(res["phi"][i], phi_i, rel_tol=1e-9)
    assert math.isclose(res["x"][j], 1.0 - Fcrit / phi_i, rel_tol=1e-9)
    assert 0.0 < res["x"][j] < 0.1
    assert math.isclose(res["Fmass_out"], Fmass, rel_tol=1e-9)

def test_energy_limited_j_stalled_branch_matches_FiEL():
    """j stalled -> phi_i = Fi_EL = Fmass / m_i (also checks m_i is in grams)."""
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
    """All x_s in [0,1] for a moderate flux."""
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
    # 50/50 by mass: j = O, and C is a minor lighter than j
    p.set_composition({"H2O": 0.50, "CO2": 0.50}, auto_normalize=False)
    return p

def test_light_minor_still_escapes_when_heavy_major_stalls():
    """
    j is picked by abundance, not mass, so a minor can be lighter than j.
    If j stalls, lighter minors still get dragged (phi_C used to come out as 0).
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

    # low enough that O stalls
    Fmass = 0.95 * m[i]["m"] * Fcrit
    res = gen.compute_fluxes(Fmass, RXUV, T, M)

    assert "j stalled" in res["mode"]
    assert res["phi"]["O"] == 0.0 and res["x"]["O"] == 0.0
    # C is lighter than O, so it still escapes
    assert res["phi"]["C"] > 0.0
    assert 0.0 < res["x"]["C"] <= 1.0

def test_stalled_branch_never_escapes_more_mass_than_supplied():
    """Escaping mass flux = F_mass on both sides of the crossover."""
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

        Fout = sum(m[s]["m"] * res["phi"][s] for s in ATOMS)
        assert math.isclose(Fout, res["Fmass_out"], rel_tol=1e-12)
        assert math.isclose(Fout, Fmass, rel_tol=1e-6), f"scale={scale}: budget not used up"

# --------------------------------------------------------------------------
# helium
# --------------------------------------------------------------------------

def test_he_pairs_default_to_literature_and_switch_to_chapman_enskog():
    p = ModelParams()
    T = 1.0e4
    # Mason & Marrero (1970): 1.04e20 m^-1 s^-1 -> 1.04e18 cm^-1 s^-1, ~8.8e20 at 1e4 K
    assert math.isclose(p.b_pair("H", "He", T), 1.04e18 * T**0.732, rel_tol=1e-12)
    assert 5e20 <= p.b_pair("H", "He", T) <= 2e21
    # Cherubim+ 2025: 2.61e19 m^-1 s^-1 -> 2.61e17 cm^-1 s^-1
    assert math.isclose(p.b_pair("O", "He", T), 2.61e17 * T**0.75, rel_tol=1e-12)

    p.use_he_diffusion_set("chapman-enskog")
    for k, (A, gamma) in p.he_diffusion_sets["chapman-enskog"].items():
        a, b = split_pair(k)
        assert math.isclose(p.b_pair(a, b, T), A * T**gamma, rel_tol=1e-12)
    # non-He pairs are untouched
    assert math.isclose(p.b_pair("H", "O", T), 4.8e17 * T**0.75, rel_tol=1e-12)

    with pytest.raises(KeyError):
        p.use_he_diffusion_set("rigid-sphere")

def test_override_replaces_the_default_in_both_orders():
    """Overrides replace the default whatever the key spelling."""
    p = ModelParams()
    T = 3.0e3
    p.set_diffusion_fits({"OC": {"A": 1.0e17, "gamma": 0.7},
                          "O-He": {"A": 2.0e17, "gamma": 0.7},
                          "HHe": {"A": 3.0e18, "gamma": 0.7}})
    for a, b, A in (("O", "C", 1.0e17), ("O", "He", 2.0e17), ("H", "He", 3.0e18)):
        assert p.b_pair(a, b, T) == p.b_pair(b, a, T) == A * T**0.7

def test_diffusion_keys_that_do_not_name_two_atoms_are_rejected():
    p = ModelParams()
    for bad in ("HHE", "ho", "HeOS", "H"):
        with pytest.raises(ValueError):
            p.set_diffusion_fits({bad: {"A": 1e17, "gamma": 0.7}})
    with pytest.raises(ValueError):
        p.set_diffusion_fits({"He-HE": {"A": 1e17, "gamma": 0.7}})
    with pytest.raises(KeyError):
        p.set_diffusion_fits({"H-Ar": {"A": 1e17, "gamma": 0.7}})

def _h_he_case(composition):
    p = ModelParams()
    p.set_composition(composition, auto_normalize=False)
    M = 8.92 * p.mearth
    RXUV = 1.2 * 2.37 * p.rearth
    T = 1.0e4
    g = p.G * M / RXUV**2
    i, j, f = FractionationPhysics.choose_light_and_heavy_major(p, RXUV, T, M)
    m = {s: v["m"] for s, v in p.species_registry().items()}
    Fcrit = g * (m[j] - m[i]) * p.b_pair(i, j, T) / (p.k_b * T * (1.0 + f[j]))
    return p, GeneralizedFractionation(p), M, RXUV, T, i, j, f, m, Fcrit

def test_he_stalls_below_fcrit_and_is_dragged_above_it():
    p, gen, M, RXUV, T, i, j, f, m, Fcrit = _h_he_case({"H2": 0.75, "He": 0.25})
    assert (i, j) == ("H", "He")

    # below the crossover He stays behind and H takes the whole budget
    res = gen.compute_fluxes(0.95 * m["H"] * Fcrit, RXUV, T, M)
    assert res["mode"] == "energy-limited (j stalled)"
    assert res["x"]["He"] == 0.0 and res["phi"]["He"] == 0.0
    assert math.isclose(res["phi"]["H"], 0.95 * Fcrit, rel_tol=1e-9)

    # above it He is dragged, more so for higher flux, with x_He = 1 - F_crit/phi_H (Eq. 4)
    x_prev = 0.0
    for scale in (1.05, 1.5, 3.0, 10.0, 100.0):
        res = gen.compute_fluxes(scale * m["H"] * Fcrit, RXUV, T, M)
        assert res["mode"] == "energy-limited"
        x_he = res["x"]["He"]
        assert x_prev < x_he < 1.0
        assert math.isclose(x_he, 1.0 - Fcrit / res["phi"]["H"], rel_tol=1e-9)
        assert math.isclose(res["Fmass_out"], res["Fmass_in"], rel_tol=1e-6)
        x_prev = x_he

def test_minor_locked_to_the_he_background_escapes_like_he():
    """
    With He as j, O diffuses through He (b_jk = b_He,k in Eq. 5): it lags behind He,
    and with b_HeO -> 0 it escapes exactly like He.
    """
    p, gen, M, RXUV, T, i, j, f, m, Fcrit = _h_he_case({"H2": 0.74, "He": 0.25, "H2O": 0.01})
    assert (i, j) == ("H", "He")
    Fmass = 4.0 * Fcrit * (m["H"] + f["He"] * m["He"])

    res = gen.compute_fluxes(Fmass, RXUV, T, M)
    assert 0.0 < res["x"]["He"] < 1.0
    assert res["x"]["O"] < res["x"]["He"]

    p.set_diffusion_fits({"He-O": {"A": 1.0e8, "gamma": 0.75}})
    res = gen.compute_fluxes(Fmass, RXUV, T, M)
    assert math.isclose(res["x"]["O"], res["x"]["He"], rel_tol=1e-6)

def test_he_takes_over_as_light_major_once_h_is_gone():
    p = ModelParams()
    p.set_composition({"He": 0.9, "CO2": 0.1}, auto_normalize=False)
    gen = GeneralizedFractionation(p)
    M = 8.92 * p.mearth
    RXUV = 1.2 * 2.37 * p.rearth
    T = 5.0e3

    res = gen.compute_fluxes(1e-10, RXUV, T, M)
    assert (res["i"], res["j"]) == ("He", "O")
    assert res["phi"]["H"] == 0.0
    assert math.isclose(res["Fmass_out"], res["Fmass_in"], rel_tol=1e-6)
    # C is lighter than j = O, so it's dragged more easily
    assert 0.0 < res["x"]["O"] < res["x"]["C"] <= 1.0
