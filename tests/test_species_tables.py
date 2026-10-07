# tests/test_species_tables.py

import math
import random

import pytest

from boreas.parameters import (
    ModelParams, ATOMS, MOLECULES, ATOMS_PER_MOLECULE,
    canonical_atom, pair_key, split_pair,
)
from boreas.config import apply_params_from_config

# verifies:
# - species tables in parameters.py are complete (mmw, opacity, stoichiometry, mass, sigma_XUV)
# - ATOMS is ordered lightest first
# - He diffusion sets cover exactly the He pairs
# - atom counts conserve mass
# - pure He / H-He outflow, He setters and config keys


# --------------------------------------------------------------------------
# table completeness
# --------------------------------------------------------------------------

def test_every_molecule_has_all_its_numbers():
    p = ModelParams()
    assert set(ATOMS_PER_MOLECULE) == set(MOLECULES)
    assert len(p.get_X_tuple()) == len(MOLECULES)
    for mol in MOLECULES:
        atoms = ATOMS_PER_MOLECULE[mol]
        assert set(atoms) <= set(ATOMS), f"{mol} releases an atom not in ATOMS"
        assert hasattr(p, f"X_{mol}")
        assert mol in p.kappa, f"no IR opacity for {mol}"
        # hard-coded mmw vs stoichiometry
        mmw = sum(k * p.species_registry()[a]["A"] for a, k in atoms.items())
        assert getattr(p, f"mmw_{mol}") == mmw
        assert math.isclose(getattr(p, f"mmw_{mol}_outflow"), mmw / sum(atoms.values()))

def test_every_atom_has_all_its_numbers():
    p = ModelParams()
    reg = p.species_registry()
    assert set(reg) == set(ATOMS)
    assert set(p.sigma_XUV) == set(ATOMS)
    for a in ATOMS:
        assert reg[a]["m"] == getattr(p, f"m_{a}")
        assert p.sigma_XUV[a] >= 0.0
    masses = [reg[a]["m"] for a in ATOMS]
    assert masses == sorted(masses), "ATOMS must be ordered lightest first"

def test_he_diffusion_sets_cover_exactly_the_he_pairs():
    he_pairs = {pair_key("He", a) for a in ATOMS if a != "He"}
    for name, fits in ModelParams().he_diffusion_sets.items():
        assert set(fits) == he_pairs, f"{name} set does not cover the He pairs"
        for A, gamma in fits.values():
            assert A > 0.0 and 0.5 <= gamma <= 0.8

def test_pair_keys_are_canonical_and_order_free():
    assert pair_key("He", "H") == pair_key("h", "HE") == "HHe"
    assert pair_key("O", "C") == "CO"
    assert pair_key("S", "He") == "HeS"
    assert split_pair("H-He") == ["H", "He"]
    assert split_pair("OHe") == ["O", "He"]
    with pytest.raises(KeyError):
        canonical_atom("Ar")

# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------

def test_atom_counts_conserve_mass():
    """sum(N_a * A_a) = 1 for any composition."""
    p = ModelParams()
    reg = p.species_registry()
    rng = random.Random(0)
    for _ in range(200):
        X = [rng.random() ** 3 for _ in MOLECULES]
        s = sum(X)
        N = p.atomic_counts([x / s for x in X])
        assert math.isclose(sum(N[a] * reg[a]["A"] for a in ATOMS), 1.0, rel_tol=1e-12)

def test_atom_counts_reject_a_composition_of_the_wrong_length():
    p = ModelParams()
    with pytest.raises(ValueError):
        p.atomic_counts((1.0, 0.0))

def test_pure_he_outflow():
    p = ModelParams()
    p.set_composition({"He": 1.0})
    assert p.get_mmw_bolometric() == 4.0
    assert math.isclose(p.get_mu_outflow_current(), p.m_He / p.m_H)
    assert p.atomic_counts() == {**dict.fromkeys(ATOMS, 0.0), "He": 0.25}
    # one He atom per 4 m_H of gas, each absorbing sigma_He
    assert math.isclose(p.xuv_cross_section_per_mass(), p.sigma_XUV["He"] / (4.0 * p.m_H))
    assert p.kappa_p_all == p.kappa["He"]
    assert p.homopause_molecule() == ("He", 4.0)

def test_bolometric_mmw_is_harmonic_mean_of_mass_fractions():
    """Equal masses of H2 and H2O: 1/mu = 0.5/2 + 0.5/18, i.e. mu = 3.6, not 10."""
    p = ModelParams()
    p.set_composition({"H2": 0.5, "H2O": 0.5})
    assert math.isclose(p.get_mmw_bolometric(), 1.0 / (0.5 / p.mmw_H2 + 0.5 / p.mmw_H2O))
    assert math.isclose(p.get_mmw_bolometric(), 3.6)

def test_he_dilutes_the_xuv_absorption_of_an_h_envelope():
    """He absorbs per atom, but there are 4x fewer He atoms per gram."""
    p = ModelParams()
    p.set_composition({"H2": 0.75, "He": 0.25})
    N = p.atomic_counts()
    assert math.isclose(N["He"] / N["H"], 0.25 / 4.0 / 0.75)
    expected = (p.sigma_XUV["H"] * N["H"] + p.sigma_XUV["He"] * N["He"]) / p.m_H
    assert math.isclose(p.xuv_cross_section_per_mass(), expected)

# --------------------------------------------------------------------------
# He setters and config
# --------------------------------------------------------------------------

def test_composition_rejects_unknown_species():
    """'HE' is a typo, shouldn't silently give X_He = 0."""
    p = ModelParams()
    with pytest.raises(KeyError):
        p.set_composition({"H2": 0.75, "HE": 0.25})

def test_sigma_and_kappa_setters_accept_he():
    p = ModelParams()
    p.set_composition({"He": 1.0})
    p.set_sigma_XUV({"he": 1.0e-18})
    assert p.sigma_XUV["He"] == 1.0e-18
    p.set_kappa({"He": 1.0e-4})
    assert p.kappa_p_all == 1.0e-4

def test_config_he_set_then_single_pair_override():
    p = ModelParams()
    cfg = {
        "planet": {"name": "K2-18 b", "FXUV_erg_cm2_s": 100.0},
        "composition": {"H2": 0.75, "He": 0.25},
        "xuv": {"sigma_cm2": {"He": 5.0e-18}},
        "diffusion": {"he_set": "chapman-enskog",
                      "b": {"He-O": {"A": 1.0e17, "gamma": 0.7}}},
    }
    apply_params_from_config(cfg, p)
    assert p.X_He == 0.25
    assert p.sigma_XUV["He"] == 5.0e-18
    assert p.diffusion_fits["HHe"] == p.he_diffusion_sets["chapman-enskog"]["HHe"]
    assert p.diffusion_fits["HeO"] == (1.0e17, 0.7)   # [diffusion.b] wins over the set
