import math

import numpy as np
import pytest

from boreas import Fractionation, MassLoss, ModelParams
from boreas.config import apply_params_from_config, fractionation_runtime_args, mass_loss_runtime_args
from boreas.parameters import ATOMS
from boreas.spectrum import EV_ERG, HC_EV_ANGSTROM, XUVSpectrum, verner96_sigma

ME, RE = 5.972e27, 6.371e8

# a smooth, made-up XUV spectrum: F_lambda [erg cm^-2 s^-1 A^-1] over 1-1200 A
LAM_A = np.geomspace(1.0, 1200.0, 2000)
F_LAM = np.where(LAM_A > 100.0, (LAM_A / 100.0) ** -0.5, 0.3 * (LAM_A / 100.0) ** 0.5)


def _power_law():
    return XUVSpectrum(LAM_A, F_LAM, x_unit="angstrom")


def _run(params, M, R, Teq, frac=False, **kw):
    ml = MassLoss(params)
    res = ml.compute_mass_loss_parameters(np.array([M * ME]), np.array([R * RE]), np.array([Teq]), **kw)
    if frac:
        res = Fractionation(params).execute(res, ml)
    return res[0]


# --- the spectrum itself -----------------------------------------------------

def test_verner_fits_reproduce_default_cross_sections():
    p = ModelParams()
    for atom in ("O", "C", "N", "S"):
        assert math.isclose(verner96_sigma(atom, 20.0), p.sigma_XUV[atom], rel_tol=5e-3)
    # He is taken at its 24.59 eV edge, it does not absorb at 20 eV
    assert math.isclose(verner96_sigma("He", 24.59), p.sigma_XUV["He"], rel_tol=5e-3)
    assert verner96_sigma("He", 20.0) == 0.0


def test_units_give_the_same_spectrum():
    ref = _power_law()
    E = HC_EV_ANGSTROM / LAM_A
    F_E = F_LAM * LAM_A**2 / HC_EV_ANGSTROM
    variants = [
        XUVSpectrum(LAM_A / 10.0, F_LAM * 10.0, x_unit="nm"),
        XUVSpectrum(E, F_E, x_unit="eV"),
        XUVSpectrum(E / 1e3, F_E * 1e3, x_unit="keV"),
    ]
    for spec in variants:
        assert math.isclose(spec.energy_flux(), ref.energy_flux(), rel_tol=1e-10)
        assert math.isclose(spec.photon_flux(), ref.photon_flux(), rel_tol=1e-10)


def test_band_excludes_non_ionising_photons_and_respects_E_max():
    E = np.array([5.0, 10.0, 13.6, 20.0, 100.0])
    F = np.ones_like(E)                                 # 1 erg cm^-2 s^-1 eV^-1
    spec = XUVSpectrum(E, F, x_unit="eV")
    assert math.isclose(spec.energy_flux(), 100.0 - 13.6, rel_tol=1e-12)
    cut = XUVSpectrum(E, F, x_unit="eV", E_max_eV=50.0)
    assert math.isclose(cut.energy_flux(), 50.0 - 13.6, rel_tol=1e-12)
    with pytest.raises(ValueError):
        XUVSpectrum([1.0, 10.0], [1.0, 1.0], x_unit="eV")  # nothing above 13.6 eV


def test_scale_and_mean_photon_energy():
    spec = _power_law()
    doubled = XUVSpectrum(LAM_A, F_LAM, x_unit="angstrom", scale=2.0)
    assert math.isclose(doubled.energy_flux(), 2.0 * spec.energy_flux(), rel_tol=1e-12)
    assert math.isclose(doubled.mean_photon_energy_eV(), spec.mean_photon_energy_eV(), rel_tol=1e-12)
    line = XUVSpectrum.monochromatic(20.0, 5.0)
    assert math.isclose(line.mean_photon_energy_eV(), 20.0, rel_tol=1e-12)
    assert math.isclose(line.photon_flux(), 5.0 / (20.0 * EV_ERG), rel_tol=1e-12)


def test_mass_absorption_line_is_exact_and_spectrum_sits_between_extremes():
    n = {"H": 1.0 / 1.6738e-24}                         # pure atomic H, atoms per gram
    line = XUVSpectrum.monochromatic(20.0, 1.0)
    assert math.isclose(line.mass_absorption(n), verner96_sigma("H", 20.0) * n["H"], rel_tol=1e-12)

    spec = _power_law()
    chi = spec.mass_absorption(n)
    kappa = verner96_sigma("H", spec.E_eV) * n["H"]
    assert kappa.min() < chi < kappa.max()
    # adding hard photons can only push the tau = 1 surface deeper
    harder = XUVSpectrum(LAM_A, np.where(LAM_A < 20.0, 50.0 * F_LAM, F_LAM), x_unit="angstrom")
    assert harder.mass_absorption(n) < chi
    assert spec.mass_absorption({"He": 0.0}) == 0.0


# --- ModelParams with a spectrum ---------------------------------------------

def test_set_spectrum_sets_fxuv_and_guards_it():
    p = ModelParams()
    p.set_xuv_spectrum(_power_law(), normalize_to=1234.0)
    assert math.isclose(p.FXUV, 1234.0, rel_tol=1e-12)
    assert math.isclose(p.fxuv_incident(), 1234.0, rel_tol=1e-12)
    with pytest.raises(ValueError):
        p.update_param("FXUV", 99.0)
    p.FXUV = 99.0                                       # behind the spectrum's back
    with pytest.raises(ValueError):
        p.fxuv_incident()
    p.set_xuv_spectrum(None)
    assert p.fxuv_incident() == 99.0


def test_normalisation_leaves_the_absorption_unchanged():
    p = ModelParams(); p.chi_from_spectrum = True
    p.set_xuv_spectrum(_power_law(), normalize_to=10.0)
    chi_a = p.xuv_cross_section_per_mass()
    p.set_xuv_spectrum(_power_law(), normalize_to=1e5)
    assert math.isclose(p.xuv_cross_section_per_mass(), chi_a, rel_tol=1e-12)


@pytest.mark.parametrize("composition", [{"H2": 1.0}, {"H2": 0.5, "H2O": 0.5}, {"H2O": 1.0}],
                         ids=["H2", "H2-H2O", "H2O"])
@pytest.mark.parametrize("planet", [(5.0, 2.4, 600.0, 2000.0),            # EL
                                    (25.0, 12.05, 491.67, 156368.9)],     # RL
                         ids=["EL", "RL"])
def test_single_line_reproduces_the_scalar_picture(planet, composition):
    """A 20 eV line through the spectrum path == the scalar path at 20 eV."""
    M, R, Teq, F = planet
    scalar = ModelParams(); scalar.set_composition(composition); scalar.update_param("FXUV", F)
    scalar.E_photon = 20.0 * EV_ERG
    scalar.set_sigma_XUV({a: verner96_sigma(a, 20.0) for a in ATOMS})
    spec = ModelParams(); spec.set_composition(composition)
    spec.set_xuv_spectrum(XUVSpectrum.monochromatic(20.0, F))
    spec.chi_from_spectrum = True

    for frac in (False, True):
        a, b = _run(scalar, M, R, Teq, frac), _run(spec, M, R, Teq, frac)
        assert a["regime"] == b["regime"]
        for key in ("RXUV", "cs", "Mdot", "Mdot_EL_target"):
            assert math.isclose(a[key], b[key], rel_tol=1e-9), key


def test_spectrum_runs_through_fractionation():
    p = ModelParams(); p.set_composition({"H2": 0.5, "H2O": 0.5})
    p.set_xuv_spectrum(_power_law(), normalize_to=2000.0)
    r = _run(p, 5.0, 2.4, 600.0, frac=True)
    assert r["regime"] in ("EL", "RL") and r["Mdot"] > 0.0
    assert r["phi_O_num"] >= 0.0


# --- item 4: EL target output and rl_policy ----------------------------------

def test_el_target_is_reported_at_the_el_solution():
    # Vincent's inflated Saturn: c_s is capped, so the reported Mdot is far below the EL target
    p = ModelParams(); p.update_param("FXUV", 156368.94125198497); p.eff = 0.92
    r = _run(p, 1.4997082443552137e+29 / ME, 7679613095.1451025 / RE, 491.67203048224724, rl_policy="never")
    assert r["regime"] == "EL" and r["cs"] == 1.2e6
    expected = MassLoss(p).compute_mdot_el_target(r["RXUV"], r["m_planet"])
    assert math.isclose(r["Mdot_EL_target"], expected, rel_tol=1e-12)
    assert r["Mdot_EL_target"] > 10.0 * r["Mdot"]


def test_unknown_rl_policy_raises():
    p = ModelParams()
    with pytest.raises(ValueError):
        _run(p, 5.0, 2.4, 600.0, rl_policy="sometimes")
    with pytest.raises(ValueError):
        Fractionation(p).execute([], MassLoss(p), rl_policy="if_H")


def test_fractionation_rl_policy_never_keeps_el():
    p = ModelParams(); p.update_param("FXUV", 156368.9)
    planet = (25.0, 12.05, 491.67)
    assert _run(p, *planet, frac=True)["regime"] == "RL"
    ml = MassLoss(p)
    res = ml.compute_mass_loss_parameters(np.array([planet[0] * ME]), np.array([planet[1] * RE]),
                                          np.array([planet[2]]), rl_policy="never")
    assert Fractionation(p).execute(res, ml, rl_policy="never")[0]["regime"] == "EL"


# --- config ------------------------------------------------------------------

def _cfg(tmp_path, fxuv, **xuv):
    path = tmp_path / "spec.txt"
    np.savetxt(path, np.column_stack([LAM_A, F_LAM]), header="wavelength[A] flux[erg/cm2/s/A]")
    return {"planet": {"name": "my_planet", "FXUV_erg_cm2_s": fxuv},
            "composition": {"H2": 1.0},
            "xuv": {"spectrum_file": str(path), "spectrum_x_unit": "angstrom", **xuv},
            "advanced": {"rl_policy": "never"}}


def test_config_spectrum_from_file(tmp_path):
    p = ModelParams()
    apply_params_from_config(_cfg(tmp_path, "from_spectrum", spectrum_scale=3.0), p)
    assert math.isclose(p.FXUV, 3.0 * _power_law().energy_flux(), rel_tol=1e-10)

    p = ModelParams()
    apply_params_from_config(_cfg(tmp_path, 500.0), p)
    assert math.isclose(p.FXUV, 500.0, rel_tol=1e-12)

    with pytest.raises(ValueError):
        apply_params_from_config(_cfg(tmp_path, "from_data"), ModelParams())


def test_config_rl_policy(tmp_path):
    cfg = _cfg(tmp_path, "from_spectrum")
    assert mass_loss_runtime_args(cfg) == {"rl_policy": "never"}
    assert fractionation_runtime_args(cfg)["rl_policy"] == "never"
    assert mass_loss_runtime_args({})["rl_policy"] == "auto"
    with pytest.raises(ValueError):
        mass_loss_runtime_args({"advanced": {"rl_policy": "if_H"}})


def test_chi_uses_fixed_sigmas_unless_asked():
    scalar = ModelParams()
    p = ModelParams(); p.set_xuv_spectrum(_power_law())
    assert p.xuv_cross_section_per_mass() == scalar.xuv_cross_section_per_mass()
    p.chi_from_spectrum = True
    assert p.xuv_cross_section_per_mass() != scalar.xuv_cross_section_per_mass()


def test_config_chi_from_spectrum(tmp_path):
    p = ModelParams()
    apply_params_from_config(_cfg(tmp_path, "from_spectrum", chi_from_spectrum=True), p)
    assert p.chi_from_spectrum is True
    cfg = _cfg(tmp_path, 500.0); cfg["xuv"] = {"chi_from_spectrum": True}
    with pytest.raises(ValueError):
        apply_params_from_config(cfg, ModelParams())
