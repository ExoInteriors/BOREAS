import math

import numpy as np
import pytest

from boreas import MassLoss, ModelParams
from boreas.mass_loss import CorePoweredEscapeError, SoundSpeedRootError, _yes_no


def _stub_el_solution(monkeypatch, ml, rxuv, cs=1.0e6, mdot=1.0e10):
    """Pin the EL branch so the tests only exercise the regime-boundary flags."""
    monkeypatch.setattr(ml, "find_RXUV_solution_EL", lambda *a, **k: (rxuv, 10.0, 1.0e-12, 1.0e-13))
    monkeypatch.setattr(ml, "compute_sound_speed", lambda *a, **k: cs)
    monkeypatch.setattr(ml, "compute_mdot_only", lambda *a, **k: mdot)


# --------------------------------------------------------------------------
# homopause
# --------------------------------------------------------------------------

def test_n_homopause_is_where_kzz_equals_dzz():
    p = ModelParams()
    p.Kzz = 1.0e8
    T, mmw_i = 800.0, 18.0

    n_homo = p.n_homopause(T, mmw_i)

    assert math.isclose(p.D_molecular(T, n_homo, mmw_i), p.Kzz, rel_tol=1e-12)


def test_n_homopause_scales_inversely_with_kzz():
    p = ModelParams()
    p.Kzz = 1.0e8
    n1 = p.n_homopause(1000.0, 18.0)
    p.Kzz = 1.0e10
    n2 = p.n_homopause(1000.0, 18.0)

    # Dzz ~ 1/n_tot, so stronger mixing pushes the homopause to lower density (higher up)
    assert math.isclose(n1 / n2, 100.0, rel_tol=1e-12)


def test_d_molecular_matches_the_h2_background_scaling():
    p = ModelParams()
    T, n_tot, mmw_i = 1200.0, 1.0e13, 18.0

    expected = 2.2965e17 * T**0.765 / n_tot * ((16.04 / mmw_i) * ((mmw_i + 2.016) / 18.059))**0.5

    assert math.isclose(p.D_molecular(T, n_tot, mmw_i), expected, rel_tol=1e-12)


def test_homopause_molecule_is_the_most_abundant_one_present():
    p = ModelParams()
    p.set_composition({"H2": 0.2, "H2O": 0.8})
    assert p.homopause_molecule() == ("H2O", p.mmw_H2O)

    p.set_composition({"CO2": 0.5, "CO": 0.2, "N2": 0.2, "O2": 0.1})
    assert p.homopause_molecule() == ("CO2", p.mmw_CO2)


def test_homopause_radius_sits_on_the_hydrostatic_profile():
    p = ModelParams()
    ml = MassLoss(p)

    r_p, m_p, teq = 1.5 * p.rearth, 5.0 * p.mearth, 800.0
    mmw_bolo = p.get_mmw_bolometric()
    cs_bolo = np.sqrt(p.k_b * teq / (p.m_H * mmw_bolo))
    rho_bolo = 1.0e-9

    R_homo, n_homo, species = ml.compute_homopause(r_p, m_p, teq, rho_bolo, cs_bolo, mmw_bolo)

    assert R_homo > r_p
    # the returned radius must reproduce n_homo on the isothermal hydrostatic profile
    rho_at_R = rho_bolo * np.exp((p.G * m_p / cs_bolo**2) * (1.0 / R_homo - 1.0 / r_p))
    assert math.isclose(rho_at_R / (mmw_bolo * p.m_H), n_homo, rel_tol=1e-8)
    assert species == "H2"


def test_homopause_clipped_to_photosphere_when_already_separated():
    p = ModelParams()
    ml = MassLoss(p)

    r_p, m_p, teq = 1.5 * p.rearth, 5.0 * p.mearth, 800.0
    mmw_bolo = p.get_mmw_bolometric()
    cs_bolo = np.sqrt(p.k_b * teq / (p.m_H * mmw_bolo))

    # photosphere thinner than the homopause density -> homopause is at/below Rp
    R_homo, n_homo, _ = ml.compute_homopause(r_p, m_p, teq, 1.0e-30, cs_bolo, mmw_bolo)

    assert R_homo == pytest.approx(r_p)


def test_homopause_is_on_by_default_but_can_be_switched_off(monkeypatch: pytest.MonkeyPatch):
    p = ModelParams()
    ml = MassLoss(p)
    _stub_el_solution(monkeypatch, ml, rxuv=1.6 * p.rearth)

    assert p.use_homopause is True

    p.use_homopause = False
    res = ml.compute_mass_loss_parameters(
        np.array([5.0 * p.mearth]), np.array([1.5 * p.rearth]), np.array([500.0])
    )[0]

    for key in ("R_homopause", "n_homopause", "homopause_species", "homopause_above_RXUV",
                "homopause_penetrated?", "Kzz"):
        assert key not in res


def test_homopause_flag_reported_against_rxuv_when_enabled(monkeypatch: pytest.MonkeyPatch):
    p = ModelParams()
    ml = MassLoss(p)
    _stub_el_solution(monkeypatch, ml, rxuv=1.6 * p.rearth)

    monkeypatch.setattr(ml, "compute_homopause", lambda *a, **k: (5.0 * p.rearth, 1.0e12, "H2"))

    res = ml.compute_mass_loss_parameters(
        np.array([5.0 * p.mearth]), np.array([1.5 * p.rearth]), np.array([500.0])
    )[0]

    assert res["homopause_above_RXUV"] is True
    assert res["R_homopause"] == 5.0 * p.rearth
    assert res["homopause_species"] == "H2"
    assert res["Kzz"] == p.Kzz
    # and it stays a flag: the escape base is untouched by it
    assert res["escape_base"] == "RXUV"
    assert res["escape_base_radius"] == res["RXUV"]


def test_homopause_penetrated_yes_no_tracks_rxuv_vs_homopause(monkeypatch: pytest.MonkeyPatch,
                                                              capsys):
    """RXUV below the homopause -> "Yes" plus a warning; above it -> "No" and silence."""
    p = ModelParams()
    ml = MassLoss(p)
    _stub_el_solution(monkeypatch, ml, rxuv=1.6 * p.rearth)
    m, r, teq = np.array([5.0 * p.mearth]), np.array([1.5 * p.rearth]), np.array([500.0])

    # homopause at 5 Rp, well above RXUV = 1.6 Rp -> the XUV base is still well mixed
    monkeypatch.setattr(ml, "compute_homopause", lambda *a, **k: (5.0 * p.rearth, 1.0e12, "H2"))
    res = ml.compute_mass_loss_parameters(m, r, teq)[0]
    assert res["homopause_penetrated?"] == "Yes"
    assert res["homopause_penetrated?"] == _yes_no(res["homopause_above_RXUV"])
    assert "homopause penetrated" in capsys.readouterr().out

    # homopause at 1.01 Rp, below RXUV -> diffusively separated at the base, as assumed
    ml = MassLoss(p)
    _stub_el_solution(monkeypatch, ml, rxuv=1.6 * p.rearth)
    monkeypatch.setattr(ml, "compute_homopause", lambda *a, **k: (1.01 * p.rearth, 1.0e18, "H2"))
    res = ml.compute_mass_loss_parameters(m, r, teq)[0]
    assert res["homopause_penetrated?"] == "No"
    assert res["homopause_above_RXUV"] is False
    assert "homopause penetrated" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# cold sonic point / core-powered mass loss
# --------------------------------------------------------------------------

def test_cold_sonic_point_is_the_bolometric_bondi_radius():
    p = ModelParams()
    ml = MassLoss(p)
    m_p, cs_bolo = 5.0 * p.mearth, 2.0e5

    assert math.isclose(ml.compute_cold_sonic_point(m_p, cs_bolo),
                        p.G * m_p / (2.0 * cs_bolo**2), rel_tol=1e-12)


def _cpml_case(monkeypatch, ml, p):
    """A hot, low-gravity planet whose cold sonic point falls inside RXUV."""
    m, r, teq = np.array([1.0 * p.mearth]), np.array([2.0 * p.rearth]), np.array([2500.0])
    _stub_el_solution(monkeypatch, ml, rxuv=2.5 * r[0])
    return m, r, teq


def test_cpml_flag_does_not_change_the_numbers(monkeypatch: pytest.MonkeyPatch, capsys):
    p = ModelParams()
    ml = MassLoss(p)
    m, r, teq = _cpml_case(monkeypatch, ml, p)

    res = ml.compute_mass_loss_parameters(m, r, teq)[0]

    assert res["core_powered"] is True
    assert res["core_powered?"] == "Yes"
    assert res["regime"] == "EL"           # default policy leaves the solution intact
    assert res["Mdot"] == 1.0e10
    assert res["RS_cold"] <= res["RXUV"]
    # a photoevaporative rate is still returned, so the escape base stays the XUV one
    assert res["escape_base"] == "RXUV"
    assert res["escape_base_radius"] == res["RXUV"]
    assert "core-powered" in capsys.readouterr().out


def test_cpml_nan_policy_blanks_the_solution(monkeypatch: pytest.MonkeyPatch):
    p = ModelParams()
    ml = MassLoss(p)
    m, r, teq = _cpml_case(monkeypatch, ml, p)

    res = ml.compute_mass_loss_parameters(m, r, teq, cpml_policy="nan")[0]

    assert res["regime"] == "CPML"
    assert all(np.isnan(res[k]) for k in ("RXUV", "cs", "Mdot"))
    assert np.isfinite(res["RS_cold"])
    # no photoevaporative number is reported here, so the cold sonic point is the base
    assert res["escape_base"] == "RS_cold"
    assert res["escape_base_radius"] == res["RS_cold"]


def test_cpml_skip_policy_marks_solution_skipped(monkeypatch: pytest.MonkeyPatch):
    p = ModelParams()
    ml = MassLoss(p)
    m, r, teq = _cpml_case(monkeypatch, ml, p)

    res = ml.compute_mass_loss_parameters(m, r, teq, cpml_policy="skip")[0]

    assert res["regime"] == "SKIPPED"
    assert res["skip_reason"] == "CPML"
    assert res["Mdot"] is None
    # the row is dropped from the physics but still says *why*
    assert res["core_powered?"] == "Yes"


def test_cpml_raise_policy_raises(monkeypatch: pytest.MonkeyPatch):
    p = ModelParams()
    ml = MassLoss(p)
    m, r, teq = _cpml_case(monkeypatch, ml, p)

    with pytest.raises(CorePoweredEscapeError) as exc:
        ml.compute_mass_loss_parameters(m, r, teq, cpml_policy="raise")

    assert exc.value.RS_cold <= exc.value.RXUV


def test_photoevaporative_case_is_not_flagged_core_powered(monkeypatch: pytest.MonkeyPatch):
    p = ModelParams()
    ml = MassLoss(p)
    # cold, compact, high gravity -> cold sonic point far outside RXUV
    m, r, teq = np.array([10.0 * p.mearth]), np.array([1.5 * p.rearth]), np.array([300.0])
    _stub_el_solution(monkeypatch, ml, rxuv=2.0 * r[0])

    res = ml.compute_mass_loss_parameters(m, r, teq)[0]

    assert res["core_powered"] is False
    assert res["core_powered?"] == "No"
    assert res["RS_cold"] > res["RXUV"]
    assert res["escape_base"] == "RXUV"
    assert res["escape_base_radius"] == res["RXUV"]


def test_failed_solution_still_carries_the_flag_columns(monkeypatch: pytest.MonkeyPatch):
    """A row that never got a solution reports "n/a", not a blank that reads as "No"."""
    p = ModelParams()
    ml = MassLoss(p)

    def _boom(*a, **k):
        raise SoundSpeedRootError("no root")

    monkeypatch.setattr(ml, "find_RXUV_solution_EL", _boom)

    res = ml.compute_mass_loss_parameters(
        np.array([5.0 * p.mearth]), np.array([1.5 * p.rearth]), np.array([500.0])
    )[0]

    assert res["regime"] == "SKIPPED"
    assert res["core_powered?"] == "n/a"
    assert res["homopause_penetrated?"] == "n/a"
