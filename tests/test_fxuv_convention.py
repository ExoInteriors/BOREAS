import math

from boreas import MassLoss, ModelParams


def test_fxuv_helpers_convert_incident_to_global_mean_and_photon_flux():
    p = ModelParams()
    p.FXUV = 1234.0

    assert math.isclose(p.fxuv_incident(), 1234.0, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(p.fxuv_global_mean(), 1234.0 / 4.0, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(p.fxuv_photon_incident(), 1234.0 / p.E_photon, rel_tol=0.0, abs_tol=0.0)


def test_el_target_uses_global_mean_flux_derived_from_incident_input():
    p = ModelParams()
    p.FXUV = 1600.0
    p.eff = 0.3

    ml = MassLoss(p)

    rxuv = 1.7e9
    m_planet = 4.2e28

    mdot = ml.compute_mdot_el_target(rxuv, m_planet)
    expected = p.eff * 4.0 * math.pi * rxuv**3 / (p.G * m_planet) * (p.FXUV / 4.0)

    assert math.isclose(mdot, expected, rel_tol=1e-12, abs_tol=0.0)
