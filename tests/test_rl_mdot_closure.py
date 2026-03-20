import math

import numpy as np
import pytest

from boreas import MassLoss, ModelParams


def test_rl_branch_returns_mdot_from_rl_closure(monkeypatch: pytest.MonkeyPatch):
    p = ModelParams()
    p.FXUV = 1.0e4

    ml = MassLoss(p)

    rxuv_rl = 1.5 * p.rearth
    rho_eq_rl = 1.0e-12
    rho_pe_rl = 2.0e-13

    monkeypatch.setattr(ml, "find_RXUV_solution_EL", lambda *args, **kwargs: (1.2 * p.rearth, 0.5, 3.0e-12, 4.0e-13))
    monkeypatch.setattr(ml, "compute_sound_speed", lambda *args, **kwargs: 1.2e6)
    monkeypatch.setattr(ml, "compute_mdot_only", lambda *args, **kwargs: 9.9e9)
    monkeypatch.setattr(ml, "find_RXUV_solution_RL", lambda *args, **kwargs: (rxuv_rl, rho_eq_rl, rho_pe_rl))

    m_planet = np.array([3.07 * p.mearth])
    r_planet = np.array([1.28 * p.rearth])
    teq = np.array([1862.0])

    result = ml.compute_mass_loss_parameters(m_planet, r_planet, teq, rl_policy="if_H", light_major="H")[0]

    assert result["regime"] == "RL"

    fxuv_photon = p.FXUV / p.E_photon
    mdot_expected = ml.compute_mdot_rl(rxuv_rl, result["m_planet"], fxuv_photon)

    assert math.isclose(result["Mdot"], mdot_expected, rel_tol=1e-10, abs_tol=0.0)
    assert result["RXUV"] == rxuv_rl
    assert result["rho_eq"] == rho_eq_rl
    assert result["rho_pe"] == rho_pe_rl
