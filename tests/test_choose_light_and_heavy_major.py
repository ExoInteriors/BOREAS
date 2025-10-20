import math
import pytest
from boreas.parameters import ModelParams
from boreas.fractionation import FractionationPhysics

# verifies:
# - light-major (i) and heavy-major (j) selection logic:
#     - i: lightest atomic species present (by mass, from ModelParams)
#     - j: among heavier species, pick one with largest base ratio f_j = N_j / N_i
# - tolerance rule (tol_major):
#     - keeps all species within (1 − tol_major) of max f_j as “near-top”
#     - tie-break among those by smallest F_crit (strongest coupling)
# - edge cases:
#     - returns j=None when no heavier species have f>0
#     - forced_light_major errors when species absent, works otherwise
# - physics consistency:
#     - uses masses (grams) and b_ij(T) from parameters
#     - F_crit ∝ (m_j − m_i) * b_ij / [k_B T (1 + f_j)]
# overall:
#     - guards against misordered species, wrong-mass lookups, or tolerance mishandling

class Parameters:
    def __init__(self, b_map=None):
        # pull atomic masses directly from ModelParams for consistency
        mp = ModelParams()
        self.m_H, self.m_C, self.m_N, self.m_O, self.m_S = (
            mp.m_H, mp.m_C, mp.m_N, mp.m_O, mp.m_S
        )
        self.k_b = mp.k_b
        self.G = mp.G
        self._b_map = b_map or {}

    def b_pair(self, a, b, T):
        key = (a, b)
        rkey = (b, a)
        return self._b_map.get(key, self._b_map.get(rkey, 1.0e17))

@pytest.fixture
def base_geo():
    # Any positive geometry; only g enters via Fcrit
    RXUV = 2.0e9      # cm
    m_p  = 5.97e27    # g (≈ 10 M_earth for scale)
    T    = 8000.0     # K
    return RXUV, m_p, T

def test_i_picks_lightest_present(monkeypatch, base_geo):
    p = Parameters()
    RXUV, m_p, T = base_geo

    # No H present => lightest present should be C. Make S absent so O is the largest heavier f.
    def fake_counts(_p):
        return dict(H=0.0, C=1.0, N=2.0, O=3.0, S=0.0) # <-- S=0 so j should be O
    monkeypatch.setattr(FractionationPhysics, "atomic_counts_from_X", staticmethod(fake_counts))

    i, j, f = FractionationPhysics.choose_light_and_heavy_major(
        p, RXUV, T, m_p, allow_dynamic_light_major=True
    )
    assert i == "C"
    assert j == "O" # O now has the largest f among heavier-than-C

def test_j_by_abundance_then_fcrit_tiebreak(monkeypatch, base_geo):
    # Make the b-contrast strong so O clearly wins the Fcrit tie-break.
    p = Parameters(b_map={("H","O"): 1.0e17, ("H","C"): 1.0e19})
    RXUV, m_p, T = base_geo

    def fake_counts(_p):
        # Keep both near-top so tol_major includes both
        return dict(H=10.0, C=9.9, N=0.0, O=10.0, S=0.0)
    monkeypatch.setattr(FractionationPhysics, "atomic_counts_from_X", staticmethod(fake_counts))

    i, j, f = FractionationPhysics.choose_light_and_heavy_major(
        p, RXUV, T, m_p, allow_dynamic_light_major=True, tol_major=0.02
    )
    assert i == "H"
    assert j == "O" # with HO << HC, O has the smaller Fcrit among the near-top set

def test_tol_major_excludes_nearby_but_outside_window(monkeypatch, base_geo):
    # Make C just outside the tolerance window so abundance decides (O picked purely by higher f)
    p = Parameters(b_map={("H","O"): 5.0e17, ("H","C"): 1.0e17}) # even if C had better Fcrit, it won't matter
    RXUV, m_p, T = base_geo

    def fake_counts(_p):
        return dict(H=10.0, C=9.7, O=10.0, N=0.0, S=0.0)
    monkeypatch.setattr(FractionationPhysics, "atomic_counts_from_X", staticmethod(fake_counts))

    i, j, f = FractionationPhysics.choose_light_and_heavy_major(
        p, RXUV, T, m_p, allow_dynamic_light_major=True, tol_major=0.02 # 2% window; C is 3% low
    )
    assert i == "H"
    assert j == "O" # abundance-first, tie-breaker not invoked

def test_j_none_when_no_heavier_candidates(monkeypatch, base_geo):
    p = Parameters()
    RXUV, m_p, T = base_geo

    def fake_counts(_p):
        # Only H present -> no heavier species with f>0 => j is None
        return dict(H=5.0, C=0.0, N=0.0, O=0.0, S=0.0)
    monkeypatch.setattr(FractionationPhysics, "atomic_counts_from_X", staticmethod(fake_counts))

    i, j, f = FractionationPhysics.choose_light_and_heavy_major(
        p, RXUV, T, m_p, allow_dynamic_light_major=True
    )
    assert i == "H"
    assert j is None

def test_forced_light_major_respects_presence(monkeypatch, base_geo):
    p = Parameters()
    RXUV, m_p, T = base_geo

    def fake_counts(_p):
        return dict(H=0.0, C=2.0, N=0.0, O=0.0, S=0.0)
    monkeypatch.setattr(FractionationPhysics, "atomic_counts_from_X", staticmethod(fake_counts))

    # Forcing H when H absent should raise
    with pytest.raises(ValueError):
        FractionationPhysics.choose_light_and_heavy_major(
            p, RXUV, T, m_p, allow_dynamic_light_major=False, forced_light_major="H"
        )

    # Forcing C should succeed and yield j=None (no heavier species present)
    i, j, f = FractionationPhysics.choose_light_and_heavy_major(
        p, RXUV, T, m_p, allow_dynamic_light_major=False, forced_light_major="C"
    )
    assert i == "C"
    assert j is None