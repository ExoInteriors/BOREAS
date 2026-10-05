import math
import pytest
from boreas.parameters import ModelParams, ATOMS
from boreas.fractionation import FractionationPhysics

# verifies:
# - i = lightest species present, j = most abundant heavier one (f_j = N_j / N_i)
# - species within tol_major of the top f_j are tie-broken by smallest F_crit
# - j = None when nothing heavier is present
# - forced_light_major raises if that species is absent
# - He as heavy major (H/He envelope) and as light major (no H)

class Parameters:
    def __init__(self, b_map=None):
        # real masses from ModelParams, b_ij from b_map
        mp = ModelParams()
        self.species_registry = mp.species_registry
        self.k_b = mp.k_b
        self.G = mp.G
        self._b_map = b_map or {}

    def b_pair(self, a, b, T):
        key = (a, b)
        rkey = (b, a)
        return self._b_map.get(key, self._b_map.get(rkey, 1.0e17))

def counts(**N):
    """Atom counts, zero for anything not given."""
    return {s: float(N.get(s, 0.0)) for s in ATOMS}

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
        return counts(H=0.0, C=1.0, N=2.0, O=3.0, S=0.0) # <-- S=0 so j should be O
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
        return counts(H=10.0, C=9.9, N=0.0, O=10.0, S=0.0)
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
        return counts(H=10.0, C=9.7, O=10.0, N=0.0, S=0.0)
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
        return counts(H=5.0, C=0.0, N=0.0, O=0.0, S=0.0)
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
        return counts(H=0.0, C=2.0, N=0.0, O=0.0, S=0.0)
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

# --------------------------------------------------------------------------
# helium
# --------------------------------------------------------------------------

def test_he_is_heavy_major_in_an_h_he_envelope(monkeypatch, base_geo):
    """H/He + trace O: j is He, not O."""
    p = Parameters()
    RXUV, m_p, T = base_geo
    monkeypatch.setattr(FractionationPhysics, "atomic_counts_from_X",
                        staticmethod(lambda _p: counts(H=10.0, He=0.85, O=0.01)))

    i, j, f = FractionationPhysics.choose_light_and_heavy_major(p, RXUV, T, m_p)
    assert i == "H"
    assert j == "He"
    assert math.isclose(f["He"], 0.085)

def test_he_becomes_light_major_once_h_is_gone(monkeypatch, base_geo):
    p = Parameters()
    RXUV, m_p, T = base_geo
    monkeypatch.setattr(FractionationPhysics, "atomic_counts_from_X",
                        staticmethod(lambda _p: counts(He=5.0, C=1.0, O=2.0)))

    i, j, f = FractionationPhysics.choose_light_and_heavy_major(p, RXUV, T, m_p)
    assert i == "He"
    assert j == "O"
    assert f["He"] == 1.0 and f["H"] == 0.0

def test_forced_light_major_accepts_he_in_any_case(monkeypatch, base_geo):
    """Config upper-cases forced_light_major, so 'HE' has to work."""
    p = Parameters()
    RXUV, m_p, T = base_geo
    monkeypatch.setattr(FractionationPhysics, "atomic_counts_from_X",
                        staticmethod(lambda _p: counts(He=5.0, O=2.0)))

    for name in ("He", "HE", "he"):
        i, j, _ = FractionationPhysics.choose_light_and_heavy_major(
            p, RXUV, T, m_p, allow_dynamic_light_major=False, forced_light_major=name
        )
        assert (i, j) == ("He", "O")
