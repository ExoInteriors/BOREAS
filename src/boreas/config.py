from __future__ import annotations
from importlib.resources import files
from .parameters import ModelParams
from typing import Dict, Any
from boreas.data import load_planet_params

import math

# TOML loader: stdlib for 3.11+, fallback to 'tomli' for 3.9–3.10
try:
    import tomllib as _toml  # py>=3.11
except ModuleNotFoundError:
    import tomli as _toml     # py 3.9–3.10

_COMPOSITION_KEYS = ["H2","H2O","O2","CO2","CO","CH4","N2","NH3","H2S","SO2","S2"]
_XATTR = {name: f"X_{name}" for name in _COMPOSITION_KEYS}

def load_config_toml(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return _toml.load(f)

def _load_builtin_planets() -> Dict[str, Any]:
    return load_planet_params()
    # data_path = files("boreas.data") / "planet_params.json"
    # import json
    # return json.loads(data_path.read_text(encoding="utf-8"))

def build_inputs_from_config(cfg: Dict[str, Any], params: ModelParams):
    """
    Return (mass[g], radius[cm], teq[K]) numpy arrays from config.
    NOTE: FXUV is set in apply_params_from_config(); this function does NOT touch FXUV.
    """
    import numpy as np
    mearth, rearth = params.mearth, params.rearth

    psec = cfg.get("planet", {})
    planets = _load_builtin_planets()

    if "name" in psec:
        name = psec["name"]
        if name not in planets:
            raise KeyError(f"Planet '{name}' not found in packaged data.")
        rec = planets[name]
        mass_g   = float(rec["mass"])   * mearth
        radius_c = float(rec["radius"]) * rearth
        teq_k    = float(rec["teq"])
    else:
        # explicit values
        try:
            mass_g   = float(psec["mass_mearth"])   * mearth
            radius_c = float(psec["radius_rearth"]) * rearth
            teq_k    = float(psec["teq_K"])
        except KeyError as e:
            raise KeyError(f"Missing required planet key: {e}")

    # Do NOT set FXUV here; apply_params_from_config already did it.
    return np.array([mass_g]), np.array([radius_c]), np.array([teq_k])

def apply_params_from_config(cfg: Dict[str, Any], params: ModelParams):
    """Apply user-facing config knobs to ModelParams only (no hydro/fractionation here)."""
    # --- composition ---
    comp = cfg.get("composition", {})
    if not comp:
        raise ValueError("Config must include a [composition] section.")
    auto_norm = bool(cfg.get("advanced", {}).get("auto_normalize_X", True))
    params.enable_auto_normalize(cfg.get("advanced", {}).get("auto_normalize_X", False))
    params.set_composition(cfg["composition"])

    # --- planet block & FXUV ---
    planet = cfg.get("planet", {})
    pname  = planet.get("name")
    if not pname:
        raise ValueError("Config must include [planet].name")
    # FXUV: number or "from_data"
    FXUV_val = planet.get("FXUV_erg_cm2_s", "from_data")
    if isinstance(FXUV_val, str) and FXUV_val.lower() == "from_data":
        FXUV_val = float(_load_planet_field(pname, "FXUV"))
    else:
        FXUV_val = float(FXUV_val)
    params.update_param("FXUV", FXUV_val)

    # --- physics (optional) ---
    phys = cfg.get("physics", {})
    if "efficiency" in phys:
        params.eff = float(phys["efficiency"])
    if "albedo" in phys:
        params.albedo = float(phys["albedo"])
    if "beta" in phys:
        params.beta = float(phys["beta"])
    if "emissivity" in phys:
        params.epsilon = float(phys["emissivity"])
    # (alpha_rec left as default unless you *really* want to expose it)

    # --- XUV cross-sections (atomic, cm^2) ---
    sig = cfg.get("xuv", {}).get("sigma_cm2", {})
    if sig:
        params.set_sigma_XUV(sig)

    # --- IR opacities κ (cm^2 g^-1) ---
    kap = cfg.get("infrared", {}).get("kappa_cm2_g", {})
    if kap:
        params.set_kappa(kap)

    # --- diffusion fits: b_ij(T) = A T^gamma ---
    # Accept "HO" or "H-O" keys
    diff = cfg.get("diffusion", {}).get("b", {})
    if diff:
        params.set_diffusion_fits(diff)

    # --- fractionation controller knobs (optional, you use them when calling execute) ---
    frac = cfg.get("fractionation", {})
    # return these so the caller can pass them into Fractionation.execute(...)
    return {
        "allow_dynamic_light_major": bool(frac.get("allow_dynamic_light_major", True)),
        "forced_light_major": str(frac.get("forced_light_major", "H")).upper(),
        "tol": float(frac.get("tol", 1e-5)),
        "max_iter": int(frac.get("max_iter", 100)),
        "planet_name": pname,
    }

def fractionation_runtime_args(cfg: Dict[str, Any]):
    """Return kwargs for Fractionation.execute from config."""
    frac = cfg.get("fractionation", {})
    allow_dyn = bool(frac.get("allow_dynamic_light_major", True))
    forced    = str(frac.get("forced_light_major", "H")).upper()
    tol       = float(frac.get("tol", 1e-5))
    max_iter  = int(frac.get("max_iter", 100))
    return dict(allow_dynamic_light_major=allow_dyn,
                forced_light_major=forced,
                tol=tol, max_iter=max_iter)

# Utility: load a field from packaged planet data
def _load_planet_field(name: str, field: str):
    planets = load_planet_params()
    if name not in planets:
        raise KeyError(f"Planet '{name}' not found in packaged data.")
    if field not in planets[name]:
        raise KeyError(f"Field '{field}' not present for planet '{name}'.")
    return planets[name][field]