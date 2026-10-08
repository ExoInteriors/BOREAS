from __future__ import annotations
from importlib.resources import files
from pathlib import Path
from .parameters import ModelParams
from .spectrum import XUVSpectrum, E_H_EDGE_EV
from typing import Dict, Any
from boreas.data import load_planet_params

import math

# TOML loader: stdlib for 3.11+, fallback to 'tomli' for 3.9–3.10
try:
    import tomllib as _toml  # py>=3.11
except ModuleNotFoundError:
    import tomli as _toml     # py 3.9–3.10

def load_config_toml(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        cfg = _toml.load(f)
    # remembered so that relative file paths in the config (e.g. [xuv].spectrum_file) resolve from its folder
    cfg["_config_dir"] = str(Path(path).resolve().parent)
    return cfg

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
    # FXUV: number or "from_data"; with [xuv].spectrum_file, "from_spectrum" or a number
    # to normalise the spectrum to
    FXUV_val = planet.get("FXUV_erg_cm2_s", "from_data")
    spectrum = _load_spectrum(cfg)
    if spectrum is not None:
        if isinstance(FXUV_val, str):
            if FXUV_val.lower() != "from_spectrum":
                raise ValueError(f'[xuv].spectrum_file is set, so [planet].FXUV_erg_cm2_s must be "from_spectrum" '
                                 f'(use the spectrum as given) or a number (normalise the spectrum to it), '
                                 f'not "{FXUV_val}".')
            params.set_xuv_spectrum(spectrum)
        else:
            params.set_xuv_spectrum(spectrum, normalize_to=float(FXUV_val))
    else:
        if isinstance(FXUV_val, str) and FXUV_val.lower() == "from_spectrum":
            raise ValueError('[planet].FXUV_erg_cm2_s = "from_spectrum" needs [xuv].spectrum_file.')
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
    if "use_homopause" in phys:
        params.use_homopause = bool(phys["use_homopause"])
    if "Kzz_cm2_s" in phys:
        params.Kzz = float(phys["Kzz_cm2_s"])
    # (alpha_rec left as default unless you *really* want to expose it)

    # --- XUV cross-sections (atomic, cm^2) ---
    params.chi_from_spectrum = bool(cfg.get("xuv", {}).get("chi_from_spectrum", False))
    if params.chi_from_spectrum and spectrum is None:
        raise ValueError("[xuv].chi_from_spectrum = true needs [xuv].spectrum_file.")
    sig = cfg.get("xuv", {}).get("sigma_cm2", {})
    if sig:
        params.set_sigma_XUV(sig)

    # --- IR opacities κ (cm^2 g^-1) ---
    kap = cfg.get("infrared", {}).get("kappa_cm2_g", {})
    if kap:
        params.set_kappa(kap)

    # --- diffusion fits: b_ij(T) = A T^gamma ---
    # The He pairs come as a set ("literature" or "chapman-enskog"); [diffusion.b] is
    # applied after it, so a single pair listed there still overrides the set.
    he_set = cfg.get("diffusion", {}).get("he_set")
    if he_set:
        params.use_he_diffusion_set(str(he_set))
    # Accept "HO", "H-O", "HHe" or "He-O" keys, in either order
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
                tol=tol, max_iter=max_iter,
                rl_policy=_rl_policy(cfg))

def mass_loss_runtime_args(cfg: Dict[str, Any]):
    """Return kwargs for MassLoss.compute_mass_loss_parameters from config."""
    return dict(rl_policy=_rl_policy(cfg))

def _rl_policy(cfg: Dict[str, Any]) -> str:
    """[advanced].rl_policy: "auto" (switch to RL when recombination-limited, default) or "never" (always EL)."""
    policy = str(cfg.get("advanced", {}).get("rl_policy", "auto")).lower()
    if policy not in ("auto", "never"):
        raise ValueError(f'Unknown [advanced].rl_policy "{policy}". Valid: "auto", "never"')
    return policy

def _load_spectrum(cfg: Dict[str, Any]):
    """XUVSpectrum from [xuv].spectrum_file, or None. A relative path is taken from the config file's folder."""
    xuv = cfg.get("xuv", {})
    path = xuv.get("spectrum_file")
    if not path:
        return None
    path = Path(path).expanduser()
    if not path.is_absolute() and "_config_dir" in cfg:
        path = Path(cfg["_config_dir"]) / path
    return XUVSpectrum.from_file(
        path,
        x_unit=xuv.get("spectrum_x_unit", "angstrom"),
        scale=float(xuv.get("spectrum_scale", 1.0)),
        E_min_eV=float(xuv.get("spectrum_E_min_eV", E_H_EDGE_EV)),
        E_max_eV=xuv.get("spectrum_E_max_eV"),
    )

# Utility: load a field from packaged planet data
def _load_planet_field(name: str, field: str):
    planets = load_planet_params()
    if name not in planets:
        raise KeyError(f"Planet '{name}' not found in packaged data.")
    if field not in planets[name]:
        raise KeyError(f"Field '{field}' not present for planet '{name}'.")
    return planets[name][field]