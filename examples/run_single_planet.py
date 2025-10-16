from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np

from boreas.config import load_config_toml
from boreas import ModelParams, MassLoss, Fractionation
from boreas.config import apply_params_from_config, build_inputs_from_config, fractionation_runtime_args

HERE = Path(__file__).resolve().parent
DEFAULT_CFG = HERE / "configs" / "my_planet.toml" # <- default lives next to this script

def parse_args():
    ap = argparse.ArgumentParser(description="Run BOREAS for a single TOML config.")
    ap.add_argument("-c", "--config", default=None, help="Path to a TOML file.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print extra info.")
    return ap.parse_args()

def resolve_cfg(arg: str | None) -> Path:
    if arg is None:
        return DEFAULT_CFG
    p = Path(arg)
    if p.suffix.lower() != ".toml":
        p = p.with_suffix(".toml")
    if p.exists():
        return p.resolve()
    candidate = HERE / "configs" / p.name
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Config not found: {arg}\nTried: {p} and {candidate}")

def main(cfg_path: Path, verbose: bool = False):
    if verbose:
        print(f"[runner] reading config: {cfg_path}")
    with cfg_path.open("rb") as f:
        cfg = load_config_toml(cfg_path)

    if verbose:
        print("[runner] init params/mass-loss/fractionation")
    params = ModelParams()
    fx_args = apply_params_from_config(cfg, params)
    mass_loss = MassLoss(params)
    fractionation = Fractionation(params)

    # --- bulk inputs from config (mass[g], radius[cm], teq[K]) ---
    mass, radius, teq = build_inputs_from_config(cfg, params)
    if verbose:
        print(f"[runner] inputs: M={mass[0]:.3e} g, R={radius[0]:.3e} cm, Teq={teq[0]:.1f} K, FXUV={params.FXUV}")

    # --- run ---
    ml_results  = mass_loss.compute_mass_loss_parameters(mass, radius, teq)
    frac_kwargs = fractionation_runtime_args(cfg)
    f_results   = fractionation.execute(ml_results, mass_loss, **frac_kwargs)

    # --- report ---
    r0 = f_results[0]
    print(f"\n Config: {cfg_path}")
    print("Planet:", fx_args["planet_name"])
    print("Regime:", r0.get("regime"),
          "RXUV[cm]:", r0.get("RXUV"),
          "Mdot[g/s]:", r0.get("Mdot"))
    print("light_major:", r0.get("light_major_i"),
          "heavy_major:", r0.get("heavy_major_j"))
    print("T_outflow[K]:", r0.get("T_outflow"),
          "mu_outflow:", r0.get("mmw_outflow"))

if __name__ == "__main__":
    args = parse_args()
    try:
        cfg_path = resolve_cfg(args.config)
        main(cfg_path, verbose=args.verbose)
    except Exception as e:
        import traceback
        print("[runner] ERROR:", e)
        traceback.print_exc()
        raise