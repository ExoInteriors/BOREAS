from importlib.resources import files
import json

__all__ = ["planet_params_path", "load_planet_params"]

def planet_params_path():
    # Returns a pathlib.Path to the JSON file inside the installed package
    return files(__package__) / "planet_params.json"

def load_planet_params(encoding: str = "utf-8") -> dict:
    return json.loads(planet_params_path().read_text(encoding=encoding))
