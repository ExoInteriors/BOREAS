from Parameters import ModelParams
from Functions.Model_Data_Loader import ModelDataLoader
from Functions.Star_Parameters_Class import StarParams
from Functions.Mass_Loss_Class import MassLoss
from Functions.Fractionation_Class import Fractionation
from Functions.Misc_Class import Misc

import numpy as np
import pandas as pd
import json

# ---------- Initialize parameters and classes ----------
params          = ModelParams()
mearth          = params.mearth             # mass of earth, in grams
rearth          = params.rearth             # radius of earth, in cm
params.update_param('outflow_mode','HHe_H2O')   # set desired outflow mode: 'HHe', 'H2O', or 'HHe_H2O'
params.update_param('X_H2O', 0.1)           # only relevant for 'HHe_H2O' mode
mass_loss       = MassLoss(params)
fractionation   = Fractionation(params)
star_params     = StarParams(params)
misc            = Misc(params)
loader          = ModelDataLoader('/path/to/MR_perplex/OUTPUT/', params)

# ---------- Load data ----------
use_json_planet = True # if True: select planet from .json file; if False: use MRcode .ddat file or other format of M-R-Teq arrays

if use_json_planet: # -------- option 1: JSON planet file --------
    with open("planet_params.json", "r") as file:
        planet_params = json.load(file)
    planet = "K2-18 b"                      # <------------ change the planet name here (see available planets in planet_params.json)

    if planet not in planet_params:
        raise ValueError(f"Planet '{planet}' not found in planet_params.json. Check spelling.")
    else:
        params.update_param('FEUV', planet_params[planet]['FEUV'])  # update the FEUV value

    mass = np.array([planet_params[planet]["mass"] * mearth])       # g
    radius = np.array([planet_params[planet]["radius"] * rearth])   # cm
    Teq = np.array([planet_params[planet]["teq"]])                  # K

else: # -------- option 2: mass, radius, and temperature arrays in grams, cm, K ---------
    mass, radius, Teq = loader.load_single_ddat_file('3HHe_90H2O_subNeptune.ddat') # grams, cm, K
    # OR mass, radius, Teq = any_method_available_to_you, to get array(s)

# ---------- Find unique temperatures and XUV fluxes ----------
unique_Teqs = np.unique(Teq)

all_flux_results = []

for unique_Teq in unique_Teqs:
    star_params.update_param("Teq", unique_Teq)
    Fbol = star_params.get_Fbol_from_Teq()

    if use_json_planet:
        flux_range = [params.get_param('FEUV')]
    else:
        flux_range = star_params.get_FEUV_range_any_age() # get flux range from Teq

    mask = np.isclose(Teq, unique_Teq)
    mass_group = mass[mask]
    radius_group = radius[mask]
    teq_group = Teq[mask]
    
    print(f"\n Processing group with Teq = {unique_Teq} K")

    # ---------- Execute ----------
    for flux_value in flux_range:
        print(f"\n Running model with FEUV = {flux_value} erg/cm^2/s")
        params.update_param('FEUV', flux_value)

        mass_loss_results = mass_loss.compute_mass_loss_parameters(mass_group, radius_group, teq_group)                         # Mass loss model
        fractionation_results = fractionation.execute_self_consistent_fractionation(mass_loss_results, mass_loss, misc, params) # Fractionation model (skipped if mode='HHe')
        
        ### To directly print the total mass loss rate after running the model
        mdot = fractionation_results[0]['Mdot']
        print("Mdot =", mdot)

        all_flux_results.append({
            'FEUV': flux_value,
            'Teq' : unique_Teq,
            'Fbol': Fbol,
            'results': fractionation_results
        })

    # Optionally: save data for plotting
    rows = []

    for entry in all_flux_results:
        feuv = entry['FEUV']
        teq_val = entry['Teq']
        fbol = entry['Fbol']
        for sol in entry['results']:
            row = {'FEUV': feuv, 'Teq': teq_val, 'Fbol': fbol}
            row.update(sol)
            rows.append(row)

    df = pd.DataFrame(rows)
    # output_path = '/path/to/your/file.csv'
    # df.to_csv(output_path, index=False)