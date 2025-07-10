Photoevaporative Mass Loss & Chemical Fractionation Model

This repository implements a coupled photoevaporative mass loss and chemical fractionation model for the atmospheres of super-Earths and sub-Neptunes.It supports three outflow modes:

HHe: Pure H/He (no fractionation)
H2O: Pure steam (H₂O)
HHe_H2O: Mixed H2 + H2O outflow with self‑consistent fractionation



Features

1. Mass Loss Module (Mass_Loss_Class.py):
- Computes XUV‑driven mass loss rate in both energy‑limited (EL) and recombination‑limited (RL) regimes.
- Handles dissociated vs. non‑dissociated outflow compositions (modeled by outflow_mode).
- Οutputs R_XUV, sound speed, mass‑loss rate, density profiles, timescale diagnostics, and more.

2. Fractionation Module (Fractionation_Class.py): 
- Implements the Zahnle & Kasting (1986) H/O fractionation formalism.
- In self‑consistent mode, iteratively updates the outflow mean molecular weight based on the computed H/O fluxes and re‑runs the hydrodynamic solver until convergence.
- Outputs partial mass fluxes, oxygen fractionation factor, updated mean molecular weight of the outflow, outflow temperature, and pressure.

3. Data Loader (Model_Data_Loader.py): 
- Reads precomputed planetary data files from the MR code (Dorn group) (*.ddat).

4. Star Parameters (Star_Parameters_Class.py): 
- Converts equilibrium temperature to received bolometric flux.
- Provides stellar XUV flux evolution and range for a given  and stellar age.

5. Miscellaneous Utilities (Misc_Class.py):
- Contains helper functions

Installation

- git clone https://github.com/ExoInteriors/Mass-Loss.git
- cd Mass-Loss
- python -m venv venv
- source venv/bin/activate
- pip install (any requirements, like numpy)

Configuration & Key Parameters

All model parameters are defined in Parameters.py via the ModelParams class. Key fields:

- outflow_mode: one of 'HHe', 'H2O', 'HHe_H2O' (default 'H2O').
- FEUV: incident XUV flux (erg cm⁻² s⁻¹).
- X_H2O, X_HHe: mass fractions of water and H/He in the mixed mode.
- kappa_p_HHe, kappa_p_H2O, kappa_p_HHe_H2O: infrared opacities for each composition.

You can override defaults either by editing ModelParams.__init__ or at runtime, e.g.,:
params = ModelParams()
params.update_param('outflow_mode', 'HHe_H2O')
params.update_param('FEUV', 1e3)
params.update_param('X_H2O', 0.2)

Running the Model

Use Main.py as the driver:
1.	Set outflow mode at top:
 	params.update_param('outflow_mode', 'HHe_H2O')
2.	Load planetary data:
    - Bulk .ddat file via ModelDataLoader.load_single_ddat_file(...) or
    - Single planet from planet_params.json.
3.	Loop over unique (T_{})** values**:
 	for Teq in unique_Teqs:
    star_params.update_param('Teq', Teq)
    flux_range = star_params.get_FEUV_range_any_age()
    for FEUV in flux_range:
        params.update_param('FEUV', FEUV)
        mass_loss_results = mass_loss.compute_mass_loss_parameters(...)
        fractionation_results = fractionation.execute_self_consistent_fractionation(...)
        ...
4.	Save outputs to CSV:
 	import pandas as pd
    rows = []
    for entry in all_flux_results:
        for sol in entry['results']:
            record = {'FEUV': entry['FEUV'], ...}
            record.update(sol)
            rows.append(record)
    df = pd.DataFrame(rows)
    df.to_csv('results.csv', index=False)

Outputs
- Mass loss: REUV, cs, dot M, rho_EUV, rho_flow, regime (EL or RL), etc.
- Fractionation: phi_H, phi_O, x_O, mmw_outflow, T_outflow, P_EUV.
You can easily load into pandas via pd.read_csv() for analysis and plotting.

References

Model for Mass Loss from:
Owen, J. E., & Schlichting, H. E. (2023). Mapping out the parameter space for photoevaporation and core-powered mass-loss (arXiv:2308.00020). arXiv. https://doi.org/10.48550/arXiv.2308.00020

Model for Fractionation from:
Zahnle, K. J., & Kasting, J. F. (1986). Mass fractionation during transonic escape and implications for loss of water from Mars and Venus. Icarus, 68(3), 462–480. https://doi.org/10.1016/0019-1035(86)90051-5

Model for F_XUV vs Teq from:
Rogers, J. G., Gupta, A., Owen, J. E., & Schlichting, H. E. (2021). Photoevaporation vs. core-powered mass-loss: Model comparison with the 3D radius gap. Monthly Notices of the Royal Astronomical Society, 508(4), 5886–5902. https://doi.org/10.1093/mnras/stab2897
