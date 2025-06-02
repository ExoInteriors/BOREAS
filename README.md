Photoevaporative Mass Loss & Chemical Fractionation Model

This repository implements a coupled photoevaporative mass loss and chemical fractionation model for the atmospheres of super-Earths and sub-Neptunes. The code supports pure H₂O, pure H/He, and mixed H/He–H₂O compositions.

Features

Mass Loss Module (Mass_Loss_Class.py): Computes the XUV-driven mass loss rate (Mdot) in energy-limited (EL) and radiation-limited (RL) regimes, with options for dissociated vs. non‑dissociated compositions. Also outputs RXUV, sound speed, and more.

Fractionation Module (Fractionation_Class.py): Calculates the chemical fractionation of hydrogen and oxygen using Zahnle & Kasting (1986) formalisms, optionally in a self-consistent feedback loop that updates the outflow mean molecular weight. Outputs the escape flux rates phi of oxygen and hydrogen, the fractionation factor of oxygen, and more.

Data Loader (Model_Data_Loader.py): Reads precomputed planetary data files from the MR code (Dorn group) (*.ddat) or JSON parameter lists.

Star Parameters (Star_Parameters.py): Converts equilibrium temperature (Teq) to bolometric flux (Fbol) and XUV flux evolution.

Miscellaneous Utilities (Misc_Class.py): Helper functions, e.g., ideal-gas pressure and Bondi radius calculations.

Installation

Clone the repository:

git clone https://github.com/ExoInteriors/Mass-Loss.git
cd Mass-Loss

Create and activate a Python environment, and install dependencies (numpy, pandas etc)


Configuration & Key Parameters

Most user-configurable parameters live in Parameters.py via the ModelParams class. Important fields:

FEUV (erg cm⁻² s⁻¹): XUV flux received by the planet. Updated at runtime based on JSON, or input a fixed value, or calculate a value/range using Star_Parameters.py.

X_HHe, X_H2O: Mass fractions of the H/He and H₂O components in the atmospheric mixture (e.g., 0.8 and 0.2).

kappa_p_HHe, kappa_p_H2O, kappa_p_HHe_H2O: Infrared opacities (mean opacities in the IR) for pure and mixed compositions. Adjusting these tunes the photospheric recombination depth.

To override defaults, edit ModelParams.__init__ or call:

params.update_param('FEUV', 1e3)
params.update_param('X_H2O', 0.1)
params.update_param('kappa_p_HHe_H2O', 0.5)

Running the Model

The main driver script is Main.py:

Select a planet in planet_params.json by setting:

selected_planet = "TRAPPIST-1 h"

Load planetary properties (mass, radius, Teq, FEUV) from JSON.

Iterate over unique Teq values, compute Fbol, and run:

MassLoss.compute_mass_loss_parameters(...) (mass-loss only).

Fractionation.execute_self_consistent_fractionation(...) (coupled fractionation + mass loss).

Collect results in all_flux_results for further analysis or export.

Model Coupling

Mass Loss calculates the XUV penetration radius (REUV), sound speed (cs), mass-loss rate (Mdot) under the chosen heating regime (EL or RL), and more parameters.

Fractionation takes Mdot, REUV, and cs to compute the outflow temperature (T_outflow), diffusion-limited oxygen loss, and the H/O escape fluxes (φ_H, φ_O).

In self-consistent mode, the fractionation code updates the outflow mean molecular weight (mmw_outflow) based on the H/O flux ratio and re-runs the mass-loss solver until convergence on Mdot and cs.

The coupling ensures that changes to composition (via fractionation) feed back on the hydrodynamic escape solution.

Outputs

Results for each run are stored in Python structures:

all_flux_results: List of dicts with keys:

FEUV, Teq, Fbol

end_results: list of per-planet result dicts containing:

m_planet, r_planet, REUV, cs, Mdot, rho_EUV, rho_flow, regime, etc.

Fractionation outputs: T_outflow, P_EUV, R_b, phi_O, phi_H, x_O, mmw_outflow (if self-consistent).

You can easily export to CSV/JSON using pandas:

import pandas as pd
df = pd.json_normalize(all_flux_results, record_path=['end_results'], meta=['FEUV','Teq','Fbol'])
df.to_csv('results.csv', index=False)

References

Model for Mass Loss from:
Owen, J. E., & Schlichting, H. E. (2023). Mapping out the parameter space for photoevaporation and core-powered mass-loss (arXiv:2308.00020). arXiv. https://doi.org/10.48550/arXiv.2308.00020

Model for Fractionation from:
Zahnle, K. J., & Kasting, J. F. (1986). Mass fractionation during transonic escape and implications for loss of water from Mars and Venus. Icarus, 68(3), 462–480. https://doi.org/10.1016/0019-1035(86)90051-5

Model for F_XUV vs Teq from:
Rogers, J. G., Gupta, A., Owen, J. E., & Schlichting, H. E. (2021). Photoevaporation vs. core-powered mass-loss: Model comparison with the 3D radius gap. Monthly Notices of the Royal Astronomical Society, 508(4), 5886–5902. https://doi.org/10.1093/mnras/stab2897