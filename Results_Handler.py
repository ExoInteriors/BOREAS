import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from Parameters import ModelParams

class ResultsHandler:

    @staticmethod
    def plot_regime_scatter(all_flux_results):
        """Scatter plot showing which regime each planet is in. X-axis: FEUV, Y-axis: planet mass."""
        params = ModelParams()
        mearth = params.mearth

        x_vals = [] # will store flux
        y_vals = [] # will store planet mass (in Earth masses)
        colors = [] # store color depending on regime

        for flux_data in all_flux_results:
            flux = flux_data['FEUV']
            results_list = flux_data['end_results']
            for planet_res in results_list:
                # Retrieve the regime if it exists; skip if not present
                regime = planet_res.get('regime')
                if regime is None:
                    continue

                mass_earth = planet_res['m_planet'] / mearth

                x_vals.append(flux)
                y_vals.append(mass_earth)

                # colors
                if regime == 'RL':
                    colors.append('red')
                else:
                    colors.append('blue')

        plt.figure(figsize=(3, 3))
        plt.scatter(x_vals, y_vals, c=colors, alpha=0.8)

        plt.xlabel("EUV Flux (erg cm$^{-2}$ s$^{-1}$)")
        plt.ylabel("Planet Mass (Earth Masses)")
        plt.yscale('linear')
        plt.xscale('log')

        legend_elems = [
            Line2D([0], [0], marker='o', color='w', label='EL', markerfacecolor='blue', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='RL', markerfacecolor='red', markersize=8)
        ]
        plt.legend(handles=legend_elems, loc='best')

        plt.tight_layout()
        plt.show()




    @staticmethod
    def plot_m_planet_r_planet(results):
        """Plot planetary mass vs. radius."""
        params = ModelParams()
        mearth = params.mearth
        rearth = params.rearth
        m_planet = [res['m_planet']/mearth for res in results]
        r_planet = [res['r_planet']/rearth for res in results]

        plt.figure()
        plt.plot(m_planet, r_planet, 'o-', label="Radius vs Mass")
        plt.xlabel("Planet Mass (Earth Masses)")
        plt.ylabel("Planet Radius (cm)")
        plt.show()

    @staticmethod
    def plot_cs_REUV(results):
        """Plot sound speed vs REUV and REUV/r_planet."""
        params = ModelParams()
        mearth = params.mearth
        
        m_planet = [res['m_planet'] / mearth for res in results]
        cs = [res['cs'] for res in results]
        REUV = [res['REUV'] for res in results]
        REUV_r = [res['REUV'] / res['r_planet'] for res in results]

        # fig, ax1 = plt.subplots(figsize=(7, 5))
        fig, ax1 = plt.subplots()

        ax1.set_xlabel("$c_{s}$ (cm/s)")
        ax1.set_ylabel("$R_{EUV}$ (cm)", color="black")
        ax1.plot(cs, REUV, '.-', color="black")
        ax1.tick_params(axis='y', labelcolor="black")
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        ax2 = ax1.twinx()
        ax2.set_ylabel("$R_{EUV}/R_{planet}$", color="orangered")
        ax2.plot(cs, REUV_r, '.-', color="orangered")
        ax2.tick_params(axis='y', labelcolor="orangered")

        skip_masses = {10, 12, 14}
        for i, mass in enumerate(m_planet):
            if round(mass) not in skip_masses:
                ax1.annotate(f"{mass:.0f}M⊕", (cs[i], REUV[i]), textcoords="offset points", xytext=(-13,5), ha='left', fontsize=6, color='black')

        skip_masses2 = {6, 8, 10, 11, 13, 14}
        for i, mass in enumerate(m_planet):
            if round(mass) not in skip_masses2:
                ax2.annotate(f"{mass:.0f}M⊕", (cs[i], REUV_r[i]), textcoords="offset points", xytext=(-7,-12), ha='left', fontsize=6, color='black')

        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_REUV_r_planet_vs_m_planet(results):
        """Double y-axis: REUV and r_planet vs m_planet."""
        params = ModelParams()
        mearth = params.mearth
        REUV = [res['REUV'] for res in results]
        r_planet = [res['r_planet'] for res in results]
        m_planet = [res['m_planet']/mearth for res in results]

        plt.figure()
        plt.plot(m_planet, REUV, 'o-', color='blue', label="REUV")
        plt.plot(m_planet, r_planet, 'o-', color='red', label="r_planet")
        plt.xlabel("Planet Mass (Earth Masses)")
        plt.ylabel("Radius (cm)")
        plt.yscale('log')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_phiH_phiO_vs_REUV(results):
        """Phi_H and Phi_O on the same axis."""
        params = ModelParams()
        mearth = params.mearth
        
        m_planet = [res['m_planet'] / mearth for res in results]
        REUV = [res['REUV'] for res in results]
        phi_H = [res['phi_H'] for res in results]
        phi_O = [res['phi_O'] for res in results]

        plt.figure(figsize=(8, 5))
        plt.plot(REUV, phi_H, '.-', color='darkorchid', label="Hydrogen")
        plt.plot(REUV, phi_O, '.-', color='gold', label="Oxygen")
        plt.xlabel("$R_{EUV}$ (cm)")
        plt.ylabel("Escape fluxes (g/cm$^{2}\cdot$ s)")
        plt.yscale('log')
        plt.xscale('log')

        skip_masses = {0, 4, 6, 8, 9, 11, 12, 13, 14}
        for i, mass in enumerate(m_planet):
            if round(mass) not in skip_masses:
                plt.annotate(f"{mass:.0f}M⊕", (REUV[i], phi_H[i]), textcoords="offset points", xytext=(-5,5), ha='left', fontsize=10, color='black')
                plt.annotate(f"{mass:.0f}M⊕", (REUV[i], phi_O[i]), textcoords="offset points", xytext=(-5,5), ha='left', fontsize=10, color='black')

        plt.legend()
        plt.show()

    @staticmethod
    def plot_phi_vs_m_planet_across_fluxes(results):
        "phi_H and phi_O vs. planet mass for multiple fluxes."
        params = ModelParams()
        mearth = params.mearth

        flux_values = [fd['FEUV'] for fd in results]
        flux_min = min(flux_values)
        flux_max = max(flux_values)

        cmap = get_cmap("inferno_r")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

        for flux_data in results:
            flux = flux_data['FEUV']
            results_list = flux_data['end_results']

            # Map flux to a color fraction between 0 and 1
            if flux_max > flux_min:
                color_fraction = (flux - flux_min) / (flux_max - flux_min)
            else:
                color_fraction = 0.5

            color = cmap(color_fraction)

            m_planet = []
            phi_H = []
            phi_O = []

            for result in results_list:
                if 'phi_H' in result and 'phi_O' in result:
                    m_planet.append(result['m_planet'] / mearth)
                    phi_H.append(result['phi_H'])
                    phi_O.append(result['phi_O'])

            ax1.plot(m_planet, phi_H, '.-', label=flux, color=color)
            ax2.plot(m_planet, phi_O, '.-', label=flux, color=color)

        # --- Top subplot (phi_H) ---
        ax1.set_ylabel(r"$\phi_H$ (g cm$^{-2}$ s$^{-1}$)")
        ax1.set_yscale('log')
        ax1.legend(title='EUV flux (ergs/cm2/s)', loc='center left', bbox_to_anchor=(1, 0.5))

        # --- Bottom subplot (phi_O) ---
        ax2.set_xlabel("Planet Mass (Earth Masses)")
        ax2.set_ylabel(r"$\phi_O$ (g cm$^{-2}$ s$^{-1}$)")
        ax2.set_yscale('log')
        ax2.legend(title='EUV flux (ergs/cm2/s)', loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_phi_vs_flux_across_masses(all_flux_results):
        """X-axes: phi_H and phi_O, Y-axis: EUV flux (FEUV), with a separate line for each planet mass."""
        params = ModelParams()
        mearth = params.mearth

        # Dictionary to gather data by planet mass
        # mass_map[mass_in_earths] = {
        #    'flux': [flux1, flux2, ...],
        #    'phi_H': [phi_H1, phi_H2, ...],
        #    'phi_O': [phi_O1, phi_O2, ...]
        # }
        mass_map = {}

        for flux_data in all_flux_results:
            flux = flux_data['FEUV']
            results_list = flux_data['end_results']

            for planet_res in results_list:
                if 'phi_H' not in planet_res or 'phi_O' not in planet_res:
                    continue
                
                # 1) Convert raw planet mass to Earth-mass float
                raw_mass = float(planet_res['m_planet'] / mearth)

                # 2) Round or otherwise convert to remove tiny float differences
                m_key = round(raw_mass, 4)

                # 3) Ensure the dictionary entry exists
                if m_key not in mass_map:
                    mass_map[m_key] = {
                        'flux': [],
                        'phi_H': [],
                        'phi_O': []
                    }

                # 4) Safely append
                mass_map[m_key]['flux'].append(flux)
                mass_map[m_key]['phi_H'].append(planet_res['phi_H'])
                mass_map[m_key]['phi_O'].append(planet_res['phi_O'])
                
        # -- Sort each mass's data by ascending flux for nice plotting --
        for m_key, data in mass_map.items():
            combined = sorted(
                zip(data['flux'], data['phi_H'], data['phi_O']),
                key=lambda x: x[0] # sort by flux
            )
            data['flux']  = [c[0] for c in combined]
            data['phi_H'] = [c[1] for c in combined]
            data['phi_O'] = [c[2] for c in combined]

        # color lines by planet mass
        all_masses = sorted(mass_map.keys())
        mass_min = min(all_masses)
        mass_max = max(all_masses)

        cmap = get_cmap("copper_r")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        
        for m_key in all_masses:
            data = mass_map[m_key]
            if mass_max > mass_min:
                color_fraction = (m_key - mass_min) / (mass_max - mass_min)
            else:
                color_fraction = 0.5

            color = cmap(color_fraction)

            ax1.plot(data['flux'], data['phi_H'], '.-', label=f"{m_key:.2f} M⊕", color=color)
            ax2.plot(data['flux'], data['phi_O'], '.-', label=f"{m_key:.2f} M⊕", color=color)

        # top subplot (phi_H)
        ax1.set_ylabel(r"$\phi_\mathrm{H}$ [g cm$^{-2}$ s$^{-1}$]")
        ax1.set_yscale('log')
        ax1.set_xscale('log')

        # bottom subplot (phi_O)
        ax2.set_xlabel("EUV Flux (erg cm$^{-2}$ s$^{-1}$)")
        ax2.set_ylabel(r"$\phi_\mathrm{O}$ [g cm$^{-2}$ s$^{-1}$]")
        ax2.set_yscale('log')

        # single legend
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5))
    
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_mmw_vs_masses_across_fluxes(all_flux_results):
        """X-axis: mmw_outflow, Y-axis: planet mass, lines for each FEUV."""
        params = ModelParams()
        mearth = params.mearth

        flux_map = {}

        for flux_data in all_flux_results:
            flux = flux_data['FEUV']
            results_list = flux_data['end_results']

            if flux not in flux_map:
                flux_map[flux] = {
                    'mmw': [],
                    'm_planet': []
                }

            for planet_res in results_list:
                if 'mmw_outflow' not in planet_res:
                    continue

                mmw = planet_res['mmw_outflow']
                mass_earths = planet_res['m_planet'] / mearth

                flux_map[flux]['mmw'].append(mmw)
                flux_map[flux]['m_planet'].append(mass_earths)

        # Sort each flux's data by mmw so lines go left-to-right
        for flux, data in flux_map.items():
            combined = sorted(zip(data['m_planet'], data['mmw']), key=lambda x: x[0])
            data['m_planet'] = [c[0] for c in combined]
            data['mmw']      = [c[1] for c in combined]

        # color lines by flux
        all_fluxes = sorted(flux_map.keys())
        flux_min   = min(all_fluxes)
        flux_max   = max(all_fluxes)

        cmap = get_cmap("Wistia")

        fig, ax = plt.subplots(figsize=(7, 5))

        for flux in all_fluxes:
            data = flux_map[flux]
            if flux_max > flux_min:
                color_fraction = (flux - flux_min) / (flux_max - flux_min)
            else:
                color_fraction = 0.5
            color = cmap(color_fraction)

            ax.plot(data['m_planet'], data['mmw'], '.-', label={flux}, color=color)

        ax.set_ylabel("Mean Molecular Weight of Outflow")
        ax.set_xlabel("Planet Mass (Earth Masses)")
        # ax.set_yscale('log')
        ax.set_xscale('linear')
        ax.legend(loc='best',  title='EUV flux')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_mmw_vs_flux_across_masses(all_flux_results):
        """X-axis: mmw_outflow, Y-axis: FEUV, lines for each planet mass."""

        params = ModelParams()
        mearth = params.mearth

        mass_map = {}

        for flux_data in all_flux_results:
            flux = flux_data['FEUV']
            results_list = flux_data['end_results'] # each is a list of planet dicts

            for planet_res in results_list:
                if 'mmw_outflow' not in planet_res:
                    continue

                m_earth = planet_res['m_planet'] / mearth
                mmw     = planet_res['mmw_outflow']

                m_key = round(float(m_earth), 4) # round to avoid float issues

                if m_key not in mass_map:
                    mass_map[m_key] = {
                        'mmw': [],
                        'flux': []
                    }

                mass_map[m_key]['mmw'].append(mmw)
                mass_map[m_key]['flux'].append(flux)

        # For each planet mass, sort the data by mmw_outflow so lines connect in ascending order of mmw
        for m_key, data in mass_map.items():
            combined = sorted(zip(data['flux'], data['mmw']), key=lambda x: x[0])  
            data['flux']  = [c[0] for c in combined]
            data['mmw'] = [c[1] for c in combined]

        # For coloring lines by planet mass
        all_masses = sorted(mass_map.keys())
        mass_min   = min(all_masses)
        mass_max   = max(all_masses)

        cmap = get_cmap("Wistia")

        fig, ax = plt.subplots(figsize=(7, 5))

        for m_key in all_masses:
            data = mass_map[m_key]
            if mass_max > mass_min:
                color_fraction = (m_key - mass_min) / (mass_max - mass_min)
            else:
                color_fraction = 0.5
            color = cmap(color_fraction)

            ax.plot(data['flux'], data['mmw'], '.-', label=f"{m_key:.2f} $M_\oplus$", color=color)

        ax.set_ylabel("Mean Molecular Weight of Outflow")
        ax.set_xlabel("EUV Flux (erg cm$^{-2}$ s$^{-1}$)")
        # ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(loc='best')

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_combined_T_P(results):
        """Plot temperature at REUV vs pressure with planetary mass labels."""
        params = ModelParams()
        mearth = params.mearth

        m_planet = [res['m_planet'] / mearth for res in results] # Earth masses
        T_outflow = [res['T_outflow'] for res in results] # Temperature in K
        P_EUV_dyn_ideal = [res['P_EUV'] for res in results] # Pressure in dyn/cm^2
        P_EUV_Pa = [p * 0.1 for p in P_EUV_dyn_ideal] # Convert dyn/cm^2 to Pa
        P_EUV_bar = [p * 1e-6 for p in P_EUV_Pa] # Convert Pa to bar

        fig, ax1 = plt.subplots(figsize=(6, 4))

        ax1.plot(T_outflow, P_EUV_Pa, '.-', color='black', linewidth=0.75, label="Pressure (Pa)")
        ax1.set_xlabel("Temperature in outflow region (K)")
        ax1.set_ylabel("Pressure at $R_{EUV}$ (Pa)")
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()
        ax2.plot(T_outflow, P_EUV_bar, color='black', alpha=0, label="Pressure (bar)")
        ax2.set_ylabel("Pressure at REUV (bar)")
        ax2.set_yscale('log')
        ax2.tick_params(axis='y')

        skip_masses = {10, 12, 14}
        for i, mass in enumerate(m_planet):
            if round(mass) not in skip_masses:
                ax1.annotate(f"{mass:.0f}M⊕", (T_outflow[i], P_EUV_Pa[i]), textcoords="offset points", xytext=(-13,3), ha='left', fontsize=6, color='black')

        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_radii_comparison(results):
        """Plot R_transit, REUV, RS_flow, and R_b vs Planet Mass."""
        params = ModelParams()
        mearth = params.mearth
        m_planet = [res['m_planet']/mearth for res in results]
        REUV = [res['REUV'] for res in results]
        RS_flow = [res['RS_flow'] for res in results]
        R_b = [res['R_b'] for res in results]
        R_t = [res['r_planet'] for res in results]

        plt.figure()
        plt.plot(m_planet, REUV, '.-', label="REUV")
        plt.plot(m_planet, RS_flow, '.-', label="RS_flow")
        plt.plot(m_planet, R_b, '.-', label="R_b")
        plt.plot(m_planet, R_t, '.-', label="R_transit")
        
        plt.xlabel("Planet Mass (Earth Masses)")
        plt.ylabel("Radius (m)")
        plt.yscale("log")
        plt.legend()
        plt.show()

    @staticmethod
    def water_loss_over_time(results):
        """Plot water loss (in Earth Oceans) over different timescales vs planet mass."""
        params = ModelParams()
        mearth = params.mearth

        # timescales in seconds
        timescales = {
            "1 Myr": 1e6 * 3.154e7,
            "50 Myr": 50e6 * 3.154e7,
            "100 Myr": 100e6 * 3.154e7,
            "200 Myr": 200e6 * 3.154e7,
        }

        EO_mass = 1.39e24 # mass of EO in grams
        m_planet = [res['m_planet'] / mearth for res in results]
        water_loss_EOs_phi_H = {label: [] for label in timescales.keys()}

        for res in results:
            phi_H = res["phi_H"] # Hydrogen flux (g cm^-2 s^-1)
            REUV = res["REUV"] # Radius at which flux is measured (cm)

            Mdot_H = phi_H * (4 * np.pi * REUV**2) # flux to mass loss rate (g/s)
            Mdot_H2O = Mdot_H

            for label, time_sec in timescales.items():
                total_water_loss_phi_H = Mdot_H2O * time_sec  # Total mass lost in grams
                water_loss_EOs_phi_H[label].append(total_water_loss_phi_H / EO_mass)  # Convert to EO units

        plt.figure()

        for label in timescales.keys():
            plt.plot(m_planet, water_loss_EOs_phi_H[label], '.-', label=f"{label}")

        plt.xlabel("Planet Mass (Earth Masses)")
        plt.ylabel("Water Loss (Earth Oceans)")
        plt.yscale("log")
        plt.legend()

        plt.show()

    @staticmethod
    def plot1(df):
        """
        Plot total mass loss rate (Mdot) versus planetary mass (in Earth masses).
        - x-axis: Planet mass (Earth masses, on a log scale)
        - y-axis: Total mass loss rate (Mdot, on a log scale)
        - Line style: Different for each Teq (for example: solid, dashed, dotted, dash-dot)
        - Marker/line color: Encodes FEUV (with a colorbar)
        
        Expected DataFrame columns:
            'm_planet' : planetary mass (cgs)
            'Mdot'     : total mass loss rate
            'Teq'      : equilibrium temperature (K)
            'FEUV'     : flux (erg cm^-2 s^-1)
        """
        params = ModelParams()
        mearth = params.mearth

        df['m_planet_Earth'] = df['m_planet'] / mearth # convert to earth masses for plotting

        line_styles = {300: '-', 400: '--', 1000: '-.', 2000: ':'} # styles for different Teq lines
        unique_Teq = np.sort(df['Teq'].unique())
        for teq in unique_Teq:
            if teq not in line_styles:
                line_styles[teq] = '-' # default solid line

        cmap = get_cmap("viridis") # colormap for FEUV
        # norm = mcolors.Normalize(vmin=df['FEUV'].min(), vmax=df['FEUV'].max())
        norm = mcolors.LogNorm(vmin=df['FEUV'].min(), vmax=df['FEUV'].max())

        fig, ax = plt.subplots(figsize=(10,8))
        
        # Group the DataFrame by FEUV and Teq values: for a given combination of flux and temperature we draw one curve
        grouped = df.groupby(['FEUV', 'Teq'])
        for (flux, teq), group in grouped:
            group = group.sort_values('m_planet_Earth')
            style = line_styles.get(teq, '-')  # get the line style for this Teq
            ax.plot(group['m_planet_Earth'], group['Mdot'],
                    linestyle=style, marker='o',
                    color=cmap(norm(flux)),
                    label=f"Teq={teq:.0f}K, FEUV={flux:.1e}")
        
        ax.set_xscale('linear')
        ax.set_yscale('log')
        ax.set_xlabel("Planet Mass (Earth Masses)")
        ax.set_ylabel("Mass Loss Rate (Mdot)")

        # create and attach a colorbar for FEUV
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax) # pass the axis explicitly
        cbar.set_label("FEUV (erg cm$^{-2}$ s$^{-1}$)")

        # custom legend for the style lines, by Teq
        custom_lines = [Line2D([], [], color='black', linestyle=line_styles[teq], lw=2) for teq in unique_Teq]
        custom_labels = [f"T = {teq:.0f} K" for teq in unique_Teq]
        ax.legend(custom_lines, custom_labels, loc='best', fontsize='small')

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot2(df):
        """
        Plot FEUV versus planet mass with:
          - x-axis: Planet mass in Earth masses (log scale)
          - y-axis: FEUV (log scale)
          - Marker color determined by Mdot (with a colorbar)
          - Lines connecting points of the same Teq group, with distinct line styles for each Teq
          
        Expected DataFrame columns:
            'm_planet' : planetary mass (cgs)
            'FEUV'     : flux (erg cm^-2 s^-1)
            'Mdot'     : mass loss rate
            'Teq'      : equilibrium temperature (K)
        """
        # Get conversion factors from ModelParams.
        params = ModelParams()
        mearth = params.mearth
        df['m_planet_Earth'] = df['m_planet'] / mearth  # convert to Earth masses

        # Define line styles for specific Teq values.
        line_styles = {300: '-', 400: '--', 1000: '-.', 2000: ':'}
        unique_Teq = np.sort(df['Teq'].unique())
        for teq in unique_Teq:
            if teq not in line_styles:
                line_styles[teq] = '-'  # default line style

        cmap = get_cmap("viridis")
        norm = mcolors.LogNorm(vmin=df['Mdot'].min(), vmax=df['Mdot'].max())

        fig, ax = plt.subplots(figsize=(10,8))

        for teq, group in df.groupby("Teq"): # group the data by Teq so that each group is plotted with its own line style
            group = group.sort_values("m_planet_Earth")
            # plot the connecting line (using a fixed color, e.g. black, with a style to indicate Teq)
            ax.plot(group["m_planet_Earth"], group["FEUV"], linestyle=line_styles[teq],
                    color="black", alpha=0.5)
            sc = ax.scatter(group["m_planet_Earth"], group["FEUV"], c=group["Mdot"], 
                            cmap=cmap, norm=norm, edgecolor="k", s=50,
                            label=f"Teq = {teq:.0f} K") # markers colored by Mdot
        
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_xlabel("Planet Mass (Earth Masses)")
        ax.set_ylabel("FEUV (erg cm$^{-2}$ s$^{-1}$)")

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Mass Loss Rate (Mdot)")

        # custom legend for the Teq line styles
        custom_lines = [Line2D([], [], color="black", linestyle=line_styles[teq], lw=2) for teq in unique_Teq]
        ax.legend(custom_lines, [f"T = {teq:.0f} K" for teq in unique_Teq],
                  loc="best", fontsize="small")

        plt.tight_layout()
        plt.show()

    
    @staticmethod
    def plot3(df):
        """
        Plot FEUV (x-axis) vs. Mdot (y-axis) where:
          - Marker size is scaled by planet mass (converted to Earth masses),
          - Marker color is given by equilibrium temperature (Teq) with a colorbar,
          - Each planet type using a distinct marker shape and edge color.
        
        Expected DataFrame columns:
            'm_planet' : planetary mass (cgs)
            'Mdot'     : mass loss rate
            'FEUV'     : EUV flux (erg cm^-2 s^-1)
            'Teq'      : equilibrium temperature (K)
            'planet_type': Group identifier for different datasets.
        """
        params = ModelParams()
        mearth = params.mearth

        df['m_planet_Earth'] = df['m_planet'] / mearth
        
        scale_factor = 20
        
        cmap = cm.Set2
        norm = mcolors.Normalize(vmin=df['Teq'].min(), vmax=df['Teq'].max())

        marker_styles = {
        '3H2O_superEarth': 'o',        # circle
        '3HHe_subNeptune': 's',        # square
        '3HHe_10H2O_subNeptune': '^'   # triangle
        }
        edge_colors = {
        '3H2O_superEarth': 'black',
        '3HHe_subNeptune': 'red',
        '3HHe_10H2O_subNeptune': 'blue'
        }
        
        fig, ax = plt.subplots(figsize=(8,6))

        for ptype, marker in marker_styles.items():
            # subdf = df[df['planet_type'] == ptype]
            subdf = df[df['planet_type'] == ptype]
            sizes = subdf['m_planet_Earth'] * scale_factor
            sc = ax.scatter(
                subdf['FEUV'], 
                subdf['Mdot'], 
                s=sizes, 
                c=subdf['Teq'], 
                cmap=cmap, 
                norm=norm,
                marker=marker, 
                edgecolor=edge_colors[ptype], 
                label=ptype, 
                alpha=0.8
            )
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("FEUV (erg cm$^{-2}$ s$^{-1}$)")
        ax.set_ylabel("Mass Loss Rate (Mdot)")
        ax.legend(title="Planet Type", loc='lower right')
                
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Equilibrium Temperature (K)")

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot3_1(df):
        """
        Plot FEUV (x-axis) vs. Mdot (y-axis) where:
          - Marker size is scaled by planet mass (converted to Earth masses),
          - Marker color is given by equilibrium temperature (Teq) with a colorbar,
          - Each planet type using a distinct marker shape and edge color.
        
        Expected DataFrame columns:
            'm_planet' : planetary mass (cgs)
            'Mdot'     : mass loss rate
            'FEUV'     : EUV flux (erg cm^-2 s^-1)
            'Teq'      : equilibrium temperature (K)
            'planet_type': Group identifier for different datasets.
        """
        params = ModelParams()
        mearth = params.mearth

        df['m_planet_Earth'] = df['m_planet'] / mearth
        
        scale_factor = 20
        
        cmap = cm.Set2
        norm = mcolors.Normalize(vmin=df['Teq'].min(), vmax=df['Teq'].max())

        marker_styles = {
        '3H2O_superEarth': 'o',        # circle
        '3HHe_subNeptune': 's',        # square
        '3HHe_10H2O_subNeptune': '^'   # triangle
        }
        edge_colors = {
        '3H2O_superEarth': 'black',
        '3HHe_subNeptune': 'red',
        '3HHe_10H2O_subNeptune': 'blue'
        }
        
        fig, ax = plt.subplots(figsize=(8,6))

        plotted_dfs = []

        for ptype, marker in marker_styles.items():
            subdf = df[df['planet_type'] == ptype]
            plotted_dfs.append(subdf)
            sizes = subdf['m_planet_Earth'] * scale_factor
            sc = ax.scatter(
                subdf['FEUV'], 
                subdf['Mdot'], 
                s=sizes, 
                c=subdf['Teq'], 
                cmap=cmap, 
                norm=norm,
                marker=marker, 
                edgecolor=edge_colors[ptype], 
                label=ptype, 
                alpha=0.8
            )
        
        plotted_df = pd.concat(plotted_dfs)

        for mass in plotted_df['m_planet_Earth'].unique():
            group = plotted_df[plotted_df['m_planet_Earth'] == mass]
            if len(group) > 1:
                # Sort the group by FEUV for proper line connection.
                group_sorted = group.sort_values('FEUV')
                ax.plot(group_sorted['FEUV'], group_sorted['Mdot'], linestyle='-', color='gray', linewidth=0.5, alpha=0.5)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("FEUV (erg cm$^{-2}$ s$^{-1}$)")
        ax.set_ylabel("Mass Loss Rate (Mdot)")
        ax.legend(title="Planet Type", loc='lower right')
                
        cbar = fig.colorbar(ax.collections[-1], ax=ax)
        cbar.set_label("Equilibrium Temperature (K)")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot3_5(df):
        """
        Plot FEUV (x-axis) vs. Mdot (y-axis) where:
          - Marker size is scaled by planet mass (converted to Earth masses),
          - Marker color is given by equilibrium temperature (Teq) with a colorbar,
          - Each planet type using a distinct marker shape and edge color.
        
        Expected DataFrame columns:
            'm_planet' : planetary mass (cgs)
            'Mdot'     : mass loss rate
            'FEUV'     : EUV flux (erg cm^-2 s^-1)
            'Teq'      : equilibrium temperature (K)
            'planet_type': Group identifier for different datasets.
        """
        params = ModelParams()
        mearth = params.mearth
        df['m_planet_Earth'] = df['m_planet'] / mearth
        
        scale_factor = 20
        
        cmap = cm.Set2
        norm = mcolors.Normalize(vmin=df['Teq'].min(), vmax=df['Teq'].max())

        marker_styles = {
        '3H2O_superEarth': 'o',        # circle
        '3HHe_subNeptune': 's',        # square
        '3HHe_10H2O_subNeptune': '^'   # triangle
        }
        edge_colors = {
        '3H2O_superEarth': 'black',
        '3HHe_subNeptune': 'red',
        '3HHe_10H2O_subNeptune': 'blue'
        }
        
        fig, ax = plt.subplots(figsize=(8,6))

        plotted_dfs = []

        for ptype, marker in marker_styles.items():
            subdf = df[df['planet_type'] == ptype].iloc[::2]
            plotted_dfs.append(subdf)
            sizes = subdf['m_planet_Earth'] * scale_factor
            sc = ax.scatter(
                subdf['FEUV'], 
                subdf['Mdot'], 
                s=sizes, 
                c=subdf['Teq'], 
                cmap=cmap, 
                norm=norm,
                marker=marker, 
                edgecolor=edge_colors[ptype], 
                label=ptype, 
                alpha=0.8
            )
        
        plotted_df = pd.concat(plotted_dfs)

        for mass in plotted_df['m_planet_Earth'].unique():
            group = plotted_df[plotted_df['m_planet_Earth'] == mass]
            if len(group) > 1:
                # Sort the group by FEUV for proper line connection.
                group_sorted = group.sort_values('FEUV')
                ax.plot(group_sorted['FEUV'], group_sorted['Mdot'], linestyle='-', color='gray', linewidth=0.5, alpha=0.5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("FEUV (erg cm$^{-2}$ s$^{-1}$)")
        ax.set_ylabel("Mass Loss Rate (Mdot)")
        ax.legend(title="Planet Type", loc='lower right')
                
        cbar = fig.colorbar(ax.collections[-1], ax=ax)
        cbar.set_label("Equilibrium Temperature (K)")

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot4(df):
        """
        Plot FEUV (x-axis) vs. Mdot (y-axis) where:
          - Marker size is scaled by planet mass (converted to Earth masses),
          - Marker color is given by equilibrium temperature (Teq) with a colorbar.
        
        Expected DataFrame columns:
            'm_planet' : planetary mass (cgs)
            'Mdot'     : mass loss rate
            'FEUV'     : EUV flux (erg cm^-2 s^-1)
            'Teq'      : equilibrium temperature (K)
        """
        params = ModelParams()
        mearth = params.mearth

        df['m_planet_Earth'] = df['m_planet'] / mearth

        scale_factor = 20
        sizes = df['m_planet_Earth'] * scale_factor
        
        cmap = cm.viridis
        norm = mcolors.LogNorm(vmin=df['Mdot'].min(), vmax=df['Mdot'].max())
        
        fig, ax = plt.subplots(figsize=(10,8))
        
        # Plot scatter: x is FEUV, y is Teq.
        # Points are sized by planet mass and colored by Mdot.
        sc = ax.scatter(df['FEUV'], df['Teq'], s=sizes, c=df['Mdot'], cmap=cmap,
                        norm=norm, edgecolor='k', alpha=0.8)

        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlabel("FEUV (erg cm$^{-2}$ s$^{-1}$)")
        ax.set_ylabel("Equlibrium temperature (K)")
        
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Mass Loss Rate (g/s)")
        
        plt.tight_layout()
        plt.show()