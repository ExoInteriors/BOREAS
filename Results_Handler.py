import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from Parameters import ModelParams

class ResultsHandler:
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

            ax1.plot(m_planet, phi_H, '.-', label=f"Flux={flux}", color=color)
            ax2.plot(m_planet, phi_O, '.-', label=f"Flux={flux}", color=color)

        # --- Top subplot (phi_H) ---
        ax1.set_ylabel(r"$\phi_H$ (g cm$^{-2}$ s$^{-1}$)")
        ax1.set_yscale('log')
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # --- Bottom subplot (phi_O) ---
        ax2.set_xlabel("Planet Mass (Earth Masses)")
        ax2.set_ylabel(r"$\phi_O$ (g cm$^{-2}$ s$^{-1}$)")
        ax2.set_yscale('log')
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_phi_vs_flux_across_masses(all_flux_results):
        """phi_H and phi_O vs. the EUV flux (FEUV), with a separate line for each planet mass."""
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

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        
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
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # bottom subplot (phi_O)
        ax2.set_xlabel("EUV Flux (erg cm$^{-2}$ s$^{-1}$)")
        ax2.set_ylabel(r"$\phi_\mathrm{O}$ [g cm$^{-2}$ s$^{-1}$]")
        ax2.set_yscale('log')
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

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