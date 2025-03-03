import pickle
import matplotlib.pyplot as plt
import numpy as np
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
        ax1.set_ylabel("$R_{EUV}$ (cm)")
        ax1.plot(cs, REUV, '.-')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y')

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

        plt.title("No fractionation (mmw of outflow = 6)")
        plt.legend()
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
            plt.plot(m_planet, water_loss_EOs_phi_H[label], '.-', label=f"{label} (phi_H-based)")

        plt.xlabel("Planet Mass (Earth Masses)")
        plt.ylabel("Water Loss (Earth Oceans)")
        plt.yscale("log")
        plt.legend()

        plt.show()