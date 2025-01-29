import pickle
import matplotlib.pyplot as plt
from Parameters import ModelParams

class ResultsHandler:
    @staticmethod
    def save_results(results, filename):
        """Save results to a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(results, f)

    @staticmethod
    def load_results(filename):
        """Load results from a pickle file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

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
        cs = [res['cs'] for res in results]
        REUV = [res['REUV'] for res in results]
        REUV_r = [res['REUV'] / res['r_planet'] for res in results]

        fig, ax1 = plt.subplots()

        ax1.set_xlabel("Sound Speed (cm/s)")
        ax1.set_ylabel("REUV (cm)", color="blue")
        ax1.plot(cs, REUV, 'o-', color="blue", label="REUV")
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor="blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("REUV/r_planet", color="red")
        ax2.plot(cs, REUV_r, 'o-', color="red", label="REUV/r_planet")
        ax2.tick_params(axis='y', labelcolor="red")

        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_REUV_r_planet_vs_m_planet(results):
        """Double y-axis: REUV and r_planet vs m_planet."""
        params = ModelParams()
        mearth = params.mearth
        m_planet = [res['m_planet']/mearth for res in results]
        REUV = [res['REUV'] for res in results]
        r_planet = [res['r_planet'] for res in results]

        plt.figure()
        plt.plot(m_planet, REUV, 'o-', color='blue', label="REUV")
        plt.plot(m_planet, r_planet, 'o-', color='red', label="r_planet")
        plt.xlabel("Planet Mass (Earth Masses)")
        plt.ylabel("Radius (cm)")
        plt.yscale('log')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_T_REUV_m_planet(results):
        """Plot temperature at REUV vs planet mass."""
        params = ModelParams()
        mearth = params.mearth
        m_planet = [res['m_planet']/mearth for res in results]
        T_REUV = [res['T_REUV'] for res in results]

        plt.figure()
        plt.plot(m_planet, T_REUV, 'o-', label="T_REUV")
        plt.xlabel("Planet Mass (Earth Masses)")
        plt.ylabel("Temperature at REUV (K)")
        plt.show()

    @staticmethod
    def plot_xO_phiO_vs_m_planet(results):
        """Double y-axis: x_O and phi_O vs m_planet."""
        params = ModelParams()
        mearth = params.mearth
        m_planet = [res['m_planet']/mearth for res in results]
        x_O = [res['x_O'] for res in results]
        phi_O = [res['phi_O'] for res in results]

        fig, ax1 = plt.subplots()

        ax1.set_xlabel("Planet Mass (Earth Masses)")
        ax1.set_ylabel("x_O", color="blue")
        ax1.set_yscale('log')
        ax1.plot(m_planet, x_O, 'o-', color="blue", label="x_O")
        ax1.tick_params(axis="y", labelcolor="blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("phi_O (g/cm^2/s)", color="red")
        ax2.set_yscale('log')
        ax2.plot(m_planet, phi_O, 'o-', color="red", label="phi_O")
        ax2.tick_params(axis="y", labelcolor="red")

        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_phiH_phiO_vs_m_planet(results):
        """Double y-axis: phi_H and phi_O vs m_planet."""
        params = ModelParams()
        mearth = params.mearth
        m_planet = [res['m_planet']/mearth for res in results]
        phi_H = [res['phi_H'] for res in results]
        phi_O = [res['phi_O'] for res in results]

        fig, ax1 = plt.subplots()

        ax1.set_xlabel("Planet Mass (kg)")
        ax1.set_ylabel("phi_H (g/cm^2/s)", color="blue")
        ax1.set_yscale('log')
        ax1.plot(m_planet, phi_H, 'o-', color="blue", label="phi_H")
        ax1.tick_params(axis="y", labelcolor="blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("phi_O (g/cm^2/s)", color="red")
        ax2.set_yscale('log')
        ax2.plot(m_planet, phi_O, 'o-', color="red", label="phi_O")
        ax2.tick_params(axis="y", labelcolor="red")

        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_phiH_phiO_vs_m_planet_color(results):
        """Phi_H and Phi_O on the same axis with colors mapped to m_planet."""
        params = ModelParams()
        mearth = params.mearth
        m_planet = [res['m_planet']/mearth for res in results]
        phi_H = [res['phi_H'] for res in results]
        phi_O = [res['phi_O'] for res in results]

        plt.figure()
        plt.scatter(m_planet, phi_H, color='blue', label="phi_H")
        plt.scatter(m_planet, phi_O, color='red', label="phi_O")
        plt.xlabel("Planet Mass (Earth Masses)")
        plt.ylabel("Flux Values (g/cm^2/s)")
        plt.yscale('log')
        plt.show()

    @staticmethod
    def plot_pressure_vs_m_planet(results):
        """Plot pressures vs planetary mass for all three methods."""
        params = ModelParams()
        mearth = params.mearth

        m_planet = [res['m_planet'] / mearth for res in results] # Earth masses
        P_EUV_dyn_ideal = [res['P_EUV'] for res in results] # Pressure in dyn/cm2
        P_EUV_Pa = [p * 0.1 for p in P_EUV_dyn_ideal] # Convert dyn/cm2 to Pa
        P_EUV_bar = [p * 1e-6 for p in P_EUV_Pa] # Convert Pa to bar

        fig, ax1 = plt.subplots()

        ax1.plot(m_planet, P_EUV_Pa, 'o-', color="blue", label="Pressure (Pa)")
        ax1.set_xlabel("Planet Mass (Earth Masses)")
        ax1.set_ylabel("Pressure at REUV (Pa)", color="blue")
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor="blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Pressure at REUV (bar)", color="red")
        ax2.plot(m_planet, P_EUV_bar, 's-', color="red", label="Pressure (bar)")  
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor="red")

        plt.tight_layout()
        plt.show()
