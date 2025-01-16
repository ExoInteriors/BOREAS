import os
import numpy as np

class ModelDataLoader:
    def __init__(self, base_path):
        self.base_path = base_path

    def load_model_data(self, m_planet, water_percent):
        """
        Load model data for given planet masses and water mass fractions.
        """
        radii_list = []
        mass_list = []
        teq_list = []

        for mass in m_planet:
            for wmf in water_percent:
                model_path = os.path.join(self.base_path, f"M{mass}W{wmf}_p_T_g_r_m_rho.ddat")
                
                if os.path.exists(model_path):
                    with open(model_path, 'r') as file:
                        lines = file.readlines()

                    pressure_row = lines[1].strip().split()
                    temperature_row = lines[2].strip().split()
                    radius_row = lines[4].strip().split()
                    mass_row = lines[5].strip().split()

                    pressure = np.array([float(value) for value in pressure_row[-400:]])
                    temperature = np.array([float(value) for value in temperature_row[-400:]])
                    r_planet = np.array([float(value) for value in radius_row[-400:]])
                    m_planet = np.array([float(value) for value in mass_row[-400:]])

                    r_planet = r_planet[-1] # planet radius, cm.
                    m_planet = m_planet[-1] # planet mass in grams
                    Teq = temperature[-1]   # equilibrium temperature, K.

                    radii_list.append(r_planet)
                    mass_list.append(m_planet)
                    teq_list.append(Teq)

        return radii_list, mass_list, teq_list