import os
import numpy as np

class ModelDataLoader:
    def __init__(self, base_path, params):
        self.base_path = base_path
        self.params = params

    def load_model_data(self, m_planet, water_percent):
        """
        Load model data for given planet masses and water mass fractions.

        Load with
        --
        loader = ModelDataLoader('/Users/mvalatsou/PhD/Repos/MR_perplex_old/OUTPUT/CW/critical_WMF/gridmin295/')
        m_planet = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        water_percent = [x / 10 for x in range(1, 81)]
        radius, mass, Teq = loader.load_model_data(m_planet, water_percent)
        --
        in the main function.
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
                    # Teq = 2500.

                    radii_list.append(r_planet)
                    mass_list.append(m_planet)
                    teq_list.append(Teq)

        return radii_list, mass_list, teq_list

    
    def load_single_ddat_file(self, ddat_filename):
        """
        Load a single .ddat file by name and return arrays for mass, radius, and temperature.
        """
        file_path = os.path.join(self.base_path, ddat_filename)

        rows = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comment lines and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse the line and skip if the radius value (2nd column) is -1
                row = [float(x) for x in line.split()]
                if row[1] == -1.0:
                    continue
                rows.append(row)

        data = np.array(rows)
        mass = data[:, 0] # in earth masses
        radius = data[:, 1] # in earth radii
        temperature = data[:, 2] # K

        mass = mass * self.params.mearth # convert to g
        radius = radius * self.params.rearth # convert to cm

        return mass, radius, temperature
    
    
    def load_all_ddat_files(self):
        """
        Loop over all .ddat files in base_path, parse them, and store their mass, radius, and temperature columns.
        
        Returns:
        data_dict : dict
            Keys are filenames (e.g., '3H2O_superEarth.ddat').
            Values are dictionaries with 'mass', 'radius', and 'temperature' arrays.
        """
        data_dict = {}

        for filename in os.listdir(self.base_path):
            if filename.endswith('.ddat'):
                mass, radius, temperature = self.load_single_ddat_file(filename)
                data_dict[filename] = {'mass': mass, 'radius': radius, 'temperature': temperature}

        return data_dict