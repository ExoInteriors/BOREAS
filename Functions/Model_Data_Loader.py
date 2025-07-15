import os
import numpy as np

class ModelDataLoader:
    def __init__(self, base_path, params):
        self.base_path = base_path
        self.params = params
    
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