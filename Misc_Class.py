import numpy as np

class Misc:
    def __init__(self, params):
        """
        Initialize the MassLoss class with model parameters.
        :param params: An instance of the ModelParams class or a similar object containing constants.
        """
        self.params = params
        
    ### Misc ###
    
    def calculate_pressure_ideal_gas(self, rho_EUV, T_REUV):
        """
        Calculate pressure values at REUV assuming hydrostatic equilibrium, after Debrecht et al. 2019 framework
        """
        k_b = self.params.k_b
        m_H = self.params.m_H
        mmw_H = self.params.mmw_H

        P_REUV = rho_EUV * k_b * T_REUV / (mmw_H * m_H) # ideal gas law

        return P_REUV
    
    def calculate_R_b(self, M_p, c_s):
        """
        Calculate the Bondi radius, where thermal energy and gravitational energy are comparable.
        """
        G = self.params.G

        R_b = (G * M_p) / (c_s ** 2)

        return R_b
    
    def get_flux_range(self, Teq):
        """
        Return a flux range (an array) for a given equilibrium temperature.
        """
        if np.isclose(Teq, 300):
            return np.logspace(np.log10(0.3175), np.log10(1968.2999), 10)
        elif np.isclose(Teq, 400):
            return np.logspace(np.log10(1.004), np.log10(6220.7999), 10)
        elif np.isclose(Teq, 1000):
            return np.logspace(np.log10(39.2057), np.log10(242999.9999), 10)
        elif np.isclose(Teq, 2000):
            return np.logspace(np.log10(627.2926), np.log10(3887999.9999), 10)
        else:
            # Default range if no specific Teq match.
            return np.logspace(np.log10(100), np.log10(10000), 10)