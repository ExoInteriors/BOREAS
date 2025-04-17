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