import numpy as np


class StarParams():
    """
    Computes planetary properties such as they are coherent with stellar properties
    """

    def __init__(self, params, star_mass=None, star_age=None):
        """
        Initialise the StarParams class with model parameters, and optionally a stellar mass and age which can also be set later
        :param params: An instance of the ModelParams class or a similar object containing constants.
        :param star_mass: float, mass of star in solar masses
        :param star_age: float, age of star in Gyrs
        """
        self.params = params
        self.star_mass = star_mass  # mass of star in solar masses
        self.star_age = star_age    # age of star in Gyrs
        self.Teq = 300.             # equilibrium temperature of planet in Kelvin

    def update_FEUV_from_Teq(self):
        """
        Computes the XUV flux received by a planet compatible with a given equilibrium temperature to update the associated 
        ModelParams object FEUV attribute. For this the function combines the planetary radiation balance, 
        equations 12 and 13 from Rogers et al. 2021 and the flux definition. In complement to the equilibrium temperature, 
        the function also needs the star age and mass and the planet albedo and fraction of the surface re-radiating the absorbed flux.
        The flux is noted as FEUV to stay consistent with the rest of the code.
        """

        ###### constants #######
        # in both cases we take the values used in Rogers et al. 2021, taken from Wright et al. 2011 and Jackson et al. 2012
        alpha = -1.5
        saturation_ratio = 10**(-3.5)

        if self.star_age is None:
            raise ValueError('star_age must have a value to compute the the EUV flux received from the star')
        if self.star_mass is None:
            raise ValueError('star_mass must have a value to compute the the EUV flux received from the star')
        
        t_sat = 1E-1 / self.star_mass

        if self.star_age < t_sat:
            FEUV = 4 * self.params.beta_planet * self.params.epsilon * self.params.stefan_boltzmann * saturation_ratio * self.Teq**4 * 1E3 / ((1 - self.params.albedo) * self.star_mass**0.5)
        else:
            FEUV = 4 * self.params.beta_planet * self.params.epsilon * self.params.stefan_boltzmann * saturation_ratio * (self.star_age/ t_sat)**alpha * self.Teq**4 * 1E3 / ((1 - self.params.albedo) * self.star_mass**0.5)

        self.params.FEUV = FEUV

    def update_param(self, param_name, value):
        """Dynamically update a parameter value."""
        if hasattr(self, param_name):
            setattr(self, param_name, value)
        else:
            raise AttributeError(f"Parameter '{param_name}' does not exist.")
    
    def get_param(self, param_name):
        """Retrieve the value of a parameter."""
        return getattr(self, param_name, None)