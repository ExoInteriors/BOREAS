import numpy as np
import warnings

class StarParams():
    """
    Computes planetary properties such as they are coherent with stellar properties
    Added by Pierlou Marty, 2025.
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
        self.Teq = None             # equilibrium temperature of planet in Kelvin

    def update_param(self, param_name, value):
        """Dynamically update a parameter value."""
        if hasattr(self, param_name):
            setattr(self, param_name, value)
        else:
            raise AttributeError(f"Parameter '{param_name}' does not exist.")
    
    def get_param(self, param_name):
        """Retrieve the value of a parameter."""
        return getattr(self, param_name, None)
    
    def update_FEUV_from_Teq(self, star_mass=None, star_age=None, Teq=None):
        """
        /!\ This method can be used in 2 ways: either the 3 entry parameters are not given values when the method is called and they
        will take the respective values of the star_mass, star_age and Teq attributes of the StarParams object or they are given values
        when the method is called. If only some of them are given values when calling the method, the others will take the values of their
        associated StarParams attributes, but it is not recommended to use the method in this way.

        Computes the XUV flux received by a planet compatible with a given equilibrium temperature to update the associated 
        ModelParams object FEUV attribute. For this the function combines the planetary radiation balance, 
        equations 12 and 13 from Rogers et al. 2021 and the flux definition. In complement to the equilibrium temperature, 
        the function also needs the star age and mass and the planet albedo and fraction of the surface re-radiating the absorbed flux.
        The flux is noted as FEUV to stay consistent with the rest of the code.

        :param star_mass: float, mass of star in solar masses
        :param star_age: float, age of star in Gyrs
        :param Teq: float, equilibrium temperature of the planet in K
        """

        ###### constants #######
        # in both cases we take the values used in Rogers et al. 2021, taken from Wright et al. 2011 and Jackson et al. 2012
        alpha = -1.5
        saturation_ratio = 10**(-3.5)

        if (star_mass is None) or (star_age is None) or (Teq is None):
            if (star_mass is None) and (star_age is None) and (Teq is None):
                if self.star_age is None:
                   raise ValueError('star_age must have a value to compute the the EUV flux received from the star')
                if self.star_mass is None:
                    raise ValueError('star_mass must have a value to compute the the EUV flux received from the star')
                if self.Teq is None:
                    raise ValueError('Teq must have a value to compute the the EUV flux received from the star')
                star_mass = self.star_mass
                star_age = self.star_age
                Teq = self.Teq
            else:
                if star_mass is None:
                    if self.star_mass is None:
                        raise ValueError('star_mass must have a value to compute the the EUV flux received from the star')
                    else:
                        star_mass = self.star_mass
                        warnings.warn('The method is using a star_mass value from the attribute of the StarParams object while ' \
                        + 'star_age and/or Teq are directly taken as input of the method, which is not a recommanded use of it.')
                if star_age is None:
                    if self.star_age is None:
                        raise ValueError('star_age must have a value to compute the the EUV flux received from the star')
                    else:
                        star_age = self.star_age
                        warnings.warn('The method is using a star_age value from the attribute of the StarParams object while ' \
                        + 'star_mass and/or Teq are directly taken as input of the method, which is not a recommanded use of it.')
                if Teq is None:
                    if self.Teq is None:
                        raise ValueError('Teq must have a value to compute the the EUV flux received from the star')
                    else:
                        Teq = self.Teq
                        warnings.warn('The method is using a Teq value from the attribute of the StarParams object while ' \
                        + 'star_mass and/or star_age are directly taken as input of the method, which is not a recommanded use of it.')
                  
        t_sat = 1E-1 / star_mass

        if star_age < t_sat:
            FEUV = 4 * self.params.beta_planet * self.params.epsilon * self.params.Stefan_SI * saturation_ratio * Teq**4 * 1E3 / ((1 - self.params.albedo) * star_mass**0.5)
        else:
            FEUV = 4 * self.params.beta_planet * self.params.epsilon * self.params.Stefan_SI * saturation_ratio * (star_age/ t_sat)**alpha * Teq**4 * 1E3 / ((1 - self.params.albedo) * star_mass**0.5)

        self.params.update_param('FEUV', FEUV)
    
    def get_FEUV_range_from_age(self, star_age=None):
        """
        /!\ This method can be used in 2 ways: either a star_age value is provided when the method is called or the value of
        the star_age attribute of the StarParams object will be used.

        Computes the range of XUV flux possible for a given star age and planet equilibrium temperature with the star mass
        being a free parameter. Uses Rogers et al 2021 and Baraffe et al. 2015.
        
        Return:
        FEUV_range: numpy array, range of XUV flux for the given star age and equilibrium temperature (in erg/s/cm2)
        """

        if star_age is None:
            star_age = self.star_age

        star_mass_ar = np.linspace(0.1, 1.4, 50)
        F_EUV = np.empty_like(star_mass_ar)
        for i in range(len(star_mass_ar)):
            self.update_FEUV_from_Teq(star_mass=star_mass_ar[i], star_age=star_age, Teq=self.Teq)
            F_EUV[i] = self.params.FEUV
        
        F_EUV_max = np.max(F_EUV)
        F_EUV_min = np.min(F_EUV)

        FEUV_range = np.logspace(np.log10(F_EUV_min), np.log10(F_EUV_max), 10)

        return FEUV_range
    
    def get_FEUV_range_any_age(self):
        """
        Computes the range of XUV flux possible for a given star age and planet equilibrium temperature with the star mass 
        and age being free parameters. Uses Rogers et al 2021 and Baraffe et al. 2015.

        Return:
        FEUV_range: numpy array, range of XUV flux for this equilibrium temperature (in erg/s/cm2)
        """
        
        star_age_ar = np.linspace(0.001, 13.4, 1000) # the maximum age explored here is the age of the oldest Milky Way stars
        F_EUV_max_inter = np.empty_like(star_age_ar)
        F_EUV_min_inter = np.empty_like(star_age_ar)

        for i in range(len(star_age_ar)):
            F_EUV_inter = self.get_FEUV_range_from_age(star_age_ar[i])
            F_EUV_max_inter[i], F_EUV_min_inter[i] = F_EUV_inter[-1], F_EUV_inter[0]

        F_EUV_max = np.max(F_EUV_max_inter)
        F_EUV_min = np.min(F_EUV_min_inter)

        FEUV_range = np.logspace(np.log10(F_EUV_min), np.log10(F_EUV_max), 10)

        return FEUV_range
    
    def get_Fbol_from_Teq(self):
        """
        Computes bolometric luminosity received by a planet compatible with a given equilibrium temperature to update the associated 
        ModelParams object Fbol attribute. The equilibrium temperature is an attribute of the StarParams object that must be up to date
        before calling this method.
        """
        
        Fbol = 4 * self.params.epsilon * self.params.beta_planet * self.params.Stefan_SI * self.Teq**4 * 1E3 / (1 - self.params.albedo)
        
        # self.params.update_param('Fbol', Fbol)
        return Fbol