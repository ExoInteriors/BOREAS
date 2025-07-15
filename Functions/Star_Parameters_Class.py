import numpy as np
import warnings
import scipy

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
        self.isochrone_file = "BHAC15_iso.2mass.txt" # name of the file containing isochrone data to infer the bolometric luminosity
        self.Lbol = None            # bolometric luminosity of the star in erg/s
        self.Teff = None            # effective temperature of the star in K
        self.star_radius = None     # radius of the star in solar radii

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
    
    def get_data(self):
        '''
        Extract data from isochrones given by Baraffe et al. 2015
        Side note: some data points are missing after a certain time for stars of mass < 0.07 solar masses and > 0.9 solar masses, those data points are replaced by 0
        
        Uses the attribute:
        isochrone_file: string, path of the file containing data from which effective temperature, bolometric luminosity and radius will be interpolated

        Return:
        t: numpy array, points in time for which we have isochrones (in Gyrs)
        M: numpy array, star masses for which we have data (in solar masses)
        Teff: numpy array, effective temperature of stars (in K), for each point in M (axis 0) and t (axis 1)
        L: numpy array, bolometric lumimosity of stars (in log solar luminosity), for each point in M (axis 0) and t (axis 1)
        R: numpy array, radius of stars (in solar radii), for each point in M (axis 0) and t (axis 1)
        '''

        file = self.isochrone_file

        # time in Gyrs
        t = np.array([0.0005, 0.0010, 0.0020, 0.0030, 0.0040, 0.0050, 0.0080, 0.0100, 0.0150, 0.0200, 0.0250, 0.0300, 0.0400, 0.0500, 0.0800, 0.1000, 0.1200, 0.2000, 0.3000, 0.4000, 0.5000, 0.6250, 0.8000, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 8.0000, 10.0000])

        rawtab = np.loadtxt(file, dtype=float, comments='!', skiprows=8)
        # mass in solar masses
        M = rawtab.copy()[:30, 0]

        # temperature in K, luminosity in solar luminosity and radius in solar radii
        Teff = np.full((len(M), len(t)), fill_value=0., dtype=float)
        L = np.full_like(Teff, fill_value=0.)
        R = np.full_like(Teff, fill_value=0.)

        n = 0 #little helpful index

        for i in range(len(t)):
            if (i < 8):
                Teff[:, i] = rawtab.copy()[n:n+len(M), 1]
                L[:, i] = rawtab.copy()[n:n+len(M), 2]
                R[:, i] = rawtab.copy()[n:n+len(M), 4]
                n += len(M)
            elif (i > 7) and (i < 14):
                Teff[1:, i] = rawtab.copy()[n:n+len(M)-1, 1]
                L[1:, i] = rawtab.copy()[n:n+len(M)-1, 2]
                R[1:, i] = rawtab.copy()[n:n+len(M)-1, 4]
                n += len(M) - 1
            elif (i > 13) and (i < 17):
                Teff[3:, i] = rawtab.copy()[n:n+len(M)-3, 1]
                L[3:, i] = rawtab.copy()[n:n+len(M)-3, 2]
                R[3:, i] = rawtab.copy()[n:n+len(M)-3, 4]
                n += len(M) - 3
            elif (i > 16) and (i < 19):
                Teff[4:, i] = rawtab.copy()[n:n+len(M)-4, 1]
                L[4:, i] = rawtab.copy()[n:n+len(M)-4, 2]
                R[4:, i] = rawtab.copy()[n:n+len(M)-4, 4]
                n += len(M) - 4
            elif (i > 18) and (i < 22):
                Teff[5:, i] = rawtab.copy()[n:n+len(M)-5, 1]
                L[5:, i] = rawtab.copy()[n:n+len(M)-5, 2]
                R[5:, i] = rawtab.copy()[n:n+len(M)-5, 4]
                n += len(M) - 5
            elif (i > 21) and (i < 24):
                Teff[6:, i] = rawtab.copy()[n:n+len(M)-6, 1]
                L[6:, i] = rawtab.copy()[n:n+len(M)-6, 2]
                R[6:, i] = rawtab.copy()[n:n+len(M)-6, 4]
                n += len(M) - 6
            elif (i == 24):
                Teff[7:, i] = rawtab.copy()[n:n+len(M)-7, 1]
                L[7:, i] = rawtab.copy()[n:n+len(M)-7, 2]
                R[7:, i] = rawtab.copy()[n:n+len(M)-7, 4]
                n += len(M) - 7
            else:
                Teff[7:-(i-24), i] = rawtab.copy()[n:n+len(M)-i+17, 1]
                L[7:-(i-24), i] = rawtab.copy()[n:n+len(M)-i+17, 2]
                R[7:-(i-24), i] = rawtab.copy()[n:n+len(M)-i+17, 4]
                n += len(M) - i + 17

        return t, M, Teff, L, R

    def double_interpolate(self, t, M, tab, mass_star, t_final):
        '''
        Uses cubic splines interpolation to get the values of the quantity stored in tab for the wanted star mass and time points

        Parameters:
        t: numpy array, points in time for which we have isochrones (in Gyrs)
        M: numpy array, star masses for which we have data (in solar masses)
        tab: numy array, data to interpolate for each point in M (axis 0) and t (axis 1)
        mass_star: float, mass of studied star (in solar masses)
        t_final: numpy array, points in time where we compute the EUV flux (in Gyrs)

        Return:
        tab_final: numpy array, interpolated values
        '''

        # we perform a first series of interpolations to get values for the chosen star mass over the initial time grid
        tab_inter = np.full(len(t), fill_value=0., dtype=float)
        for i in range(len(tab_inter)):
            f = scipy.interpolate.CubicSpline(M, tab[:, i], bc_type='natural')
            tab_inter[i] = f([mass_star])[0]
        
        # we perform another interpolation to get the values for the desired time points
        f = scipy.interpolate.CubicSpline(t, tab_inter, bc_type='natural')
        tab_final = f(t_final)

        return tab_final
    
    def update_Lbol(self):
        '''
        Updates the bolometric luminosity of the star based on its age using isochrones
        Also updates the bolometric flux received by the planet to keep consistence
        '''
        
        t_final = np.linspace(0.001, self.star_age, 3000)
        t, M, Teff, L, R = self.get_data()
        L = 10**self.double_interpolate(t, M, L, self.star_mass, t_final) * 3.826E33
        self.Lbol = L[-1]
        self.params.update_param("Fbol", L[-1]/(4*np.pi*(self.params.aplau*1.5E13)**2))
    
    def update_Teff(self):
        '''
        Updates the effective temperature of the star based on its age using isochrones
        '''
        
        t_final = np.linspace(0.001, self.star_age, 3000)
        t, M, Teff, L, R = self.get_data()
        Teff_final = self.double_interpolate(t, M, Teff, self.star_mass, t_final)
        self.Teff = Teff_final[-1]
    
    def update_star_radius(self):
        '''
        Updates the radius of the star based on its age using isochrones
        '''
        
        t_final = np.linspace(0.001, self.star_age, 3000)
        t, M, Teff, L, R = self.get_data()
        R_final = self.double_interpolate(t, M, R, self.star_mass, t_final)
        self.star_radius = R_final[-1]
    
    def update_Teq(self):
        '''
        Updates the equilibrium temperature of the planet based on the bolometric luminosity of the star
        '''

        if self.Lbol is None:
            raise ValueError("Lbol attribute must have a value to be able to compute the equilibrium temperature.")

        self.Teq = ((1-self.params.albedo) * self.Lbol * 10**(-7) / (16 * self.params.epsilon * self.params.beta_planet * self.params.Stefan_SI * np.pi * (self.params.aplau*1.5E11)**2))**(1/4)
    
    def update_FEUV_from_Lbol(self):
        '''
        Updates the XUV flux received by the planet after deriving it from the bolometric luminosity, age and mass of the star
        Uses equations 12 and 13 from Rogers et al. 2021
        '''

        ##### constants ######
        # in both cases we take the values used in Rogers et al. 2021, taken from Wright et al. 2011 and Jackson et al. 2012
        alpha = -1.5
        saturation_ratio = 10**(-3.5)

        t_sat = 1E-1 / self.star_mass

        if self.star_age < t_sat:
            LXUV = saturation_ratio * self.star_mass**(-0.5) * self.Lbol
        else:
            LXUV = saturation_ratio * self.star_mass**(-0.5) * (self.star_age / t_sat)**alpha * self.Lbol
        
        self.params.update_param("FEUV", LXUV/(4*np.pi*(self.params.aplau*1.5E13)**2))
