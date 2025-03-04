class ModelParams:
    """
    Base class for handling model parameters and physical constants.
    """
    def __init__(self):
        # --- Physical constants
        self.am_h = 1                   # Αtomic mass hydrogen, u
        self.am_o = 16                  # Αtomic mass oxygen, u
        self.m_H = 1.6735575e-24        # Mass of hydrogen atom, g (also proton mass)
        self.m_O = self.am_o * self.m_H # Μass of oxygen atom, grams
        self.mmw_H = 2.35                                   # Mean molecular weight (H/He-ish envelope)
        self.mmw_eq = 2 * self.am_h + self.am_o             # Mean molecular weight (water) at bolometrically heated region
        self.mmw_outflow = ((2 * self.am_h + self.am_o)/3)  # Mean molecular weight (H and O) assuming full dissociation of water

        self.k_b = 1.380649e-16         # Boltzmann constant, erg K-1
        self.G = 6.67430e-8             # Gravitational constant, cm3 g-1 s-2
        self.rearth = 6.371e8           # Radius earth in cgs
        self.mearth = 5.97424e27        # Mass earth in cgs

        # --- Model-specific parameters
        self.kappa_p = 1                # original value 1e-2 for H/He, opacity to outgoing thermal radiation, i.e. mean opacity in infrared
        self.E_photon = 20 * 1.6e-12    # photon energy
        self.FEUV = 450.                # received EUV flux, ergs cm-2 s-1
        self.sigma_EUV = 1.89e-18       # EUV cross-section (of H? H2?), cm2
        self.alpha_rec = 2.6e-13        # Recombination coefficient, cm3 s-1
        self.eff = 0.3                  # Mass-loss efficiency factor

    def update_param(self, param_name, value):
        """Dynamically update a parameter value."""
        if hasattr(self, param_name):
            setattr(self, param_name, value)
        else:
            raise AttributeError(f"Parameter '{param_name}' does not exist.")
    
    def get_param(self, param_name):
        """Retrieve the value of a parameter."""
        return getattr(self, param_name, None)
