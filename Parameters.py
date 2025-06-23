class ModelParams:
    """
    Base class for handling model parameters and physical constants.
    """

    def update_param(self, param_name, value):
        """Dynamically update a parameter value."""
        if hasattr(self, param_name):
            setattr(self, param_name, value)
        else:
            raise AttributeError(f"Parameter '{param_name}' does not exist.")
    
    def get_param(self, param_name):
        """Retrieve the value of a parameter."""
        return getattr(self, param_name, None)

    def __init__(self):
        self.outflow_mode = 'HHe_H2O' # 'HHe' for pure H/He, 'H2O' for pure steam, 'HHe_H2O' for mixed

        # --- Physical constants
        self.am_h           = 1                                  # Αtomic mass hydrogen, u
        self.am_o           = 16                                 # Αtomic mass oxygen, u
        self.m_H            = 1.6735575e-24                      # Mass of hydrogen atom, g (also proton mass)
        self.m_O            = self.am_o * self.m_H               # Μass of oxygen atom, grams
        self.m_CO2          = 44 * self.m_H                      # Mass of CO2 molecule, grams

        self.mmw_H          = 1                                  # Mean molecular weight (H)
        self.mmw_HHe        = 2.35                               # Mean molecular weight (HHe)

        # - For fully dissociated species assuming *strong* photoevaporation that fully dissociates them (outflow region)
        self.mmw_HHe_outflow = self.mmw_H                         # 1, ignores He, full dissociation gives free H atoms (mmw = 1 per H atom)
        self.mmw_H2O_outflow = (2 * self.am_h + self.am_o) / 3    # = (2+16) / 3 = 6, max mean molecular weight (H, H, and O) assuming full dissociation (H2O -> 2H + O)
        
        # - For mixture of HHe an H2O (outflow region)
        self.X_H2O  = 0.1                                         # H2O mass fraction
        self.X_HHe  = 1 - self.X_H2O                              # HHe mass fraction
        N_HHe       = self.X_HHe / self.mmw_HHe_outflow           # e.g. = 0.9 / 1 = 0.9
        N_H2O       = self.X_H2O / self.mmw_H2O_outflow           # e.g. = 0.1 / 6 ≈ 0.01667
        self.N_tot  = N_HHe + N_H2O                               # total free particles per unit mass
        self.mmw_HHe_H2O_outflow = 1 / self.N_tot                 # e.g. ≈ 1.09, max mean molecular weight (90% of H, H, 10% of H, H, and O) assuming full dissociation of H2 and water
        
        # - Mean molecular weights for non-dissociated species (bolometrically heated region)
        self.mmw_H2O        = 2 * self.am_h + self.am_o                             # Mean molecular weight (water)
        self.mmw_HHe_H2O    = self.X_HHe * self.mmw_HHe + self.X_H2O * self.mmw_H2O # Mean molecular weight (HHe & water)
        
        self.k_b    = 1.380649e-16       # Boltzmann constant, erg K-1
        self.Stefan_SI  = 5.670374419e-8 # Stefan-Boltzmann constant (SI) W m-2 K-4
        self.Stefan_cgs = 5.670374419e-5 # Stefan-Boltzmann constant (cgs) erg cm-2 s-1 K-4
        self.G      = 6.67430e-8         # Gravitational constant, cm3 g-1 s-2
        self.rearth = 6.371e8            # Radius earth in cgs
        self.mearth = 5.97424e27         # Mass earth in cgs

        # --- Model-specific parameters
        self.kappa_p_HHe        = 1e-2   # opacity to outgoing thermal radiation, i.e. mean opacity in infrared 
        self.kappa_p_H2O        = 1      # ↳ roughly pump up the H one by 100
        self.kappa_p_HHe_H2O    = self.X_HHe * self.kappa_p_HHe + self.X_H2O * self.kappa_p_H2O # ↳ approximate this for whichever fraction of water we have

        self.E_photon   = 20 * 1.6e-12   # photon energy
        self.FEUV       = 450.           # received EUV flux, ergs cm-2 s-1
        self.Fbol       = 450.           # received bolometric flux, ergs cm-2 s-1
        self.albedo     = 0.3            # albedo of planet
        self.beta_planet= 0.75           # fraction of the planet's surface that re-emits radiation
        self.epsilon    = 1.             # emissivity of planet
        self.sigma_EUV  = 1.89e-18       # EUV cross-section (of H? H2? similar for O), cm2
        self.alpha_rec  = 2.6e-13        # Recombination coefficient, cm3 s-1
        self.eff        = 0.3            # Mass-loss efficiency factor