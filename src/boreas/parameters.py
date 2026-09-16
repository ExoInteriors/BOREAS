class ModelParams:
    """
    Base class for handling model parameters and physical constants.
    Supports preset mixing modes for H2, H2O, CO2, CH4 combinations.
    """
    
    def __init__(self):
        # --- default mode controls ---
        self.FXUV       = 450.0           # erg cm^-2 s^-1 (placeholder default)

        # other model parameters and constants
        self.albedo     = 0.3             # albedo of planet
        self.beta       = 0.75            # fraction of the planet's surface that re-emits radiation
        self.epsilon    = 1.              # emissivity of planet
        self.alpha_rec  = 2.6e-13         # Recombination coefficient, cm3 s-1
        self.eff        = 0.3             # Mass-loss efficiency factor

        # XUV cross-sections (cm2) per atom assuming neutral species and monochromatic XUV at 20 eV,
        # for computing the mass absorption coefficient in XUV (χ_XUV, cm2 g-1).
        self.sigma_XUV = {'H': 1.89e-18,  # atomic XUV cross-section (of H), cm2
                        #   'O': 1.89e-18,  # placeholder
                        #   'C': 2.50e-18,  # placeholder
                        #   'N': 3.00e-18,  # placeholder
                        #   'S': 6.00e-18,  # placeholder
                        
                        # Verner+1996 fits at 20 eV, but we keep the older value for hydrogen for now,
                        # since the fits are not great near 20 eV and the older value is more commonly used.
                        # 'H': 2.21e-18,
                        'O': 1.09e-17,
                        'C': 1.01e-17,
                        'N': 1.41e-17,
                        'S': 3.27e-17
                        }

        # --- physical constants ---
        # Universal constants
        self.G          = 6.67430e-8      # Gravitational constant, cm^3 g^-1 s^-2-1
        self.mearth     = 5.972e27        # Earth mass, grams
        self.rearth     = 6.371e8         # Earth radius, cm
        self.E_photon   = 20 * 1.6e-12    # Photon energy
        self.k_b        = 1.380649e-16    # Boltzmann constant, erg K^
        # Integer mass numbers, used only for molecular-weight bookkeeping below
        # (mmw_H2O = 18, mmw_CO2 = 44, ...). These are counts, not masses.
        self.am_h       = 1.0
        self.am_o       = 16.0
        self.am_c       = 12.0
        self.am_n       = 14.0
        self.am_s       = 32.0
        # Particle masses (grams), from standard atomic weights x the atomic mass unit.
        # Do NOT build these as am_X * m_H: m_H is the mass of a hydrogen atom
        # (1.008 amu), not the amu itself, so scaling it by an integer mass number
        # makes every heavy atom ~0.8% too heavy. These are the same weights the
        # b_ij diffusion fits were generated with (see tools/gen_diffusion.py).
        self.amu        = 1.66053906660e-24     # g
        self.m_H        = 1.008  * self.amu     # 1.6738e-24 g
        self.m_C        = 12.011 * self.amu
        self.m_N        = 14.007 * self.amu
        self.m_O        = 15.999 * self.amu
        self.m_S        = 32.06  * self.amu
        
        # --- base composition: mass fractions X_* (sum must be 1) ---
        self.X_H2       = 1.0
        self.X_H2O      = 0.0
        self.X_O2       = 0.0
        self.X_CO2      = 0.0
        self.X_CO       = 0.0
        self.X_CH4      = 0.0
        self.X_N2       = 0.0
        self.X_NH3      = 0.0
        self.X_H2S      = 0.0
        self.X_SO2      = 0.0
        self.X_S2       = 0.0
        
        self.auto_normalize_X  = False   # default; set True via config or at runtime
        self._norm_warned_once = False  # to avoid spamming messages
        
        # ------------------------------
        # Region A: bolometric (non-dissociated) mean molecular weights
        # ------------------------------
        # Not fractionated in the escape network, but contributes to μ in the
        # sub-R_XUV, bolometrically heated region.
        self.mmw_H2     = 2.0*self.am_h                 # 2
        self.mmw_H2O    = 2.0*self.am_h + self.am_o     # 18
        self.mmw_O2     = 2.0*self.am_o                 # 32
        self.mmw_CO2    = self.am_c + 2.0*self.am_o     # 44
        self.mmw_CO     = self.am_c + self.am_o         # 28
        self.mmw_CH4    = self.am_c + 4.0*self.am_h     # 16
        self.mmw_N2     = 2.0*self.am_n                 # 28
        self.mmw_NH3    = self.am_n + 3.0*self.am_h     # 17
        self.mmw_H2S    = 2.0*self.am_h + self.am_s     # 34
        self.mmw_SO2    = self.am_s + 2.0*self.am_o     # 64
        self.mmw_S2     = 2.0*self.am_s                 # 64
        
        # --- opacities in the IR (cm2 g-1); coarse 1-30 um Planck-mean-ish defaults ---
        self.kappa = {'H2': 1e-2, 'H2O': 1.0, 'O2': 2e-2, 'CO2': 5e-1,
                      'CO': 1e-1, 'CH4': 5e-1, 'N2': 1e-2, 'NH3': 5e-1,
                      'H2S': 8e-1, 'SO2': 1.0, 'S2': 2e-1}

        # ------------------------------
        # Region B: outflow (fully dissociated) mean molecular weights
        # ------------------------------
        # “Outflow” (fully dissociated) per-atom μ for reservoir bookkeeping
        # (mean mass per atom from each molecular reservoir; m_H units)
        self.mmw_H2_outflow  = (2.0*self.am_h)/2.0             # 1
        self.mmw_H2O_outflow = (2.0*self.am_h+self.am_o)/3.0   # 6
        self.mmw_O2_outflow  = (2.0*self.am_o)/2.0             # 16
        self.mmw_CO2_outflow = (self.am_c+2.0*self.am_o)/3.0   # 44/3
        self.mmw_CO_outflow  = (self.am_c+self.am_o)/2.0       # 14
        self.mmw_CH4_outflow = (self.am_c+4.0*self.am_h)/5.0   # 16/5
        self.mmw_N2_outflow  = (2.0*self.am_n)/2.0             # 14
        self.mmw_NH3_outflow = (self.am_n+3.0*self.am_h)/4.0   # 17/4
        self.mmw_H2S_outflow = (2.0*self.am_h+self.am_s)/3.0   # 34/3
        self.mmw_SO2_outflow = (self.am_s+2.0*self.am_o)/3.0   # 64/3
        self.mmw_S2_outflow  = (2.0*self.am_s)/2               # 32
        
        # --- compute composites & opacities ---
        self._recompute_composites()
        self._init_opacities()
        
        # --- default diffusion fits b_ij(T) = A*T**gamma (cm^-1 s^-1) ---
        self.diffusion_fits = {
            # H-O: Zahnle & Kasting (1986). The only pair with an independent
            # literature value, and the one the fractionation hinges on.
            "HO": (4.8e17, 0.75),

            # INACTIVE -- rigid sphere (Banks & Kockarts, "Aeronomy" 1973).
            # Atoms treated as billiard balls of one universal diameter (~3 A), so
            # the only thing separating one pair from another is the reduced mass:
            #     b_ij(T) = 1.52e18 * sqrt(1/m_i + 1/m_j) * T**0.5     (m in amu)
            # The exponent 0.5 is exact for this model, not a fit: mean relative
            # speed goes as sqrt(T) and the cross-section is constant. Each A below
            # is just that prefactor with the mass factor folded in.
            # Retired because this is B&K's fallback for pairs they do not tabulate
            # and it is too shallow at high T: against the H-O anchor it runs a
            # factor ~2 low at 3 kK and ~3 low at 10 kK.
            # "HC": (1.577e18, 0.5),
            # "HN": (1.569e18, 0.5),
            # "HS": (1.539e18, 0.5),
            # "OC": (5.807e17, 0.5),
            # "ON": (5.566e17, 0.5),
            # "OS": (4.656e17, 0.5),
            # "CN": (5.981e17, 0.5),
            # "CS": (5.146e17, 0.5),
            # "NS": (4.872e17, 0.5)

            # ACTIVE SET -- Chapman-Enskog, Lennard-Jones 12-6 potential.
            # Atoms are soft rather than hard: they attract at long range and repel
            # at short range, so the effective cross-section shrinks as collisions
            # get more energetic. The exact expression is NOT a power law:
            #     b_ij(T) = (3/16)*sqrt(2*pi*k*T/mu_ij) / (pi*sigma_ij**2*Omega(T*))
            # with Omega the collision integral and T* = kT/eps. The values below are
            # least-squares power-law fits to that curve over 1000-15000 K, where
            # Omega is deep in its high-T* limit and the curve really is a power law
            # with gamma = 0.5 + 0.1561 = 0.6561. The S pairs sit a little above that
            # because Svehla's eps/k = 847 K for sulphur still bites at 1000 K.
            # sigma/eps are Svehla (1962) atomic values. Regenerate with
            # tools/gen_diffusion.py, which also prints the fit residuals.
            # This set is 1.5-2.2x the rigid-sphere one at 3-10 kK and agrees with
            # the independent H-O anchor to within 30% across 300-15000 K, so it is
            # the better-supported set in BOREAS's temperature range.
            # H-O is deliberately NOT taken from here: the measured Zahnle & Kasting
            # value above is kept for continuity with the escape literature. The two
            # agree to within 30%, so the set stays internally consistent -- b_HO sits
            # 1.1-1.4x above b_HC, against 1.7-3.4x under the rigid-sphere set.
            # "HO": (8.350e17, 0.6561),   # C-E value, superseded by ZK86 above
            "HC": (8.303e17, 0.6561),
            "HN": (7.954e17, 0.6561),
            "HS": (5.261e17, 0.6593),
            "OC": (2.523e17, 0.6561),
            "ON": (2.323e17, 0.6562),
            "OS": (1.212e17, 0.6692),
            "CN": (2.486e17, 0.6561),
            "CS": (1.478e17, 0.6585),
            "NS": (1.274e17, 0.6643)
        }
        
        self._warned_pairs = set()
        
        # --- eddy mixing / homopause controls ---
        # Homopause: the level where eddy mixing (Kzz) equals molecular diffusion (Dzz).
        # Below it the atmosphere is well mixed (one scale height for everything); above it
        # species separate diffusively, which is what the fractionation network assumes.
        # Purely diagnostic: on by default so every run carries the validity flag, but it
        # only reports numbers. Nothing downstream (RXUV, Mdot, fractionation) ever uses
        # the homopause. Set False to drop the homopause keys from the output entirely.
        self.use_homopause = True
        self.Kzz = 1.0e8                        # eddy diffusion coefficient, cm^2 s^-1
        # Dzz(T, n_tot) = A * T**gamma / n_tot, fitted for CH4 in an H2 background and
        # rescaled to other species by reduced mass (see _homopause_mass_factor).
        self.D_molecular_fit = (2.2965e17, 0.765)
        
        self.atomic_y_xuv = None   # optional dict of atomic number fractions at RXUV
                
    # =================================================
    # Basic helpers
    # =================================================
    
    # --- composition ---
    def get_X_tuple(self):
        return (self.X_H2, self.X_H2O, self.X_O2, self.X_CO2, self.X_CO, self.X_CH4, 
                self.X_N2, self.X_NH3, self.X_H2S, self.X_SO2, self.X_S2)

    def _check_X_sum(self, tol=1e-8):
        s = self._sum_X()
        if abs(s - 1.0) > tol:
            if self.auto_normalize_X:
                self._normalize_X_inplace()
                return
            raise ValueError(f"X fractions must sum to 1 (got {s:.5f}).")

    def update_param(self, key, value):
        setattr(self, key, value)
        if key.startswith('X_') or key in ('FXUV',):
            self._recompute_composites()
            self._init_opacities()
            self.mmw_outflow_eff = None

    def get_param(self, key, default=None):
        return getattr(self, key, default)

    def fxuv_incident(self):
        """User/input XUV energy flux at the planet's orbit [erg cm^-2 s^-1]."""
        return float(self.FXUV)

    def fxuv_global_mean(self):
        """Planet-mean XUV energy flux, i.e. incident flux divided by 4 for EL bookkeeping."""
        return 0.25 * self.fxuv_incident()

    def fxuv_photon_incident(self):
        """Incident stellar XUV photon flux at the planet's orbit [photons cm^-2 s^-1]."""
        return self.fxuv_incident() / self.E_photon
    
    # --- cross-section and opacity setters ---
    def _init_opacities(self):
        self._check_X_sum()
        X_H2, X_H2O, X_O2, X_CO2, X_CO, X_CH4, X_N2, X_NH3, X_H2S, X_SO2, X_S2 = self.get_X_tuple()
        self.kappa_p_all = (
            X_H2 * self.kappa['H2']   + X_H2O * self.kappa['H2O'] +
            X_O2 * self.kappa['O2']   + X_CO2 * self.kappa['CO2'] +
            X_CO * self.kappa['CO']   + X_CH4 * self.kappa['CH4'] +
            X_N2 * self.kappa['N2']   + X_NH3 * self.kappa['NH3'] +
            X_H2S * self.kappa['H2S'] + X_SO2 * self.kappa['SO2'] +
            X_S2 * self.kappa['S2']
        )
        
    def xuv_cross_section_per_mass(self):
        """
        This is the mass absorption coefficient in XUV (units cm2 g-1), not the microscopic cross section sigma (cm2).
        Return χ_XUV = (Σ n_s sigma_s) / rho  [cm^2 g^-1] at the XUV base, assuming full dissociation in the outflow region.
        OR     χ_XUV =  Σ (sigma_atom * atoms per gram); units: cm^2 g^-1.
        Uses reservoirs and their outflow mu to count atoms per gram.
        
        IMPORTANT: This function implicitly assumes all absorbers are neutral. 
        Near the base, hydrogen may be partly ionized in RL conditions. We ignore these cases.
        """
        
        # -----------------------
        # 1) Atomic override path
        # -----------------------
        # --- If atomic mixture at RXUV is provided, use it (fully dissociated) ---
        y = getattr(self, "atomic_y_xuv", None)
        mu_eff = getattr(self, "mmw_outflow_eff", None)

        if y is not None and mu_eff is not None and mu_eff > 0:
            # normalize defensively
            s = sum(max(v, 0.0) for v in y.values())
            if s <= 0:
                raise ValueError("atomic_y_xuv provided but sums to 0.")
            yN = {k: max(v, 0.0)/s for k, v in y.items()}

            # chi = Σ sigma_i * y_i / (mu m_H)
            chi = 0.0
            for sp in ("H","C","N","O","S"):
                chi += self.sigma_XUV[sp] * yN.get(sp, 0.0)
            chi /= (mu_eff * self.m_H)
            
            return chi # cm^2 g^-1
            
        # ------------------------------------
        # 2) Reservoir fallback
        # ------------------------------------
        (X_H2, X_H2O, X_O2, X_CO2, X_CO, X_CH4, X_N2, X_NH3, X_H2S, X_SO2, X_S2) = self.get_X_tuple()

        chi = 0.0
        # H-bearing reservoirs → H atoms
        if X_H2  > 0: chi += self.sigma_XUV['H'] * X_H2  / (self.mmw_H2_outflow  * self.m_H)
        if X_H2O > 0: chi += self.sigma_XUV['H'] * X_H2O / (self.mmw_H2O_outflow * self.m_H) * 2.0/3.0 # 2 of 3 atoms are H
        if X_CH4 > 0: chi += self.sigma_XUV['H'] * X_CH4 / (self.mmw_CH4_outflow * self.m_H) * 4.0/5.0
        if X_NH3 > 0: chi += self.sigma_XUV['H'] * X_NH3 / (self.mmw_NH3_outflow * self.m_H) * 3.0/4.0
        if X_H2S > 0: chi += self.sigma_XUV['H'] * X_H2S / (self.mmw_H2S_outflow * self.m_H) * 2.0/3.0

        # O-bearing reservoirs → O atoms
        if X_H2O > 0: chi += self.sigma_XUV['O'] * X_H2O / (self.mmw_H2O_outflow * self.m_H) * 1.0/3.0
        if X_O2  > 0: chi += self.sigma_XUV['O'] * X_O2  / (self.mmw_O2_outflow  * self.m_H) * 1.0     # 2/2
        if X_CO2 > 0: chi += self.sigma_XUV['O'] * X_CO2 / (self.mmw_CO2_outflow * self.m_H) * 2.0/3.0
        if X_CO  > 0: chi += self.sigma_XUV['O'] * X_CO  / (self.mmw_CO_outflow  * self.m_H) * 1.0/2.0
        if X_SO2 > 0: chi += self.sigma_XUV['O'] * X_SO2 / (self.mmw_SO2_outflow * self.m_H) * 2.0/3.0

        # C-bearing reservoirs → C atoms
        if X_CO2 > 0: chi += self.sigma_XUV['C'] * X_CO2 / (self.mmw_CO2_outflow * self.m_H) * 1.0/3.0
        if X_CO  > 0: chi += self.sigma_XUV['C'] * X_CO  / (self.mmw_CO_outflow  * self.m_H) * 1.0/2.0
        if X_CH4 > 0: chi += self.sigma_XUV['C'] * X_CH4 / (self.mmw_CH4_outflow * self.m_H) * 1.0/5.0

        # N-bearing reservoirs → N atoms
        if X_N2  > 0: chi += self.sigma_XUV['N'] * X_N2  / (self.mmw_N2_outflow  * self.m_H) * 1.0     # 2/2
        if X_NH3 > 0: chi += self.sigma_XUV['N'] * X_NH3 / (self.mmw_NH3_outflow * self.m_H) * 1.0/4.0

        # S-bearing reservoirs → S atoms
        if X_H2S > 0: chi += self.sigma_XUV['S'] * X_H2S / (self.mmw_H2S_outflow * self.m_H) * 1.0/3.0
        if X_SO2 > 0: chi += self.sigma_XUV['S'] * X_SO2 / (self.mmw_SO2_outflow * self.m_H) * 1.0/3.0
        if X_S2  > 0: chi += self.sigma_XUV['S'] * X_S2  / (self.mmw_S2_outflow  * self.m_H) * 1.0     # 2/2

        return chi # cm^2 g^-1
    
    def set_sigma_XUV(self, mapping: dict):
        """Override atomic sigma_XUV (cm^2). Keys case-insensitive among H,C,N,O,S."""
        for k, v in mapping.items():
            key = k.upper()
            if key not in self.sigma_XUV:
                raise KeyError(f"Unknown sigma_XUV species '{k}'. Valid: {list(self.sigma_XUV)}")
            self.sigma_XUV[key] = float(v)

    def set_kappa(self, mapping: dict):
        """Override IR κ (cm^2 g^-1) per molecule."""
        for mol, val in mapping.items():
            if mol not in self.kappa:
                raise KeyError(f"Unknown κ species '{mol}'. Valid: {list(self.kappa)}")
            self.kappa[mol] = float(val)
        self._init_opacities()

    # --- mu (bolometric) & reservoir bookkeeping ---
    def _recompute_composites(self):
        X_H2, X_H2O, X_O2, X_CO2, X_CO, X_CH4, X_N2, X_NH3, X_H2S, X_SO2, X_S2 = self.get_X_tuple()
        self._check_X_sum()
        self.mmw_bolometric_all = (
            X_H2  * self.mmw_H2  + X_H2O * self.mmw_H2O + X_O2  * self.mmw_O2 +     # O and H species
            X_CO2 * self.mmw_CO2 + X_CO  * self.mmw_CO  + X_CH4 * self.mmw_CH4 +    # C species
            X_N2  * self.mmw_N2  + X_NH3 * self.mmw_NH3 +                           # N species
            X_H2S * self.mmw_H2S + X_SO2 * self.mmw_SO2 + X_S2  * self.mmw_S2       # S species
        )

    def get_mmw_bolometric(self):
        return self.mmw_bolometric_all

    def mixing_ratios_H_O_C_N_S(self, *X):
        # Atomic mixing ratios at the base of the flow (fully dissociated,
        # per H atom), not volume mixing ratios of intact molecules.
        (X_H2, X_H2O, X_O2, X_CO2, X_CO, X_CH4, X_N2, X_NH3, X_H2S, X_SO2, X_S2) = X
        
        # particle numbers per unit bulk mass from each reservoir
        # OR atoms per bulk mass from each reservoir (up to a constant 1/m_H)
        N_H2  = X_H2  / self.mmw_H2_outflow   if X_H2  > 0 else 0.0
        N_H2O = X_H2O / self.mmw_H2O_outflow  if X_H2O > 0 else 0.0
        N_O2  = X_O2  / self.mmw_O2_outflow   if X_O2  > 0 else 0.0
        N_CO2 = X_CO2 / self.mmw_CO2_outflow  if X_CO2 > 0 else 0.0
        N_CO  = X_CO  / self.mmw_CO_outflow   if X_CO  > 0 else 0.0
        N_CH4 = X_CH4 / self.mmw_CH4_outflow  if X_CH4 > 0 else 0.0
        N_N2  = X_N2  / self.mmw_N2_outflow   if X_N2  > 0 else 0.0
        N_NH3 = X_NH3 / self.mmw_NH3_outflow  if X_NH3 > 0 else 0.0
        N_H2S = X_H2S / self.mmw_H2S_outflow  if X_H2S > 0 else 0.0
        N_SO2 = X_SO2 / self.mmw_SO2_outflow  if X_SO2 > 0 else 0.0
        N_S2  = X_S2  / self.mmw_S2_outflow   if X_S2  > 0 else 0.0

        # element atom counts per bulk mass
        N_H = (2.0/2.0)*N_H2  + (2.0/3.0)*N_H2O + (4.0/5.0)*N_CH4 + (3.0/4.0)*N_NH3 + (2.0/3.0)*N_H2S
        N_O = (1.0/3.0)*N_H2O + 1.0*N_O2        + (2.0/3.0)*N_CO2 + (1.0/2.0)*N_CO  + (2.0/3.0)*N_SO2
        N_C = (1.0/3.0)*N_CO2 + (1.0/2.0)*N_CO  + (1.0/5.0)*N_CH4
        N_N = 1.0*N_N2        + (1.0/4.0)*N_NH3
        N_S = (1/3)*N_H2S     + (1.0/3.0)*N_SO2 + 1.0*N_S2

        # mixing ratios relative to H from Odert et al. 2018
        if N_H <= 0.0:
            return 0.0, 0.0, 0.0, 0.0
        f_O = N_O / N_H
        f_C = N_C / N_H
        f_N = N_N / N_H
        f_S = N_S / N_H
        return f_O, f_C, f_N, f_S

    def outflow_from_X(self, *X):
        (X_H2, X_H2O, X_O2, X_CO2, X_CO, X_CH4, X_N2, X_NH3, X_H2S, X_SO2, X_S2) = X
        
        N_H2  = X_H2  / self.mmw_H2_outflow   if X_H2  > 0 else 0.0
        N_H2O = X_H2O / self.mmw_H2O_outflow  if X_H2O > 0 else 0.0
        N_O2  = X_O2  / self.mmw_O2_outflow   if X_O2  > 0 else 0.0
        N_CO2 = X_CO2 / self.mmw_CO2_outflow  if X_CO2 > 0 else 0.0
        N_CO  = X_CO  / self.mmw_CO_outflow   if X_CO  > 0 else 0.0
        N_CH4 = X_CH4 / self.mmw_CH4_outflow  if X_CH4 > 0 else 0.0
        N_N2  = X_N2  / self.mmw_N2_outflow   if X_N2  > 0 else 0.0
        N_NH3 = X_NH3 / self.mmw_NH3_outflow  if X_NH3 > 0 else 0.0
        N_H2S = X_H2S / self.mmw_H2S_outflow  if X_H2S > 0 else 0.0
        N_SO2 = X_SO2 / self.mmw_SO2_outflow  if X_SO2 > 0 else 0.0
        N_S2  = X_S2  / self.mmw_S2_outflow   if X_S2  > 0 else 0.0
        
        N_H = (2.0/2.0)*N_H2  + (2.0/3.0)*N_H2O + (4.0/5.0)*N_CH4 + (3.0/4.0)*N_NH3 + (2.0/3.0)*N_H2S
        N_O = (1.0/3.0)*N_H2O + 1.0*N_O2        + (2.0/3.0)*N_CO2 + (1.0/2.0)*N_CO  + (2.0/3.0)*N_SO2
        N_C = (1.0/3.0)*N_CO2 + (1.0/2.0)*N_CO  + (1.0/5.0)*N_CH4
        N_N = (2.0/2.0)*N_N2  + (1.0/4.0)*N_NH3
        N_S = (1/3)*N_H2S     + (1.0/3.0)*N_SO2 + 1.0*N_S2
        
        N_tot = N_H + N_O + N_C + N_N + N_S
        
        if N_tot <= 0.0:
            return 1.0
        mean_mass = (self.m_H*N_H + self.m_O*N_O + self.m_C*N_C + self.m_N*N_N + self.m_S*N_S) / N_tot
        return mean_mass / self.m_H

    def get_mu_outflow_current(self):
        return self.outflow_from_X(*self.get_X_tuple())

    # =================================================
    # Binary diffusion coefficients b_ij(T)  [cm^-1 s^-1]
    # =================================================
    # b_ij is the "binary diffusion parameter", b_ij = n * D_ij, where n is the total
    # number density and D_ij the binary diffusion coefficient. Multiplying by n is what
    # removes the density dependence (D_ij falls as 1/n, because a denser gas impedes
    # diffusion), leaving something that depends on temperature alone. It measures how
    # readily species i and j slide past each other, and in the fractionation network it
    # sets the critical flux F_crit ~ g*(m_j - m_i)*b_ij / (k*T), i.e. how hard the
    # escaping light species has to blow to drag the heavy one along.
    #
    # These methods are a LEGACY fallback. b_pair() consults self.diffusion_fits first,
    # and that table covers every pair, so in practice nothing reaches here. Edit the
    # table, not these; they are kept only so that older configs keep working.
    #
    # Two physical pictures are on offer, and they disagree by a factor of ~2 at 3 kK:
    #
    #   RIGID SPHERE (Banks & Kockarts, "Aeronomy" 1973) -- active set.
    #   Atoms are billiard balls of one universal diameter (~3 A). Nothing distinguishes
    #   one pair from another except how fast the two partners move relative to each
    #   other, which is fixed by the reduced mass:
    #       b_ij(T) = 1.52e18 * sqrt(1/m_i + 1/m_j) * T**0.5      (m in amu)
    #   The exponent 0.5 is exact for this model rather than fitted: mean relative speed
    #   goes as sqrt(T) and the cross-section never changes. Simple and parameter-free,
    #   but it is B&K's fallback for pairs they do not tabulate and it is too shallow at
    #   high T -- a factor ~2 below the H-O anchor at 3 kK, ~3 below it at 10 kK.
    #
    #   CHAPMAN-ENSKOG, Lennard-Jones 12-6 (see diffusion_fits for the values).
    #   Atoms are soft: they attract at long range and repel at short range, so a faster
    #   collision digs deeper into the repulsive core and the effective cross-section
    #   shrinks with temperature. That extra shrinkage is why b rises faster than sqrt(T).
    #   The exact expression is not a power law; the stored values are fits to it over
    #   1000-15000 K. Regenerate with tools/gen_diffusion.py.
    #
    # H-O is the one pair with an independent literature value (Zahnle & Kasting 1986),
    # so it doubles as the calibration check. Against it, Chapman-Enskog agrees to within
    # 30% over 300-15000 K while the rigid-sphere form drifts to a factor 3.4 low. Keeping
    # ZK's H-O alongside rigid-sphere values for the other nine therefore leaves H-O
    # sitting 1.7-3.4x above its neighbours, which biases O relative to C/N/S in the
    # fractionation. Switching the whole set to Chapman-Enskog shrinks that to 1.1-1.4x.

    # Kept in step with diffusion_fits above so the two can never disagree.
    def b_HO(self, T):  return 4.8e17   * (T**0.75)    # Zahnle & Kasting 1986

    def b_HC(self, T):  return 8.303e17 * (T**0.6561)  # Chapman-Enskog, LJ 12-6
    def b_HN(self, T):  return 7.954e17 * (T**0.6561)  # Chapman-Enskog, LJ 12-6
    def b_HS(self, T):  return 5.261e17 * (T**0.6593)  # Chapman-Enskog, LJ 12-6

    def b_OC(self, T):  return 2.523e17 * (T**0.6561)  # Chapman-Enskog, LJ 12-6
    def b_ON(self, T):  return 2.323e17 * (T**0.6562)  # Chapman-Enskog, LJ 12-6
    def b_OS(self, T):  return 1.212e17 * (T**0.6692)  # Chapman-Enskog, LJ 12-6

    def b_CN(self, T):  return 2.486e17 * (T**0.6561)  # Chapman-Enskog, LJ 12-6
    def b_CS(self, T):  return 1.478e17 * (T**0.6585)  # Chapman-Enskog, LJ 12-6

    def b_NS(self, T):  return 1.274e17 * (T**0.6643)  # Chapman-Enskog, LJ 12-6
    
    # map species keys to masses (g) and atomic masses (amu-like counts)
    def species_registry(self):
        return {
            'H': {'m': self.m_H, 'A': self.am_h},
            'O': {'m': self.m_O, 'A': self.am_o},
            'C': {'m': self.m_C, 'A': self.am_c},
            'N': {'m': self.m_N, 'A': self.am_n},
            'S': {'m': self.m_S, 'A': self.am_s},
        }

    def b_pair(self, a, b, T):
        """Return b_ij(T) (cm^-1 s^-1) using, in order: user fits, built-ins, symmetry, or geometric mean fallback."""
        a = a.upper(); b = b.upper()
        if a == b:
            return 1e40 # effectively "infinite" to avoid division by ~0 in ratios

        # user/builtin fits table. The table is written light-species-first ("HC", "OC",
        # "ON"), which is not the same as alphabetical, so look the pair up under both
        # spellings instead of trusting one convention: sorting alone missed H-C, C-O and
        # N-O, which then fell through to the legacy methods below.
        k = "".join(sorted([a, b]))
        fit = self.diffusion_fits.get(a + b, self.diffusion_fits.get(b + a))
        if fit:
            A, gamma = fit
            return A * (T ** gamma)

        # legacy hard-coded methods
        name = f"b_{a}{b}"
        if hasattr(self, name):
            return getattr(self, name)(T)
        name_sym = f"b_{b}{a}"
        if hasattr(self, name_sym):
            return getattr(self, name_sym)(T)

        # geometric-mean fallback via H
        try:
            b_aH = self.b_pair(a, 'H', T)
            b_bH = self.b_pair(b, 'H', T)
            if k not in self._warned_pairs:
                print(f"[b_pair] Using geometric-mean fallback for {a}-{b}")
                self._warned_pairs.add(k)
            return (b_aH * b_bH) ** 0.5
        except Exception:
            raise NotImplementedError(f"No diffusion coefficient for pair {a}-{b}. Add it to diffusion_fits or implement b_{a}{b}.")

    # --- molecular diffusion & the homopause ---
    # The Dzz fit is calibrated on CH4 diffusing through H2; every other pair is reached
    # by rescaling with the reduced mass, which is the only species information it needs.
    _HOMOPAUSE_MOLECULES = ["H2", "H2O", "O2", "CO2", "CO", "CH4", "N2", "NH3", "H2S", "SO2", "S2"]
    _MMW_H2_FIT   = 2.016                           # H2 as used in the published fit
    _MU_RED_CALIB = 16.04 * 2.016 / 18.059          # the fit's own CH4-H2 reduced mass

    @staticmethod
    def _reduced_mass(mmw_a, mmw_b):
        return mmw_a * mmw_b / (mmw_a + mmw_b)

    def _homopause_mass_factor(self, mmw_i, mmw_bg=None):
        """
        Reduced-mass rescaling of the CH4-in-H2 diffusion coefficient to the pair (i, bg),
        i.e. sqrt(mu_red(CH4,H2) / mu_red(i,bg)). Dimensionless, 1 for the calibration pair.

        With mmw_bg = 2.016 this reduces exactly to the published H2-background form
        sqrt( 16.04/mmw_i * (mmw_i + 2.016)/18.059 ). The calibration reduced mass keeps
        the published 18.059 denominator rather than 16.04 + 2.016, so the H2-background
        case is reproduced bit for bit.
        """
        if mmw_bg is None:
            mmw_bg = self._MMW_H2_FIT
        return (self._MU_RED_CALIB / self._reduced_mass(mmw_i, mmw_bg)) ** 0.5

    def D_molecular(self, T, n_tot, mmw_i, mmw_bg=None):
        """
        Binary molecular diffusion coefficient Dzz of molecule i through a background
        of molecular weight mmw_bg (cm^2 s^-1). mmw_i is the molecular weight of the
        diffusing species (e.g. 18 for H2O); n_tot is the total number density (cm^-3),
        i.e. P/(k_B T) for an ideal gas. mmw_bg defaults to H2 (the fit's own background).

        NOTE: only the reduced mass is rescaled. The prefactor still carries the
        CH4-H2 collision cross-section, so this stays an estimate for other pairs.
        It depends on the pair only weakly; the strong dependence is the 1/n_tot one.
        """
        if mmw_bg is None:
            mmw_bg = self._MMW_H2_FIT
        A, gamma = self.D_molecular_fit
        return A * (T ** gamma) / n_tot * self._homopause_mass_factor(mmw_i, mmw_bg)

    def n_homopause(self, T, mmw_i=None, mmw_bg=None):
        """
        Total number density at the homopause (cm^-3), from Kzz = Dzz(T, n_tot).
        Since Dzz ~ 1/n_tot this inverts directly, no profile needed.
        """
        if mmw_i is None:
            _, mmw_i = self.homopause_molecule()
        if mmw_bg is None:
            mmw_bg = self._MMW_H2_FIT
        A, gamma = self.D_molecular_fit
        return A * (T ** gamma) * self._homopause_mass_factor(mmw_i, mmw_bg) / self.Kzz

    def homopause_molecule(self, eps=1e-12):
        """
        Molecule whose Dzz is evaluated at the homopause: the most abundant one present
        in the bolometric region. Returns (name, mmw) in hydrogen-atom-mass units.

        Not a user choice. The homopause depends on the species only through the reduced
        mass, which across the whole H2-to-SO2 range moves it by well under 1% of Rp --
        far less than the orders of magnitude of uncertainty in Kzz.
        """
        mmw = {n: getattr(self, f"mmw_{n}") for n in self._HOMOPAUSE_MOLECULES}
        X = dict(zip(self._HOMOPAUSE_MOLECULES, self.get_X_tuple()))
        present = [n for n in self._HOMOPAUSE_MOLECULES if X[n] > eps]
        if not present:
            raise ValueError("No molecule present in the bolometric composition.")
        name = max(present, key=lambda n: X[n])
        return name, mmw[name]

    # --- other helpers ---   
    # to properly read the configs/*.toml files
    def set_composition(self, mapping: dict, auto_normalize: bool = True):
        """
        Set all mass fractions X_* in one shot.
        mapping keys: H2, H2O, O2, CO2, CO, CH4, N2, NH3, H2S, SO2, S2
        Unspecified species default to 0.0.
        If auto_normalize=True, values are rescaled to sum to 1.
        If auto_normalize is None, use self.auto_normalize_X.
        """
        if auto_normalize is None:
            auto_normalize = self.auto_normalize_X
        
        allowed = ["H2","H2O","O2","CO2","CO","CH4","N2","NH3","H2S","SO2","S2"]

        # collect values, defaulting missing ones to 0
        Xvals = {f"X_{sp}": float(mapping.get(sp, 0.0)) for sp in allowed}
        s = sum(Xvals.values())

        if auto_normalize:
            if s <= 0.0:
                raise ValueError("All composition mass fractions are zero.")
            scale = 1.0 / s
        else:
            if abs(s - 1.0) > 1e-8:
                raise ValueError(f"X fractions must sum to 1 (got {s:.5f}).")
            scale = 1.0

        # assign without triggering per-key recompute
        for key, val in Xvals.items():
            setattr(self, key, val * scale)

        # now recompute once
        self._recompute_composites()
        self._init_opacities()
        self.mmw_outflow_eff = None
            
    # to normalize mass fractions automatically whenever they don’t sum to 1
    def _sum_X(self):
        return sum(self.get_X_tuple())

    def _normalize_X_inplace(self):
        s = self._sum_X()
        if s <= 0.0:
            raise ValueError("All composition mass fractions are zero; cannot normalize.")
        scale = 1.0 / s
        (self.X_H2, self.X_H2O, self.X_O2, self.X_CO2, self.X_CO, self.X_CH4,
        self.X_N2, self.X_NH3, self.X_H2S, self.X_SO2, self.X_S2) = [
            x * scale for x in self.get_X_tuple()
        ]
        if not self._norm_warned_once:
            print(f"[composition] Auto-normalized mass fractions (sum={s:.6f} → 1.0).")
            self._norm_warned_once = True

    def enable_auto_normalize(self, flag: bool = True):
        """Enable/disable auto-normalization for X_* when their sum != 1."""
        self.auto_normalize_X = bool(flag)

    def set_diffusion_fits(self, mapping: dict):
        """
        mapping: {"HO": {"A":..., "gamma":...}, "H-O": {...}, ...}
        Keys can be 'HO', 'OH', 'H-O', or 'O-H'. Stored as unhyphenated, orderless.
        """
        def norm_key(key: str) -> str:
            key = key.replace("-", "").upper()
            if len(key) != 2:
                raise ValueError(f"Diffusion key '{key}' must be a 2-letter pair like 'HO'.")
            a, b = key[0], key[1]
            if a == b:
                raise ValueError("Self-diffusion pairs (e.g., 'HH') are not valid here.")
            # store unordered to auto-symmetrize
            return "".join(sorted([a, b]))
        for k, spec in mapping.items():
            A = float(spec["A"]); gamma = float(spec["gamma"])
            kN = norm_key(k)
            self.diffusion_fits[kN] = (A, gamma)
