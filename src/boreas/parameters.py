class ModelParams:
    """
    Base class for handling model parameters and physical constants.
    Supports preset mixing modes for H2, H2O, CO2, CH4 combinations.
    """
    def __init__(self):
        # --- default mode controls ---
        # Example stellar/XUV controls (set/overridden by main or star class)
        self.FXUV       = 450.0              # erg cm^-2 s^-1 (placeholder default)

        # other model parameters and constants
        self.albedo     = 0.3             # albedo of planet
        self.beta       = 0.75            # fraction of the planet's surface that re-emits radiation
        self.epsilon    = 1.              # emissivity of planet
        self.alpha_rec  = 2.6e-13         # Recombination coefficient, cm3 s-1
        self.eff        = 0.3             # Mass-loss efficiency factor
        self.aplau      = 1.              # semi-major axis of the planet
        
        # self.sigma_XUV  = 1.89e-18      # atomic XUV cross-section (of H), cm2
        self.sigma_XUV = {'H':  1.89e-18,
                          'O':  2.00e-18,  # placeholder
                          'C':  2.50e-18,  # placeholder
                          'N':  3.00e-18,  # placeholder
                          'S':  6.00e-18,  # placeholder
                        }

        # --- physical constants ---
        # Universal constants
        self.G          = 6.67430e-8      # Gravitational constant, cm^3 g^-1 s^-2-1
        self.mearth     = 5.972e27        # earth mass, grams
        self.rearth     = 6.371e8         # earth radius, cm
        self.E_photon   = 20 * 1.6e-12    # photon energy
        self.k_b        = 1.380649e-16    # Boltzmann constant, erg K^
        # Atomic masses (in units of hydrogen-atom mass counts)
        self.am_h       = 1.0
        self.am_o       = 16.0
        self.am_c       = 12.0
        self.am_n       = 14.0
        self.am_s       = 32.0
        # Particle masses (grams)
        self.m_H        = 1.6735575e-24 # g
        self.m_O        = self.am_o * self.m_H
        self.m_C        = self.am_c * self.m_H
        self.m_N        = self.am_n * self.m_H
        self.m_S        = self.am_s * self.m_H
        
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
        
        # --- opacities in the IR (cm2 g-1) (placeholders) ---
        self.kappa = {'H2': 1e-2, 'H2O': 1.0, 'O2': 1.0, 'CO2': 1.0,
                      'CO': 1.0, 'CH4': 1.0, 'N2': 1.0, 'NH3': 1.0,
                      'H2S': 1.0, 'SO2': 1.0, 'S2': 1.0}

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
        # these mirror the current hard-coded functions.
        self.diffusion_fits = {
            "HO": (4.8e17, 0.75),
            "HC": (6.5e17, 0.70),
            "HN": (5.0e17, 0.73),
            "HS": (5.8e17, 0.70),
            "ON": (9.0e16, 0.78),
            "OS": (8.5e16, 0.78),
            # You can include more; symmetry is handled in b_pair
        }
        
        self._warned_pairs = set()
        
        #TODO: if r0 is added (mesopause, base of the outflow, different from RXUV)
        #  # misc fractionation params
        # self.r0_base = None                 # if None, treat r0 ≡ RXUV for now
        # self.fractionation_T_mode = "base"  # "base" | "from_cs" | "fixed"
        # self.fractionation_T_fixed = None
                
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
    
    # --- opacities and cross sections ---
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

        # atomic counts per bulk mass
        N_H = 2.0*N_H2  + 2.0*N_H2O + 4.0*N_CH4 + 3.0*N_NH3 + 2.0*N_H2S
        N_O = 1.0*N_H2O + 2.0*N_O2  + 2.0*N_CO2 + 1.0*N_CO  + 2.0*N_SO2
        N_C = 1.0*N_CO2 + 1.0*N_CO  + 1.0*N_CH4
        N_N = 2.0*N_N2  + 1.0*N_NH3
        N_S = 1.0*N_H2S + 1.0*N_SO2 + 2.0*N_S2

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

        N_H = 2.0*N_H2  + 2.0*N_H2O + 4.0*N_CH4 + 3.0*N_NH3 + 2.0*N_H2S
        N_O = 1.0*N_H2O + 2.0*N_O2  + 2.0*N_CO2 + 1.0*N_CO  + 2.0*N_SO2
        N_C = 1.0*N_CO2 + 1.0*N_CO  + 1.0*N_CH4
        N_N = 2.0*N_N2  + 1.0*N_NH3
        N_S = 1.0*N_H2S + 1.0*N_SO2 + 2.0*N_S2
        N_tot = N_H + N_O + N_C + N_N + N_S
        
        if N_tot <= 0.0:
            return 1.0
        mean_mass = (self.m_H*N_H + self.m_O*N_O + self.m_C*N_C + self.m_N*N_N + self.m_S*N_S) / N_tot
        return mean_mass / self.m_H

    def get_mu_outflow_current(self):
        return self.outflow_from_X(*self.get_X_tuple())

    # --- binary diffusion coefficients b_ij(T) (cm^-1 s^-1) ---
    def b_HO(self, T):  return 4.8e17 * (T**0.75)     # Zahnle/Kasting 1986
    def b_HC(self, T):  return 6.5e17 * (T**0.70)     # placeholder
    def b_HN(self, T):  return 5.0e17 * (T**0.73)     # placeholder
    def b_HS(self, T):  return 5.8e17 * (T**0.70)     # placeholder
    
    def b_OC(self, T):  return (self.b_HO(T) * self.b_HC(T))**0.5 # placeholder
    def b_ON(self, T):  return 9.0e16 * (T**0.78)     # placeholder
    def b_OS(self, T):  return 8.5e16 * (T**0.78)     # placeholder

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

        # user/builtin fits table (unordered key)
        k = "".join(sorted([a, b]))
        fit = self.diffusion_fits.get(k)
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

    # --- others ---   
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

    # 
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
