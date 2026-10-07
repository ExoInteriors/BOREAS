import re

# =================================================
# Species tables
# =================================================
# Atoms in the fully dissociated outflow, lightest first. This is also the order in
# which a diffusion pair is spelled ("HHe", "HO", "CO", ...), see pair_key().
ATOMS = ("H", "He", "C", "N", "O", "S")

# Molecules in the bolometric region, in the order of ModelParams.get_X_tuple().
MOLECULES = ("H2", "He", "H2O", "O2", "CO2", "CO", "CH4", "N2", "NH3", "H2S", "SO2", "S2")

# How many of each atom one molecule releases when fully dissociated.
ATOMS_PER_MOLECULE = {
    "H2":  {"H": 2},
    "He":  {"He": 1},
    "H2O": {"H": 2, "O": 1},
    "O2":  {"O": 2},
    "CO2": {"C": 1, "O": 2},
    "CO":  {"C": 1, "O": 1},
    "CH4": {"C": 1, "H": 4},
    "N2":  {"N": 2},
    "NH3": {"N": 1, "H": 3},
    "H2S": {"H": 2, "S": 1},
    "SO2": {"S": 1, "O": 2},
    "S2":  {"S": 2},
}

_ATOM_BY_UPPER = {a.upper(): a for a in ATOMS}
_ATOM_TOKEN = re.compile(r"[A-Z][a-z]?")

def canonical_atom(name):
    """Case-insensitive atom lookup: 'h' -> 'H', 'HE' -> 'He'. KeyError if not in ATOMS."""
    try:
        return _ATOM_BY_UPPER[name.upper()]
    except KeyError:
        raise KeyError(f"Unknown atom '{name}'. Valid: {list(ATOMS)}") from None

def pair_key(a, b):
    """The one key a pair is stored under in diffusion_fits: lighter atom first ('HHe', 'CO')."""
    a, b = sorted((canonical_atom(a), canonical_atom(b)), key=ATOMS.index)
    return a + b

def split_pair(key):
    """'HO', 'OH', 'H-O', 'HHe', 'He-O', 'OHe' -> the two atom names, in the order given."""
    parts = key.split("-") if "-" in key else _ATOM_TOKEN.findall(key)
    if len(parts) != 2 or ("-" not in key and "".join(parts) != key):
        raise ValueError(f"Diffusion key '{key}' must name two atoms, like 'HO', 'HHe' or 'He-O'.")
    return parts


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
                        # He: its ionization edge is 24.59 eV, so at 20 eV sigma_He is strictly
                        # zero and He is transparent. That would leave a He-dominated outflow
                        # (once H is gone) with nothing to absorb the XUV, while a real spectrum
                        # is absorbed strongly by He above 24.59 eV. So He is taken at its edge,
                        # the closest energy to 20 eV at which it absorbs at all. Verner+1996
                        # He I: E0=13.61 eV, sigma0=949.2 Mb, ya=1.469, P=3.188, yw=2.039,
                        # y0=0.4434, y1=2.136. Set 0.0 for the strict 20 eV picture.
                        'He': 7.43e-18,
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
        self.am_he      = 4.0
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
        self.m_He       = 4.0026 * self.amu     # 6.6465e-24 g
        self.m_C        = 12.011 * self.amu
        self.m_N        = 14.007 * self.amu
        self.m_O        = 15.999 * self.amu
        self.m_S        = 32.06  * self.amu
        
        # --- base composition: mass fractions X_* (sum must be 1) ---
        self.X_H2       = 1.0
        self.X_He       = 0.0
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
        self.mmw_He     = self.am_he                    # 4
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
        # He has no IR bands of its own; its only IR opacity is collision-induced. In an
        # H2-rich gas a gram of He adds ~0.2-0.4x the H2-H2 CIA per gram behind kappa['H2']:
        # half as many He atoms per gram as H2 molecules, and an H2-He pair absorbs
        # ~0.5-1x as much as an H2-H2 pair. He-He CIA is negligible, so a He-dominated
        # gas is more transparent than this value says.
        self.kappa = {'H2': 1e-2, 'He': 3e-3,
                      'H2O': 1.0, 'O2': 2e-2, 'CO2': 5e-1,
                      'CO': 1e-1, 'CH4': 5e-1, 'N2': 1e-2, 'NH3': 5e-1,
                      'H2S': 8e-1, 'SO2': 1.0, 'S2': 2e-1}

        # ------------------------------
        # Region B: outflow (fully dissociated) mean molecular weights
        # ------------------------------
        # “Outflow” (fully dissociated) per-atom μ for reservoir bookkeeping
        # (mean mass per atom from each molecular reservoir; m_H units)
        self.mmw_H2_outflow  = (2.0*self.am_h)/2.0             # 1
        self.mmw_He_outflow  = (self.am_he)/1.0                # 4
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

        # --- He diffusion pairs: two sets of b_ij(T) = A*T**gamma (cm^-1 s^-1) ---
        # "literature" goes into diffusion_fits below; switch with use_he_diffusion_set(name)
        # or `he_set = "chapman-enskog"` under [diffusion] in a config. diffusion_fits only gets
        # a copy, so these stay intact for switching back.
        self.he_diffusion_sets = {
            # LITERATURE. Converted from the papers' m^-1 s^-1 (x 1e-2).
            "literature": {
                # Mason & Marrero (1970), as used by Genda & Ikoma (2008) and Cherubim+ (2024).
                # Measured, so it anchors the He pairs the way ZK86 anchors H-O.
                "HHe": (1.04e18, 0.732),
                # ZK86 H-O rescaled by reduced mass, b ~ mu_ij**-0.5 (Genda & Ikoma 2008
                # prescription, Cherubim+ 2025 App. C), so these share the H-O anchor in diffusion_fits.
                "HeC": (2.64e17, 0.75),     # Cherubim+ 2025
                "HeN": (2.65e17, 0.75),     # same prescription, rescaled from He-O
                "HeO": (2.61e17, 0.75),     # Cherubim+ 2025
                "HeS": (2.48e17, 0.75),     # same prescription, rescaled from He-O
            },
            # CHAPMAN-ENSKOG, LJ 12-6, Svehla (1962) He: sigma = 2.551 A, eps/k = 10.22 K. Fitted
            # over 1000-15000 K like the other pairs in diffusion_fits (tools/gen_diffusion.py).
            "chapman-enskog": {
                "HHe": (1.305e18, 0.6561),
                "HeC": (5.383e17, 0.6561),
                "HeN": (5.096e17, 0.6561),
                "HeO": (5.311e17, 0.6561),
                "HeS": (3.287e17, 0.6563),
            },
        }
        
        # --- default diffusion fits b_ij(T) = A*T**gamma (cm^-1 s^-1) ---
        # Keys are spelled lighter atom first (pair_key); b_pair() and set_diffusion_fits()
        # accept either order, but only the canonical spelling is ever stored.
        self.diffusion_fits = {
            # H-O: Zahnle & Kasting (1986). The only pair with an independent
            # literature value, and the one the fractionation hinges on.
            "HO": (4.8e17, 0.75),

            # INACTIVE -- rigid sphere (Banks & Kockarts, "Aeronomy" 1973).
            # Atoms treated as billiard balls of one universal diameter (~3 A), so
            # the only thing separating one pair from another is the reduced mass:
            #     b_ij(T) = 1.52e18 * sqrt(1/m_i + 1/m_j) * T**0.5     (m in amu)
            # The exponent 0.5 is exact for this model, not a fit
            # "HC": (1.577e18, 0.5),
            # "HN": (1.569e18, 0.5),
            # "HS": (1.539e18, 0.5),
            # "CN": (5.981e17, 0.5),
            # "CO": (5.807e17, 0.5),
            # "CS": (5.146e17, 0.5),
            # "NO": (5.566e17, 0.5),
            # "NS": (4.872e17, 0.5),
            # "OS": (4.656e17, 0.5)

            # ACTIVE -- Chapman-Enskog, Lennard-Jones 12-6 potential.
            # Atoms are soft rather than hard: they attract at long range and repel
            # at short range, so the effective cross-section shrinks as collisions
            # get more energetic. The exact expression is NOT a power law:
            #     b_ij(T) = (3/16)*sqrt(2*pi*k*T/mu_ij) / (pi*sigma_ij**2*Omega(T*))
            # with Omega the collision integral and T* = kT/eps. The values below are
            # least-squares power-law fits to that curve over 1000-15000 K, where
            # Omega is deep in its high-T* limit and the curve really is a power law
            # with gamma = 0.5 + 0.1561 = 0.6561.
            # "HO": (8.350e17, 0.6561), # C-E value, superseded by ZK86 above
            "HC": (8.303e17, 0.6561),
            "HN": (7.954e17, 0.6561),
            "HS": (5.261e17, 0.6593),
            "CN": (2.486e17, 0.6561),
            "CO": (2.523e17, 0.6561),
            "CS": (1.478e17, 0.6585),
            "NO": (2.323e17, 0.6562),
            "NS": (1.274e17, 0.6643),
            "OS": (1.212e17, 0.6692),

            # He pairs: see he_diffusion_sets above.
            **self.he_diffusion_sets["literature"],
        }
        
        self._warned_pairs = set()
        
        # --- eddy mixing / homopause controls ---
        # Homopause: the level where eddy mixing (Kzz) equals molecular diffusion (Dzz).
        # Below it the atmosphere is well mixed (one scale height for everything); above it
        # species separate diffusively, which is what the fractionation network assumes.
        self.use_homopause = True
        self.Kzz = 1.0e8 # eddy diffusion coefficient, cm^2 s^-1
        # Dzz(T, n_tot) = A * T**gamma / n_tot, fitted for CH4 in an H2 background and
        # rescaled to other species by reduced mass (see _homopause_mass_factor).
        self.D_molecular_fit = (2.2965e17, 0.765) # for CH4 in H2, cm^-1 s^-1
        
        self.atomic_y_xuv = None # optional dict of atomic number fractions at RXUV
                
    # =================================================
    # Basic helpers
    # =================================================
    
    # --- composition ---
    def get_X_tuple(self):
        return tuple(getattr(self, f"X_{mol}") for mol in MOLECULES)

    def get_X_dict(self):
        return dict(zip(MOLECULES, self.get_X_tuple(), strict=True))

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
        # mass opacities mix linearly in the mass fractions
        X = self.get_X_dict()
        self.kappa_p_all = sum(X[mol] * self.kappa[mol] for mol in MOLECULES)
        
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
            for sp in ATOMS:
                chi += self.sigma_XUV[sp] * yN.get(sp, 0.0)
            chi /= (mu_eff * self.m_H)

            return chi # cm^2 g^-1

        # ------------------------------------
        # 2) Reservoir fallback
        # ------------------------------------
        # chi = Σ sigma_atom * (atoms per gram), with the atoms counted from each reservoir
        N = self.atomic_counts()
        chi = sum(self.sigma_XUV[sp] * N[sp] for sp in ATOMS) / self.m_H

        return chi # cm^2 g^-1

    def set_sigma_XUV(self, mapping: dict):
        """Override atomic sigma_XUV (cm^2). Keys case-insensitive among H, He, C, N, O, S."""
        for k, v in mapping.items():
            self.sigma_XUV[canonical_atom(k)] = float(v)

    def set_kappa(self, mapping: dict):
        """Override IR κ (cm^2 g^-1) per molecule."""
        for mol, val in mapping.items():
            if mol not in self.kappa:
                raise KeyError(f"Unknown κ species '{mol}'. Valid: {list(self.kappa)}")
            self.kappa[mol] = float(val)
        self._init_opacities()

    # --- mu (bolometric) & reservoir bookkeeping ---
    def _recompute_composites(self):
        self._check_X_sum()
        X = self.get_X_dict()
        # X are mass fractions, so mu is the harmonic mean: 1/mu = sum_k X_k / mu_k
        # (particles per gram), not the mass-weighted sum_k X_k * mu_k.
        self.mmw_bolometric_all = 1.0 / sum(X[mol] / getattr(self, f"mmw_{mol}") for mol in MOLECULES)

    def get_mmw_bolometric(self):
        return self.mmw_bolometric_all

    def atomic_counts(self, X=None):
        """
        Atoms of each element per unit bulk mass at the base of the flow, fully dissociated
        (up to a constant 1/m_H). Returns {atom: N} for every atom in ATOMS. X is a
        composition tuple in MOLECULES order and defaults to the current one.
        """
        X = self.get_X_tuple() if X is None else X
        N = dict.fromkeys(ATOMS, 0.0)
        for mol, X_mol in zip(MOLECULES, X, strict=True):
            if X_mol <= 0.0:
                continue
            # particles per bulk mass from this reservoir, then split by its stoichiometry
            # (e.g. 2 of the 3 atoms in H2O are H)
            N_mol   = X_mol / getattr(self, f"mmw_{mol}_outflow")
            atoms   = ATOMS_PER_MOLECULE[mol]
            n_atoms = sum(atoms.values())
            for atom, k in atoms.items():
                N[atom] += N_mol * k / n_atoms
        return N

    def outflow_from_X(self, *X):
        N = self.atomic_counts(X)
        N_tot = sum(N.values())

        if N_tot <= 0.0:
            return 1.0
        reg = self.species_registry()
        mean_mass = sum(reg[sp]['m'] * N[sp] for sp in ATOMS) / N_tot
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
    # H-O is the one pair with an independent literature value (Zahnle & Kasting 1986),
    # so it doubles as the calibration check. Against it, Chapman-Enskog agrees to within
    # 30% over 300-15000 K while the rigid-sphere form drifts to a factor 3.4 low. Keeping
    # ZK's H-O alongside rigid-sphere values for the other nine therefore leaves H-O
    # sitting 1.7-3.4x above its neighbours, which biases O relative to C/N/S in the
    # fractionation. Switching the whole set to Chapman-Enskog shrinks that to 1.1-1.4x.

    # map species keys to masses (g) and atomic masses (amu-like counts)
    def species_registry(self):
        return {
            'H': {'m': self.m_H, 'A': self.am_h},
            'He': {'m': self.m_He, 'A': self.am_he},
            'O': {'m': self.m_O, 'A': self.am_o},
            'C': {'m': self.m_C, 'A': self.am_c},
            'N': {'m': self.m_N, 'A': self.am_n},
            'S': {'m': self.m_S, 'A': self.am_s},
        }

    def b_pair(self, a, b, T):
        """Return b_ij(T) (cm^-1 s^-1) from diffusion_fits, or a geometric mean via H for a pair missing from it."""
        a, b = canonical_atom(a), canonical_atom(b)
        if a == b:
            return 1e40 # effectively "infinite" to avoid division by ~0 in ratios

        # one canonical spelling per pair (pair_key), so the lookup is order-independent
        k = pair_key(a, b)
        fit = self.diffusion_fits.get(k)
        if fit:
            A, gamma = fit
            return A * (T ** gamma)

        # geometric-mean fallback via H. The default table covers every pair, so this is
        # reached only if a pair was removed from it by hand.
        if 'H' in (a, b):
            raise NotImplementedError(f"No diffusion coefficient for pair {a}-{b}. Add it to diffusion_fits.")
        b_aH = self.b_pair(a, 'H', T)
        b_bH = self.b_pair(b, 'H', T)
        if k not in self._warned_pairs:
            print(f"[b_pair] Using geometric-mean fallback for {a}-{b}")
            self._warned_pairs.add(k)
        return (b_aH * b_bH) ** 0.5

    # --- molecular diffusion & the homopause ---
    # The Dzz fit is calibrated on CH4 diffusing through H2; every other pair is reached
    # by rescaling with the reduced mass, which is the only species information it needs.
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
        mmw = {n: getattr(self, f"mmw_{n}") for n in MOLECULES}
        X = self.get_X_dict()
        present = [n for n in MOLECULES if X[n] > eps]
        if not present:
            raise ValueError("No molecule present in the bolometric composition.")
        name = max(present, key=lambda n: X[n])
        return name, mmw[name]

    # --- other helpers ---   
    # to properly read the configs/*.toml files
    def set_composition(self, mapping: dict, auto_normalize: bool = True):
        """
        Set all mass fractions X_* in one shot.
        mapping keys: any of MOLECULES (H2, He, H2O, O2, CO2, CO, CH4, N2, NH3, H2S, SO2, S2)
        Unspecified species default to 0.0; an unknown key raises, so a typo such as
        'HE' cannot silently drop a species.
        If auto_normalize=True, values are rescaled to sum to 1.
        If auto_normalize is None, use self.auto_normalize_X.
        """
        if auto_normalize is None:
            auto_normalize = self.auto_normalize_X

        unknown = sorted(set(mapping) - set(MOLECULES))
        if unknown:
            raise KeyError(f"Unknown composition species {unknown}. Valid: {list(MOLECULES)}")

        # collect values, defaulting missing ones to 0
        Xvals = {f"X_{sp}": float(mapping.get(sp, 0.0)) for sp in MOLECULES}
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
        for mol, x in self.get_X_dict().items():
            setattr(self, f"X_{mol}", x * scale)
        if not self._norm_warned_once:
            print(f"[composition] Auto-normalized mass fractions (sum={s:.6f} → 1.0).")
            self._norm_warned_once = True

    def enable_auto_normalize(self, flag: bool = True):
        """Enable/disable auto-normalization for X_* when their sum != 1."""
        self.auto_normalize_X = bool(flag)

    def set_diffusion_fits(self, mapping: dict):
        """
        mapping: {"HO": {"A":..., "gamma":...}, "H-He": {...}, ...}
        Keys name two atoms in either order: 'HO', 'OH', 'H-O', 'HHe', 'He-O', 'OHe'.
        Stored under the canonical pair_key, so an override always replaces the default
        rather than sitting next to it under the other spelling.
        """
        for k, spec in mapping.items():
            a, b = split_pair(k)
            if canonical_atom(a) == canonical_atom(b):
                raise ValueError("Self-diffusion pairs (e.g., 'HH') are not valid here.")
            A = float(spec["A"]); gamma = float(spec["gamma"])
            self.diffusion_fits[pair_key(a, b)] = (A, gamma)

    def use_he_diffusion_set(self, name: str):
        """Switch every He pair in diffusion_fits to one of he_diffusion_sets ('literature' or 'chapman-enskog')."""
        if name not in self.he_diffusion_sets:
            raise KeyError(f"Unknown He diffusion set '{name}'. Valid: {list(self.he_diffusion_sets)}")
        self.diffusion_fits.update(self.he_diffusion_sets[name])
