"""
Optional stellar XUV spectrum input.

Without a spectrum BOREAS describes the stellar input by two numbers: the incident
energy flux FXUV and one "typical" photon energy (E_photon = 20 eV), with every atomic
cross-section taken at that energy. A spectrum replaces the energy and photon fluxes,
and optionally (ModelParams.chi_from_spectrum = True) the absorption coefficient:

    energy flux      F_XUV   = ∫ F_E dE                  -> EL energy budget (Mdot target)
    photon flux      Phi_ion = ∫ F_E / E dE              -> RL base density, recombination timescale
    mass absorption  chi_XUV from sigma_s(E) over F_E    -> XUV base (tau = 1), EL branch only (optional)

all integrated from the H ionisation edge (13.6 eV) up to the end of the spectrum, or
up to E_max_eV if given. IR opacities, Teq and the recombination coefficient do not
depend on the XUV spectrum and are untouched.

Cross-sections are the Verner+1996 fits for neutral atoms (outer shells). Inner-shell
(K/L) edges are not included, so sigma is underestimated above ~0.2-0.5 keV for C, N,
O and S. That matters little here: photons that hard carry a small share of the
absorbed energy and are absorbed far below the XUV base anyway.
"""
import math
import numpy as np
from scipy.optimize import brentq
from .parameters import ATOMS, canonical_atom

EV_ERG          = 1.602176634e-12       # erg per eV
HC_EV_ANGSTROM  = 12398.419843320026    # h*c in eV * Angstrom
E_H_EDGE_EV     = 13.6                  # H ionisation edge, the lower end of "XUV"

# Verner+1996 (ApJ 465, 487) Table 1, ground state of the neutral atom:
# threshold Eth [eV] and fit parameters E0 [eV], sigma0 [Mb], ya, P, yw, y0, y1.
VERNER96_NEUTRAL = {
    "H":  dict(Eth=1.360e+1, E0=4.298e-1, sigma0_Mb=5.475e+4, ya=3.288e+1, P=2.963e+0, yw=0.0,      y0=0.0,      y1=0.0),
    "He": dict(Eth=2.459e+1, E0=1.361e+1, sigma0_Mb=9.492e+2, ya=1.469e+0, P=3.188e+0, yw=2.039e+0, y0=4.434e-1, y1=2.136e+0),
    "C":  dict(Eth=1.126e+1, E0=2.144e+0, sigma0_Mb=5.027e+2, ya=6.216e+1, P=5.101e+0, yw=9.157e-2, y0=1.133e+0, y1=1.607e+0),
    "N":  dict(Eth=1.453e+1, E0=4.034e+0, sigma0_Mb=8.235e+2, ya=8.033e+1, P=3.928e+0, yw=9.097e-2, y0=8.598e-1, y1=2.325e+0),
    "O":  dict(Eth=1.368e+1, E0=1.240e+0, sigma0_Mb=1.745e+3, ya=3.784e+0, P=1.764e+1, yw=7.583e-2, y0=8.698e+0, y1=1.271e-1),
    "S":  dict(Eth=1.036e+1, E0=1.808e+1, sigma0_Mb=4.564e+4, ya=1.000e+0, P=1.361e+1, yw=6.358e-1, y0=9.935e-1, y1=2.486e-1),
}

def verner96_sigma(atom, E_eV):
    """Photoionisation cross-section of a neutral atom [cm^2] at photon energy E_eV (scalar or array)."""
    p = VERNER96_NEUTRAL[canonical_atom(atom)]
    E = np.asarray(E_eV, dtype=float)
    x = E / p["E0"] - p["y0"]
    y = np.sqrt(x * x + p["y1"] ** 2)
    F = ((x - 1.0) ** 2 + p["yw"] ** 2) * y ** (0.5 * p["P"] - 5.5) * (1.0 + np.sqrt(y / p["ya"])) ** (-p["P"])
    sigma = np.where(E >= p["Eth"], p["sigma0_Mb"] * F * 1e-18, 0.0)  # Mb -> cm^2
    return sigma if sigma.ndim else float(sigma)

# x_unit -> (is_wavelength, factor to Angstrom or eV)
_X_UNITS = {
    "angstrom": (True, 1.0),
    "nm":       (True, 10.0),
    "ev":       (False, 1.0),
    "kev":      (False, 1.0e3),
}

class XUVSpectrum:
    """
    Incident stellar XUV spectrum at the planet's orbit.

    x, flux:  the tabulated spectrum. flux is an energy flux density in
              erg cm^-2 s^-1 per unit of x, i.e. per Angstrom, nm, eV or keV.
              Negative values (noise in observed spectra) count as zero.
    x_unit:   'angstrom', 'nm', 'eV' or 'keV' (case-insensitive).
    scale:    multiplies flux, e.g. (d_ref / a)**2 to bring a spectrum given at
              d_ref (1 au, Earth, ...) to the planet's orbit a. Has no effect on the
              result if the spectrum is later normalized to a target flux (see
              ModelParams.set_xuv_spectrum(..., normalize_to=...)), since that
              rescales to the target regardless of scale.
    E_min_eV, E_max_eV: the band that counts as XUV. Below 13.6 eV nothing ionises H,
              so the default band starts there; E_max_eV=None keeps the whole tail.

    Internally the spectrum is a set of quadrature nodes: photon energies E_k [eV]
    and the energy flux dF_k [erg cm^-2 s^-1] each node carries (trapezoid rule
    over the band, edges interpolated). Every integral is then a plain sum.
    """

    def __init__(self, x, flux, x_unit="angstrom", scale=1.0, E_min_eV=E_H_EDGE_EV, E_max_eV=None):
        unit = str(x_unit).lower()
        if unit not in _X_UNITS:
            raise ValueError(f"Unknown spectrum x_unit '{x_unit}'. Valid: angstrom, nm, eV, keV")
        x = np.asarray(x, dtype=float).ravel()
        flux = np.asarray(flux, dtype=float).ravel() * float(scale)
        if x.shape != flux.shape or x.size < 2:
            raise ValueError("Spectrum needs matching x and flux arrays with at least 2 points.")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(flux))):
            raise ValueError("Spectrum contains non-finite values.")
        if np.any(x <= 0.0):
            raise ValueError("Spectrum x values must be > 0.")
        flux = np.clip(flux, 0.0, None)

        # convert to photon energy [eV] and flux per eV
        is_wavelength, factor = _X_UNITS[unit]
        if is_wavelength:
            lam_A = x * factor
            E = HC_EV_ANGSTROM / lam_A
            F_E = (flux / factor) * lam_A**2 / HC_EV_ANGSTROM # F_lambda |dlambda/dE|
        else:
            E = x * factor
            F_E = flux / factor

        order = np.argsort(E)
        E, F_E = E[order], F_E[order]
        if np.any(np.diff(E) <= 0.0):
            raise ValueError("Spectrum x values must be unique.")

        # clip to the band, interpolating the flux density at the band edges
        lo = max(float(E_min_eV), E[0])
        hi = E[-1] if E_max_eV is None else min(float(E_max_eV), E[-1])
        if not hi > lo:
            raise ValueError(f"Spectrum has no coverage in the XUV band {E_min_eV}-{E_max_eV} eV "
                             f"(it spans {E[0]:.4g}-{E[-1]:.4g} eV).")
        inside = (E > lo) & (E < hi)
        E_band = np.concatenate(([lo], E[inside], [hi]))
        F_band = np.interp(E_band, E, F_E)

        w = np.empty_like(E_band) # trapezoid weights
        dE = np.diff(E_band)
        w[0], w[-1] = 0.5 * dE[0], 0.5 * dE[-1]
        w[1:-1] = 0.5 * (dE[:-1] + dE[1:])

        self._setup(E_band, F_band * w)

    @classmethod
    def from_file(cls, path, x_unit="angstrom", scale=1.0, E_min_eV=E_H_EDGE_EV, E_max_eV=None,
                  usecols=(0, 1), delimiter=None):
        """
        Two-column text file (x, flux), '#' comments. A .csv is read comma-separated
        unless a delimiter is given. Units as for the constructor.
        """
        if delimiter is None and str(path).lower().endswith(".csv"):
            delimiter = ","
        data = np.loadtxt(path, comments="#", delimiter=delimiter, usecols=usecols, ndmin=2)
        return cls(data[:, 0], data[:, 1], x_unit=x_unit, scale=scale, E_min_eV=E_min_eV, E_max_eV=E_max_eV)

    @classmethod
    def monochromatic(cls, E_eV, energy_flux):
        """
        A single line of energy flux energy_flux [erg cm^-2 s^-1] at E_eV. This is the
        picture BOREAS uses without a spectrum, so it is mainly a reference case.
        """
        spec = cls.__new__(cls)
        spec._setup(np.array([float(E_eV)]), np.array([float(energy_flux)]))
        return spec

    def _setup(self, E, dF):
        if not np.sum(dF) > 0.0:
            raise ValueError("Spectrum carries no flux in the XUV band.")
        self.E_eV = E
        self.dF = dF
        # sigma_s(E_k) for every atom, one row per atom in ATOMS order
        self._sigma = np.array([verner96_sigma(a, E) for a in ATOMS])
        # chi only depends on the spectral shape, so scaled copies share this cache
        self._chi_cache = {}

    def scaled(self, factor):
        """Same spectral shape, flux multiplied by factor."""
        if not factor > 0.0:
            raise ValueError(f"Spectrum scale factor must be > 0 (got {factor}).")
        spec = XUVSpectrum.__new__(XUVSpectrum)
        spec.E_eV = self.E_eV
        spec.dF = self.dF * float(factor)
        spec._sigma = self._sigma
        spec._chi_cache = self._chi_cache
        return spec

    # --- the three quantities the physics uses ---
    def energy_flux(self):
        """Incident XUV energy flux [erg cm^-2 s^-1]."""
        return float(np.sum(self.dF))

    def photon_flux(self):
        """Incident H-ionising photon flux [photons cm^-2 s^-1]."""
        return float(np.sum(self.dF / (self.E_eV * EV_ERG)))

    def mean_photon_energy_eV(self):
        """Mean energy per ionising photon, F_XUV / Phi_ion [eV]. The spectrum's counterpart of E_photon."""
        return self.energy_flux() / self.photon_flux() / EV_ERG

    def mass_absorption(self, atoms_per_gram):
        """
        Effective XUV mass absorption coefficient chi [cm^2 g^-1] of a gas with the given
        atoms per gram {atom: n}, for the energy-weighted spectrum.

        kappa(E) = sum_s sigma_s(E) n_s varies across the spectrum, so there is no single
        tau = 1 surface. chi is set by the column Sigma at which the gas has transmitted
        a fraction 1/e of the energy it can absorb:

            sum_k dF_k exp(-kappa_k Sigma) = e^-1 sum_k dF_k,   chi = 1 / Sigma,

        the sums running over the nodes the gas absorbs at all (kappa_k > 0). For a
        single line this is chi = kappa(E) exactly, the no-spectrum definition. Energy
        rather than photon weighting, because chi only enters the EL branch, where the
        question is where the heating happens. Returns 0.0 if nothing absorbs.
        """
        n = np.array([float(atoms_per_gram.get(a, 0.0)) for a in ATOMS])
        key = tuple(n)
        chi = self._chi_cache.get(key)
        if chi is not None:
            return chi

        kappa = n @ self._sigma
        absorbs = kappa > 0.0
        if not np.any(absorbs):
            chi = 0.0
        else:
            kap, w = kappa[absorbs], self.dF[absorbs]
            w = w / np.sum(w)
            target = math.exp(-1.0)

            # solve in ln(Sigma): transmission is ~1 at 1e-3/kappa_max, ~0 at 1e3/kappa_min
            def excess(lnS):
                return float(np.sum(w * np.exp(-kap * math.exp(lnS)))) - target

            lnS = brentq(excess, math.log(1e-3 / kap.max()), math.log(1e3 / kap.min()), xtol=1e-14, rtol=1e-14)
            chi = math.exp(-lnS)

        if len(self._chi_cache) > 1024:
            self._chi_cache.clear()
        self._chi_cache[key] = chi
        return chi