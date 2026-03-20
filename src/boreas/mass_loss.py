import numpy as np
from scipy.optimize import brentq
from .flow_solutions import FlowSolutions as FS

class MomentumBalanceError(RuntimeError):
    def __init__(self, message, *, min_abs_f=None, R_best=None, R_over_Rp=None):
        super().__init__(message)
        self.min_abs_f = min_abs_f
        self.R_best = R_best
        self.R_over_Rp = R_over_Rp

class SoundSpeedRootError(RuntimeError):
    pass

class MassLoss:
    def __init__(self, params):
        self.params = params

    # --- helper: pick the μ_outflow to use for transport ---
    def _mu_outflow_transport(self):
        """
        Use the effective outflow μ if fractionation has set it; otherwise
        fall back to the reservoir (composition-based) atomic μ.
        Returns a dimensionless μ in units of m_H (like mmw_*_outflow).
        """
        mu_eff = getattr(self.params, 'mmw_outflow_eff', None)
        return mu_eff if (mu_eff is not None and mu_eff > 0) else self.params.get_mu_outflow_current()


    ### Mass loss rate and sound speed ###
        
    def compute_mdot_only(self, cs, RXUV, m_planet):
        """
        Compute the instantaneous mass-loss rate for a given sound speed.
        """
        G = self.params.G
        
        # dissociated outflow mu
        mmw_outflow = self._mu_outflow_transport()
        if mmw_outflow is None:
            raise ValueError(f"Unknown outflow_mode '{self.params.outflow_mode}' for mmw selection")

        # "hot" sonic point radius
        RS_flow = G * m_planet / (2. * cs**2)

        # integrate Parker wind from RXUV outward
        r_max = max(5 * RS_flow, 5 * RXUV)
        r = np.logspace(np.log10(RXUV), np.log10(r_max), 250)

        if RS_flow >= RXUV:
            u = FS.get_parker_wind(r, cs, RS_flow)
        else: 
            # does not mean CP mass loss. this RS_flow is the "hot" one. It can live in the bolometrically heated region
            # and still lead to a photoevaporative mass loss, as long as the "cold" sonic point is outside the RXUV.
            constant = (1. - 4. * np.log(RXUV / RS_flow) - 4. * (RS_flow / RXUV) - 1e-13)
            u = FS.get_parker_wind_const(r, cs, RS_flow, constant)

            if np.any(~np.isfinite(u)) or np.any(u <= 0):
                raise ValueError(f"Bad u: min(u)={np.nanmin(u)}, RXUV={RXUV}, cs={cs}, RS_flow={RS_flow}")

        # integrate Parker wind to get tau geometrically
        rho_shape = (RS_flow / r)**2 * (cs / u)                     # dimensionless (from continuity rho u r^2 = const). captures how density falls with r, independent of units
        # tau = np.abs(np.trapz(rho_shape[::-1], r[::-1]))            # geometric column, cm. We do not use "np.trapezoid" because it assumes x increasing, while we do rho and r[::-1] = decreasing.
        tau = np.abs(np.trapezoid(rho_shape[::-1], r[::-1]))
        #  tau = np.trapezoid(rho_shape, r)  # no reversal          # we could try this line with np.trapezoid but np.trapz remains officially supported and backward-compatible for now.
        if not np.isfinite(tau) or tau == 0:
            raise ValueError(f"Bad tau: {tau}, RXUV={RXUV}, cs={cs}, RS_flow={RS_flow}, r[0]={r[0]}, r[-1]={r[-1]}")


        # mass absorption coefficient at XUV for mixtures
        chi_xuv = self.params.xuv_cross_section_per_mass()          # mass absorption coefficient at XUV for mixtures
        
        # base density at RXUV from XUV absorption
        # rho_s = 1.0 / ((sigma / (mmw_outflow * m_H/2.)) * tau)    # for H2 (+-He) envelopes
        # rho_s = 1.0 / ((sigma / (mmw_outflow * m_H)) * tau)       # do not divide m_H/2, per oxygen atom
        rho_s = 1.0 / (chi_xuv * tau)                               # use mixture instead of H atom
        
        rho = rho_shape * rho_s

        Mdot = 4 * np.pi * RXUV**2 * rho[0] * u[0]
        return Mdot

    def compute_mdot_el_target(self, RXUV, m_planet):
        """
        Energy-limited target mass-loss rate.

        The user-facing/input FXUV is the incident stellar flux at the orbit.
        For the EL branch we keep the notebook/Owen-style geometry by using the
        planet-mean flux internally, FXUV_incident / 4. This is algebraically
        equivalent to writing the standard EL expression with an explicit 1/4.
        """
        G, eta = self.params.G, self.params.eff
        FXUV_mean = self.params.fxuv_global_mean()
        return eta * 4.0 * np.pi * RXUV**3 / (G * m_planet) * FXUV_mean

    def compute_sound_speed(self, RXUV, m_planet):
        """
        Solve for c_s such that computed Mdot matches the energy-limited rate.
        """
        Mdot_EL = self.compute_mdot_el_target(RXUV, m_planet)

        lower_bound_initial, upper_bound = 2e5, 1e8 # initially 2e5, 1e13

        def mdot_difference(cs):
            Mdot = self.compute_mdot_only(cs, RXUV, m_planet)
            return (Mdot - Mdot_EL) / (Mdot_EL + 1)

        f1 = mdot_difference(lower_bound_initial)
        if f1 < 0:
            # return brentq(mdot_difference, lower_bound_initial, upper_bound)            
            try:                
                return brentq(mdot_difference, lower_bound_initial, upper_bound)
            except ValueError as e:
                raise SoundSpeedRootError(f"Sound-speed root failed at RXUV={RXUV:.3e}: {e}") from e            
        else:
            while f1 > 0:
                lower_bound_initial *= 0.95
                f1 = mdot_difference(lower_bound_initial)
            try:                
                return brentq(mdot_difference, lower_bound_initial, upper_bound / 10)
            except ValueError as e:
                raise SoundSpeedRootError(f"Sound-speed root failed at RXUV={RXUV:.3e}: {e}") from e

    ### Regime-specific density comparisons ###
    def compare_densities_EL(self, RXUV, rho_bolo, r_planet, m_planet, cs_bolo, FXUV_photon):
        """
        Momentum balance for the energy-limited (EL) case.
        Try to find an EL RXUV root.
        If no sign change and min|f| > accept_min_abs_f, raise MomentumBalanceError.
        Otherwise return the best-R solution.

        Note: FXUV_photon here is kept as the incident photon flux, matching the
        original notebook logic used for the EL/RL timescale diagnostic. The EL
        mass-loss normalization itself is set separately by compute_mdot_el_target().
        """
        G, alpha = self.params.G, self.params.alpha_rec
    
        # the same hydrostatic profile rho_bolo (density at the bolometric photosphere), but evaluated at the XUV base
        expo = (G * m_planet / cs_bolo**2) * (1.0 / RXUV - 1.0 / r_planet)
        expo = np.clip(expo, -60.0, 60.0)
        rho_eq = rho_bolo * np.exp(expo)
        # outflow cs that matches EL power (bounded)
        cs_outflow = self.compute_sound_speed(RXUV, m_planet)
        cs_outflow = min(cs_outflow, 1.2e6) # Limit sound speed to T ~ 10^4 K
        
        # parker geometry
        Rs = G * m_planet / (2. * cs_outflow**2)
        
        if RXUV <= Rs:
            u_launch = FS.get_parker_wind_single(RXUV, cs_outflow, Rs)
            # local flow scale height Hflow ~ d ln(...) / dr^(-1)
            u_launch_for_grad = FS.get_parker_wind_single(1.001 * RXUV, cs_outflow, Rs)
            Hflow = 0.001 * RXUV / (np.log(1. / RXUV**2 / u_launch) - np.log(1. / (1.001 * RXUV)**2 / u_launch_for_grad))
        else:
            u_launch = cs_outflow
            Hflow = RXUV
            
        # recombination timescale vs flow timescale (diagnostic). As in the
        # source notebook, this uses the incident XUV photon flux directly.
        time_scale_flow = Hflow / u_launch
        time_scale_recom = np.sqrt(Hflow / (FXUV_photon * alpha))
        time_scale_ratio = time_scale_recom / time_scale_flow
        if cs_outflow < 1.2e6:
            time_scale_ratio = 10. # not 'RL' flag
            
        # flow density at RXUV from continuity with the EL-consistent Mdot
        Mdot = self.compute_mdot_only(cs_outflow, RXUV, m_planet)
        # rho_pe = Mdot / (4 * np.pi * r_planet**2 * u_launch) # <--------- this should be evaluated at RXUV, correct?
        rho_pe = Mdot / (4.0 * np.pi * RXUV**2 * u_launch)
        
        # EL balance function: (H - Q)/(2H)
        H_term = rho_eq * cs_bolo**2
        Q_term = rho_pe * (u_launch**2 + cs_outflow**2)
        # diff = (rho_eq * cs_bolo**2 - rho_pe * (u_launch**2 + cs_outflow**2)) / (2. * rho_eq * cs_bolo**2)
        diff = (H_term - Q_term) / (2.0 * H_term)
        
        return diff, time_scale_ratio, rho_eq, rho_pe

    def find_RXUV_solution_EL(self, r_planet, m_planet, rho_bolo, cs_bolo, FXUV_photon, accept_min_abs_f=1e-2, nscan=24):
        """
        RXUV solution for the energy-limited (EL) case.
        """
        Rp = float(r_planet)
        R_min = Rp * 1.001
        R_max = Rp * 10
        R_grid = np.geomspace(R_min, R_max, int(nscan))
    
        def eval_at(R):
            return self.compare_densities_EL(R, rho_bolo, r_planet, m_planet, cs_bolo, FXUV_photon)

        vals = []
        first_exc = None
        
        for R in R_grid:
            try:
                d, tsr, rq, rp = eval_at(R)
            except Exception as exc:
                if first_exc is None:
                    first_exc = exc
                d, tsr, rq, rp = np.nan, np.nan, np.nan, np.nan
            vals.append((R, d, tsr, rq, rp))
            
        # keep finite f(R)
        finite_vals = [(R, d, tsr, rq, rp) for (R, d, tsr, rq, rp) in vals if np.isfinite(d)]
        if not finite_vals:
            msg = "EL scan produced no finite f(R) values"
            if first_exc is not None:
                msg += f"; first exception: {type(first_exc).__name__}: {first_exc}"
            raise MomentumBalanceError(msg)

        vals = finite_vals

        # try to find first sign change
        bracket = None
        for (R1, d1, *_), (R2, d2, *_) in zip(vals[:-1], vals[1:]):
            if d1 == 0.0:
                bracket = (R1, R1); break
            if np.sign(d1) != np.sign(d2):
                bracket = (R1, R2); break

        if bracket and bracket[0] != bracket[1]:
            def f_only(R):
                d, *_ = eval_at(R)
                return d
            RX = brentq(f_only, *bracket)
            d, tsr, rq, rp = eval_at(RX)
            return RX, tsr, rq, rp

        # no sign change → choose argmin |f|
        R_best, d_best, tsr_best, rq_best, rp_best = min(vals, key=lambda t: abs(t[1]))
        min_abs_f = abs(d_best)
        if min_abs_f > accept_min_abs_f:
            raise MomentumBalanceError(
                "EL momentum balance cannot be satisfied within scan range",
                min_abs_f=min_abs_f, R_best=R_best, R_over_Rp=R_best / Rp
            )
        return R_best, tsr_best, rq_best, rp_best

        # RXUV_lower_bound = r_planet * 1.001
        # RXUV_upper_bound = r_planet * 10

        # def root_function(RXUV):
        #     diff, _, _, _ = self.compare_densities_EL(RXUV, rho_bolo, r_planet, m_planet, cs_bolo, FXUV_photon)
        #     return diff

        # f_lower = root_function(RXUV_lower_bound)
        # f_upper = root_function(RXUV_upper_bound)
        # if f_lower * f_upper > 0:
        #     max_iterations = 100
        #     for _ in range(max_iterations):
        #         RXUV_upper_bound *= 1.5
        #         f_upper = root_function(RXUV_upper_bound)
        #         if f_lower * f_upper < 0:
        #             break
        #     else:
        #         raise ValueError("Cannot find valid bounds for RXUV root finding.")
            
        # RXUV_solution = brentq(root_function, RXUV_lower_bound, RXUV_upper_bound)
        # _, time_scale_ratio, rho_eq, rho_pe = self.compare_densities_EL(RXUV_solution, rho_bolo, r_planet, m_planet, cs_bolo, FXUV_photon)
        
        # return RXUV_solution, time_scale_ratio, rho_eq, rho_pe

    def compare_densities_RL(self, RXUV, rho_bolo, r_planet, m_planet, cs_bolo, FXUV_photon):
        """
        Momentum balance for the recombination-limited (RL) case.
        RL closure coded is H-specific (uses H recombination to set the base electron density
        and fixes cs=1.2e6 cm s-1.

        Following the original Owen notebook, the RL closure uses the incident XUV
        photon flux directly rather than a global-mean photon flux.
        """
        G, m_H, alpha_rec = self.params.G, self.params.m_H, self.params.alpha_rec

        # the same hydrostatic profile rho_bolo (density at the bolometric photosphere), but evaluated at the XUV base
        expo = (G * m_planet / cs_bolo**2) * (1.0 / RXUV - 1.0 / r_planet)
        expo = np.clip(expo, -60.0, 60.0)
        rho_eq = rho_bolo * np.exp(expo)

        cs_RL = 1.2e6 # Sound speed for T ~ 10^4 K
        Hflow = min(RXUV / 3, cs_RL**2 * RXUV**2 / (2 * G * m_planet))
        nb = np.sqrt(FXUV_photon / (alpha_rec * Hflow))

        Rs_RL = G * m_planet / (2. * cs_RL**2)
        if RXUV <= Rs_RL:
            u_RL = FS.get_parker_wind_single(RXUV, cs_RL, Rs_RL)
        else:
            u_RL = cs_RL

        Mdot_RL = 4 * np.pi * RXUV**2 * m_H * nb * u_RL
        
        # rho_pe = Mdot_RL / (4 * np.pi * r_planet**2 * u_RL) # <------------ this should be evaluated at RXUV, correct?
        rho_pe = Mdot_RL / (4 * np.pi * RXUV**2 * u_RL)

        diff = (rho_eq * cs_bolo**2 - rho_pe * (u_RL**2 + cs_RL**2)) / (2. * rho_eq * cs_bolo**2)
        
        return diff, rho_eq, rho_pe

    def find_RXUV_solution_RL(self, r_planet, m_planet, rho_bolo, cs_bolo, FXUV_photon):
        """
        RXUV solution for the recombination-limited (RL) case.
        """
        R_min = r_planet * 1.001
        R_max = r_planet * 10

        def root_function(RXUV):
            diff, _, _ = self.compare_densities_RL(RXUV, rho_bolo, r_planet, m_planet, cs_bolo, FXUV_photon)
            return diff

        f_lower = root_function(R_min)
        f_upper = root_function(R_max)
        if f_lower * f_upper > 0:
            max_iterations = 100
            for _ in range(max_iterations):
                R_max *= 1.5
                f_upper = root_function(R_max)
                if f_lower * f_upper < 0:
                    break
            else:
                raise ValueError("Cannot find valid bounds for RXUV root finding.")
        RXUV_solution = brentq(root_function, R_min, R_max)
        _, rho_eq, rho_pe = self.compare_densities_RL(RXUV_solution, rho_bolo, r_planet, m_planet, cs_bolo, FXUV_photon)
        
        return RXUV_solution, rho_eq, rho_pe

    def compute_mdot_rl(self, RXUV, m_planet, FXUV_photon):
        """
        Compute the RL mass-loss rate implied by the H recombination closure
        used in compare_densities_RL().

        FXUV_photon is the incident photon flux, matching the notebook RL branch.
        """
        G, m_H, alpha_rec = self.params.G, self.params.m_H, self.params.alpha_rec

        cs_RL = 1.2e6
        Hflow = min(RXUV / 3, cs_RL**2 * RXUV**2 / (2 * G * m_planet))
        nb = np.sqrt(FXUV_photon / (alpha_rec * Hflow))

        Rs_RL = G * m_planet / (2. * cs_RL**2)
        if RXUV <= Rs_RL:
            u_RL = FS.get_parker_wind_single(RXUV, cs_RL, Rs_RL)
        else:
            u_RL = cs_RL

        return 4.0 * np.pi * RXUV**2 * m_H * nb * u_RL

    def compute_mass_loss_parameters(self, m_planet, r_planet, teq, rl_policy='auto', light_major=None):
        """
        rl_policy:
        'auto'   -> current behavior (enter RL if time_scale_ratio<1)
        'never'  -> never switch to RL (always EL)
        'if_H'   -> switch to RL only if (light_major=='H' and time_scale_ratio<1)
        light_major: pass the chosen light major species symbol, e.g., 'H'

        Solve for RXUV, c_s, and Mdot in EL or RL, using mode flag for base molecular weight.
        """

        results = []
        for m_p, r_p, T_eq in zip(m_planet, r_planet, teq):
            try:
                # Keep the input convention explicit:
                # - params.FXUV is the incident stellar XUV energy flux at orbit
                # - EL converts that to a planet-mean flux internally
                # - RL/timescale terms use the incident photon flux directly
                FXUV_photon = self.params.fxuv_photon_incident()
                G, k_b, m_H = self.params.G, self.params.k_b, self.params.m_H

                # photo layer
                mu_bolo    = self.params.get_mmw_bolometric()
                cs_bolo    = np.sqrt(k_b * T_eq / (m_H * mu_bolo))          # always calculated with bolometric mu 
                kappa_bolo = self.params.kappa_p_all
                # kappa_bolo = np.clip(self.params.kappa_p_all, 1e-3, 10.0) # tune bounds
                rho_bolo   = G * m_p / r_p**2 / (kappa_bolo * cs_bolo**2)   # the anchor density at r=Rp used to get rho_eq by an isothermal scale height.

                # Energy-limited (EL) regime calculations
                RXUV_solution_EL, time_scale_ratio, rho_eq_EL, rho_pe_EL = self.find_RXUV_solution_EL(r_p, m_p, rho_bolo, cs_bolo, FXUV_photon)
                cs_outflow = self.compute_sound_speed(RXUV_solution_EL, m_p)
                cs_outflow = min(cs_outflow, 1.2e6)                         # Limit sound speed to T ~ 10^4 K
                Mdot_EL = self.compute_mdot_only(cs_outflow, RXUV_solution_EL, m_p)
                RS_flow = G * m_p / (2 * cs_outflow**2)

                sol = {'m_planet':m_p,'r_planet':r_p,'Teq':T_eq,
                    'RXUV':RXUV_solution_EL,'cs':cs_outflow,'Mdot':Mdot_EL,
                    'RS_flow':RS_flow,'rho_eq':rho_eq_EL,'rho_pe':rho_pe_EL,
                    'time_scale_ratio':time_scale_ratio,'regime':'EL'}

                # Recombination-limited (RL) regime check
                allow_RL = False
                if rl_policy == 'auto':
                    allow_RL = (time_scale_ratio < 1)
                elif rl_policy == 'never':
                    allow_RL = False
                elif rl_policy == 'if_H':
                    allow_RL = (light_major == 'H' and time_scale_ratio < 1)

                if allow_RL:
                # if time_scale_ratio < 1:
                    RXUV_solution_RL, rho_eq_RL, rho_pe_RL = self.find_RXUV_solution_RL(r_p, m_p, rho_bolo, cs_bolo, FXUV_photon)
                    cs_outflow_RL = 1.2e6
                    Rs_RL = G * m_p / (2. * cs_outflow_RL**2)
                    Mdot_RL = self.compute_mdot_rl(RXUV_solution_RL, m_p, FXUV_photon)

                    sol.update({'RXUV':RXUV_solution_RL, 'cs':1.2e6, 'Mdot':Mdot_RL,
                                'RS_flow': Rs_RL, 'rho_eq':rho_eq_RL, 'rho_pe':rho_pe_RL,
                                'regime':'RL'})

                results.append(sol)
                
            except SoundSpeedRootError as e:
                results.append({
                    'm_planet': m_p, 'r_planet': r_p, 'Teq': T_eq, 'FXUV': self.params.FXUV,
                    'RXUV': None, 'cs': None, 'Mdot': None,
                    'regime': 'SKIPPED',
                    'skip_reason': 'SoundSpeedRootError',
                    'error': str(e),
                })
                continue
            
            except MomentumBalanceError as e:
                results.append({'m_planet':m_p, 'r_planet':r_p, 'Teq':T_eq, 'FXUV':self.params.FXUV,
                    'RXUV': None, 'cs': None, 'Mdot': None,
                    'regime': 'SKIPPED',
                    'skip_reason': 'EL_momentum_no_root',
                    'error': str(e),
                    'EL_min_abs_f': float(e.min_abs_f) if e.min_abs_f is not None else None,
                    'EL_Rbest': float(e.R_best) if e.R_best is not None else None,
                    'EL_Rbest_over_Rp': float(e.R_over_Rp) if e.R_over_Rp is not None else None,
                })
                continue
                        
            except ValueError as e:
                results.append({'m_planet': m_p, 'r_planet': r_p, 'Teq': T_eq, 'FXUV': self.params.FXUV,
                    'RXUV': None, 'cs': None, 'Mdot': None,                                
                    'regime': 'SKIPPED',
                    'skip_reason': 'ValueError',
                    'error': str(e),
                })
                continue

        return results
