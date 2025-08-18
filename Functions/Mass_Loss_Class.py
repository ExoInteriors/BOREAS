import numpy as np
from scipy.optimize import brentq
from Functions.Flow_Solutions_Class import FlowSolutions as FS

class MassLoss:
    def __init__(self, params):
        """
        Differentiates between pure H/He, pure H2O, and mixed H/He-H2O outflows via a mode flag.
        """
        self.params = params

    ### Mass loss rate and sound speed ###
        
    def compute_mdot_only(self, cs, REUV, m_planet):
        """
        Compute the instantaneous mass-loss rate for a given sound speed.
        """
        G, sigma, m_H = self.params.G, self.params.sigma_EUV, self.params.m_H

        # Select outflow molecular weight based on mode
        mode = self.params.outflow_mode
        if mode == 'HHe':
            mmw = self.params.mmw_HHe_outflow
        elif mode == 'H2O':
            mmw = self.params.mmw_H2O_outflow
        elif mode == 'HHe_H2O':
            mmw = self.params.mmw_HHe_H2O_outflow
        else:
            raise ValueError(f"Unknown outflow_mode '{mode}'")

        RS_flow = G * m_planet / (2. * cs**2) # "hot" sonic point radius

        # integrate Parker wind from REUV outward
        r_max = max(5 * RS_flow, 5 * REUV)
        r = np.logspace(np.log10(REUV), np.log10(r_max), 250)

        if RS_flow >= REUV:
            u = FS.get_parker_wind(r, cs, RS_flow)
        else: 
            # does not mean CP mass loss. this RS_flow is the "hot" one. It can live in the bolometrically heated region
            # and still lead to a photoevaporative mass loss, as long as the "cold" sonic point is outside the REUV.
            constant = (1. - 4. * np.log(REUV / RS_flow) - 4. * (RS_flow / REUV) - 1e-13)
            u = FS.get_parker_wind_const(r, cs, RS_flow, constant)

        rho = (RS_flow / r)**2 * (cs / u)
        tau = np.fabs(np.trapz(rho[::-1], r[::-1]))
        # base density at REUV from EUV absorption
        rho_s = 1.0 / ((sigma / (mmw * m_H/2.)) * tau)
        rho *= rho_s

        Mdot = 4 * np.pi * REUV**2 * rho[0] * u[0]
        return Mdot

    def compute_sound_speed(self, REUV, m_planet):
        """
        Solve for c_s such that computed Mdot matches the energy-limited rate.
        """
        G, FEUV, eta = self.params.G, self.params.get_param('FEUV'), self.params.eff
        Mdot_EL = eta * np.pi * REUV**3 / (4 * G * m_planet) * FEUV

        lower_bound_initial, upper_bound = 2e5, 1e13

        def mdot_difference(cs):
            Mdot = self.compute_mdot_only(cs, REUV, m_planet)
            return (Mdot - Mdot_EL) / (Mdot_EL + 1)

        f1 = mdot_difference(lower_bound_initial)
        if f1 < 0:
            return brentq(mdot_difference, lower_bound_initial, upper_bound)
        else:
            while f1 > 0:
                lower_bound_initial *= 0.95
                f1 = mdot_difference(lower_bound_initial)
            return brentq(mdot_difference, lower_bound_initial, upper_bound / 10)

    ### Regime-specific density comparisons ###
    def compare_densities_EL(self, REUV, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon):
        """
        Momentum balance for the energy-limited (EL) case.
        """
        G, alpha = self.params.G, self.params.alpha_rec

        rho_EUV = rho_photo * np.exp((G * m_planet / cs_eq**2) * (1. / REUV - 1. / r_planet))
        cs_use = self.compute_sound_speed(REUV, m_planet)
        cs_use = min(cs_use, 1.2e6) # Limit sound speed to T ~ 10^4 K
        Mdot = self.compute_mdot_only(cs_use, REUV, m_planet)
        Rs = G * m_planet / (2. * cs_use**2)

        if REUV <= Rs:
            u_launch = FS.get_parker_wind_single(REUV, cs_use, Rs)
            u_launch_for_grad = FS.get_parker_wind_single(1.001 * REUV, cs_use, Rs)
            Hflow = 0.001 * REUV / (np.log(1. / REUV**2 / u_launch) - np.log(1. / (1.001 * REUV)**2 / u_launch_for_grad))
        else:
            u_launch = cs_use
            Hflow = REUV

        time_scale_flow = Hflow / u_launch
        time_scale_recom = np.sqrt(Hflow / (FEUV_photon * alpha))
        time_scale_ratio = time_scale_recom / time_scale_flow
        if cs_use < 1.2e6:
            time_scale_ratio = 10.

        rho_flow = Mdot / (4 * np.pi * r_planet**2 * u_launch)
        diff = (rho_EUV * cs_eq**2 - rho_flow * (u_launch**2 + cs_use**2)) / (2. * rho_EUV * cs_eq**2)

        return diff, time_scale_ratio, rho_EUV, rho_flow

    def find_REUV_solution_EL(self, r_planet, m_planet, rho_photo, cs_eq, FEUV_photon):
        """
        REUV solution for the energy-limited (EL) case.
        """
        REUV_lower_bound = r_planet * 1.001
        REUV_upper_bound = r_planet * 5

        def root_function(REUV):
            diff, _, _, _ = self.compare_densities_EL(REUV, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon)
            return diff

        f_lower = root_function(REUV_lower_bound)
        f_upper = root_function(REUV_upper_bound)
        if f_lower * f_upper > 0:
            max_iterations = 100
            for _ in range(max_iterations):
                REUV_upper_bound *= 1.5
                f_upper = root_function(REUV_upper_bound)
                if f_lower * f_upper < 0:
                    break
            else:
                raise ValueError("Cannot find valid bounds for REUV root finding.")
            
        REUV_solution = brentq(root_function, REUV_lower_bound, REUV_upper_bound)
        _, time_scale_ratio, rho_EUV, rho_flow = self.compare_densities_EL(REUV_solution, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon)
        
        return REUV_solution, time_scale_ratio, rho_EUV, rho_flow

    def compare_densities_RL(self, REUV, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon):
        """
        Momentum balance for the recombination-limited (RL) case.
        """
        G, m_H, alpha_rec = self.params.G, self.params.m_H, self.params.alpha_rec

        rho_EUV = rho_photo * np.exp((G * m_planet / cs_eq**2) * (1. / REUV - 1. / r_planet))

        cs_RL = 1.2e6 # Sound speed for T ~ 10^4 K
        Hflow = min(REUV / 3, cs_RL**2 * REUV**2 / (2 * G * m_planet))
        nb = np.sqrt(FEUV_photon / (alpha_rec * Hflow))

        Rs_RL = G * m_planet / (2. * cs_RL**2)
        if REUV <= Rs_RL:
            u_RL = FS.get_parker_wind_single(REUV, cs_RL, Rs_RL)
        else:
            u_RL = cs_RL

        Mdot_RL = 4 * np.pi * REUV**2 * m_H * nb * u_RL
        rho_flow = Mdot_RL / (4 * np.pi * r_planet**2 * u_RL)

        diff = (rho_EUV * cs_eq**2 - rho_flow * (u_RL**2 + cs_RL**2)) / (2. * rho_EUV * cs_eq**2)
        
        return diff, rho_EUV, rho_flow

    def find_REUV_solution_RL(self, r_planet, m_planet, rho_photo, cs_eq, FEUV_photon):
        """
        REUV solution for the recombination-limited (RL) case.
        """
        REUV_lower_bound = r_planet * 1.001
        REUV_upper_bound = r_planet * 5

        def root_function(REUV):
            diff, _, _ = self.compare_densities_RL(REUV, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon)
            return diff

        f_lower = root_function(REUV_lower_bound)
        f_upper = root_function(REUV_upper_bound)
        if f_lower * f_upper > 0:
            max_iterations = 100
            for _ in range(max_iterations):
                REUV_upper_bound *= 1.5
                f_upper = root_function(REUV_upper_bound)
                if f_lower * f_upper < 0:
                    break
            else:
                raise ValueError("Cannot find valid bounds for REUV root finding.")
        REUV_solution = brentq(root_function, REUV_lower_bound, REUV_upper_bound)
        _, rho_EUV, rho_flow = self.compare_densities_RL(REUV_solution, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon)
        
        return REUV_solution, rho_EUV, rho_flow

    ### Combine everything and run mass loss model ###

    def compute_mass_loss_parameters(self, m_p, r_planet, teq):
        """
        Solve for REUV, c_s, and Mdot in EL or RL, using mode flag for base molecular weight.
        """
        
        results = []
        mode = self.params.outflow_mode
        for m_p, r_p, T in zip(m_p, r_planet, teq):
            FEUV = self.params.get_param('FEUV')
            E_photon = self.params.E_photon
            FEUV_photon = FEUV/E_photon
            G, k_b, m_H = self.params.G, self.params.k_b, self.params.m_H
            
            # select base μ and opacity
            if mode=='HHe':
                μ_base, κ_base = self.params.mmw_HHe, self.params.kappa_p_HHe
            elif mode=='H2O':
                μ_base, κ_base = self.params.mmw_H2O, self.params.kappa_p_H2O
            elif mode == 'HHe_H2O':
                μ_base, κ_base = self.params.mmw_HHe_H2O, self.params.kappa_p_HHe_H2O
            else:
                raise ValueError(f"Unknown outflow_mode '{mode}'")
            
            print('mu', μ_base, '\n kappa', κ_base)

            # photo layer
            cs_eq   = np.sqrt(k_b*T/(m_H*μ_base))
            rho_photo  = G * m_p / r_p**2 / (κ_base * cs_eq**2)


            # Energy-limited (EL) regime calculations
            REUV_solution_EL, time_scale_ratio, rho_EUV_EL, rho_flow_EL = self.find_REUV_solution_EL(r_p, m_p, rho_photo, cs_eq, FEUV_photon)
            cs_use = self.compute_sound_speed(REUV_solution_EL, m_p)
            cs_use = min(cs_use, 1.2e6) # Limit sound speed to T ~ 10^4 K
            Mdot_EL = self.compute_mdot_only(cs_use, REUV_solution_EL, m_p)
            RS_flow = G * m_p / (2 * cs_use**2)

            sol = {'m_planet':m_p,'r_planet':r_p,'Teq':T,
                'REUV':REUV_solution_EL,'cs':cs_use,'Mdot':Mdot_EL,
                'RS_flow':RS_flow,'rho_EUV':rho_EUV_EL,'rho_flow':rho_flow_EL,
                'time_scale_ratio':time_scale_ratio,'regime':'EL'}

            # Recombination-limited (RL) regime check
            if time_scale_ratio < 1:
                REUV_solution_RL, rho_EUV_RL, rho_flow_RL = self.find_REUV_solution_RL(r_p, m_p, rho_photo, cs_eq, FEUV_photon)
                cs_use_RL = 1.2e6
                Rs_RL = G * m_p / (2. * cs_use_RL**2)
                # Mdot_RL = 4 * np.pi * REUV_solution_RL**2 * m_H * nb * u_RL
                Mdot_RL = self.compute_mdot_only(cs_use_RL, REUV_solution_RL, m_p)

                sol.update({'REUV':REUV_solution_RL,'cs':1.2e6,'Mdot':Mdot_RL,'RS_flow': Rs_RL,'rho_EUV':rho_EUV_RL,'rho_flow':rho_flow_RL,'regime':'RL'})

            # For pure H/He mode, compute outflow temperature and pressure
            if mode == 'HHe':
                mmw_out = self.params.mmw_HHe
                T_out = sol['cs']**2 * m_H * mmw_out / k_b
                P_EUV = sol['rho_EUV'] * k_b * T_out / (m_H * mmw_out)
                sol.update({'T_outflow': T_out, 'P_EUV': P_EUV})

            results.append(sol)
        return results