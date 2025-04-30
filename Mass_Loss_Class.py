import numpy as np
from scipy.optimize import brentq
from Flow_Solutions_Class import FlowSolutions as FS

class MassLoss:
    def __init__(self, params):
        """
        Initialize the MassLoss class with model parameters.
        :param params: An instance of the ModelParams class or a similar object containing constants.

        In this model, we differentiate between a pure H/He, pure H2O, and a mix of H/He-H2O mass loss in 2 different places. 
        They are pointed out by arrows in comment form. Make sure to change each time.
        """
        self.params = params

    ### Mass loss rate and sound speed ###
        
    def compute_mdot_only(self, cs, REUV, m_planet):
        """
        Direct computation of the mass-loss rate (Mdot).
        """
        G, sigma_EUV, m_H = self.params.G, self.params.sigma_EUV, self.params.m_H
        mmw_HHe = self.params.mmw_HHe
        mmw_H2O_outflow = self.params.get_param('mmw_H2O_outflow') # always use the latest value
        mmw_HHe_H2O_outflow = self.params.get_param('mmw_HHe_H2O_outflow') # always use the latest value
        RS_flow = G * m_planet / (2. * cs**2) # "hot" sonic point radius

        if RS_flow >= REUV:
            r = np.logspace(np.log10(REUV), np.log10(max(5 * RS_flow, 5 * REUV)), 250)
            u = FS.get_parker_wind(r, cs, RS_flow)
    
        # this else statement does not mean core-powered mass loss.
        # this RS_flow is the "hot" one. It can live in the bolometrically heated region
        # and still lead to a photoevaporative mass loss, as long as the "cold" sonic point
        # is outside the REUV.
        else:
            r = np.logspace(np.log10(REUV), np.log10(max(5 * RS_flow, 5 * REUV)), 250)
            constant = (1. - 4. * np.log(REUV / RS_flow) - 4. * (RS_flow / REUV) - 1e-13)
            u = FS.get_parker_wind_const(r, cs, RS_flow, constant)

        rho = (RS_flow / r)**2 * (cs / u)
        tau = np.fabs(np.trapz(rho[::-1], r[::-1]))
        # rho_s = 1. / ((sigma_EUV / (mmw_HHe * m_H / 2.)) * tau)               # <----- H2/He in outflow
        # rho_s = 1. / ((sigma_EUV / (mmw_H2O_outflow * m_H / 2.)) * tau)       # <----- H2O in outflow (dissociated)
        rho_s = 1. / ((sigma_EUV / (mmw_HHe_H2O_outflow * m_H / 2.)) * tau) # <----- H2/He & H2O outflow (dissociated)
        rho *= rho_s

        Mdot = 4 * np.pi * REUV**2 * rho[0] * u[0]

        return Mdot

    def compute_sound_speed(self, REUV, m_planet):
        """
        Calculate the energy-limited mass loss rate and corresponding sound speed.
        """
        G, FEUV, eff = self.params.G, self.params.get_param('FEUV'), self.params.eff
        Mdot_EL = eff * np.pi * REUV**3 / (4 * G * m_planet) * FEUV

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

    ### Momentum balance for EL and RL regimes ###
        
    def compare_densities_EL(self, REUV, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon):
        """
        Momentum balance for the energy-limited (EL) case.
        """
        G, alpha_rec = self.params.G, self.params.alpha_rec

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
        time_scale_recom = np.sqrt(Hflow / (FEUV_photon * alpha_rec))
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
        Momentum balance for the radiation-limited (RL) case.
        """
        G, m_H, alpha_rec = self.params.G, self.params.m_H, self.params.alpha_rec

        rho_EUV = rho_photo * np.exp((G * m_planet / cs_eq**2) * (1. / REUV - 1. / r_planet))

        cs_RL = 1.2e6  # Sound speed for T ~ 10^4 K
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
        REUV solution for the radiation-limited (RL) case.
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

    def compute_mass_loss_parameters(self, m_planet, r_planet, teq):
        """
        Compute REUV, Mdot, cs, and classify regimes for mass loss.
        """
        mass_loss_results = []
        for m_planet, r_planet, teq in zip(m_planet, r_planet, teq):
            try:
                FEUV = self.params.get_param('FEUV')
                E_photon = self.params.E_photon
                FEUV_photon = FEUV / E_photon
                alpha_rec = self.params.alpha_rec
                G = self.params.G
                k_b = self.params.k_b

                m_H = self.params.m_H

                mmw_HHe = self.params.mmw_HHe
                mmw_H2O = self.params.mmw_H2O
                mmw_HHe_H2O = self.params.mmw_HHe_H2O

                kappa_p_HHe = self.params.kappa_p_HHe
                kappa_p_H2O = self.params.kappa_p_H2O
                kappa_p_HHe_H2O = self.params.kappa_p_HHe_H2O

                g = G * m_planet / r_planet**2
                # Fbol = 4. * 5.6705e-5 * teq**4

                ### for H
                # cs_eq = np.sqrt((k_b * teq) / (m_H * mmw_HHe)) # <---- HHe in bolometrically heated region (non-dissociated)
                # rho_photo = g / (kappa_p_HHe * cs_eq**2)

                ### for H2O
                # cs_eq = np.sqrt((k_b * teq) / (m_H * mmw_H2O)) # <---- H2O in bolometrically heated region (non-dissociated)
                # rho_photo = g / (kappa_p_H2O * cs_eq**2)

                ### for HHe and H2O
                cs_eq = np.sqrt((k_b * teq) / (m_H * mmw_HHe_H2O)) # <---- HHe and H2O in bolometrically heated region (non-dissociated)
                rho_photo = g / (kappa_p_HHe_H2O * cs_eq**2)

                result = {'m_planet': m_planet, 'r_planet': r_planet, 'Teq': teq}

                # Energy-limited (EL) regime calculations
                REUV_solution_EL, time_scale_ratio, rho_EUV_EL, rho_flow_EL = self.find_REUV_solution_EL(r_planet, m_planet, rho_photo, cs_eq, FEUV_photon)
                cs_use = self.compute_sound_speed(REUV_solution_EL, m_planet)
                cs_use = min(cs_use, 1.2e6)
                Mdot_EL = self.compute_mdot_only(cs_use, REUV_solution_EL, m_planet)
                RS_flow = G * m_planet / (2. * cs_use**2)

                result.update({
                    'REUV': REUV_solution_EL,
                    'cs': cs_use,
                    'Mdot': Mdot_EL,
                    'RS_flow': RS_flow,
                    'rho_EUV': rho_EUV_EL,
                    'rho_flow': rho_flow_EL,
                    'time_scale_ratio': time_scale_ratio,
                    'regime': 'EL',
                })

                # Recombination-limited (RL) regime check
                if time_scale_ratio < 1:
                    REUV_solution_RL, rho_EUV_RL, rho_flow_RL = self.find_REUV_solution_RL(r_planet, m_planet, rho_photo, cs_eq, FEUV_photon)
                    cs_use_RL = 1.2e6
                    Hflow_RL = min(REUV_solution_RL / 3, cs_use_RL**2 * REUV_solution_RL**2 / (2 * G * m_planet))
                    nb = np.sqrt(FEUV_photon / (alpha_rec * Hflow_RL))
                    Rs_RL = G * m_planet / (2. * cs_use_RL**2)
                    u_RL = (FS.get_parker_wind_single(REUV_solution_RL, cs_use_RL, Rs_RL)
                        if REUV_solution_RL <= Rs_RL
                        else cs_use_RL)
                    # Mdot_RL = 4 * np.pi * REUV_solution_RL**2 * m_H * nb * u_RL
                    Mdot_RL = self.compute_mdot_only(cs_use_RL, REUV_solution_RL, m_planet)

                    result.update({
                        'REUV': REUV_solution_RL,
                        'cs': cs_use_RL,
                        'Mdot': Mdot_RL,
                        'RS_flow': Rs_RL,
                        'rho_REUV': rho_EUV_RL,
                        'rho_flow': rho_flow_RL,
                        'regime': 'RL',
                    })

                if result['RS_flow'] >= result['REUV']:
                    mass_loss_results.append(result)
                else:
                    print(
                        f"Warning: Check for CPML. RS_flow ({result['RS_flow']:.2e}) < REUV ({result['REUV']:.2e}) "
                        f"for planet mass={m_planet/self.params.mearth:.2f} M_earth"
                    )
                    mass_loss_results.append(result)

            except Exception as e:
                print(f"Error for planet {m_planet/self.params.mearth:.2f} Mearth and radius {r_planet/self.params.rearth:.2f} Rearth: {e}")
        
        return mass_loss_results