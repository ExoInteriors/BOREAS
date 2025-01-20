import numpy as np
from scipy.optimize import brentq
from Flow_Solutions_Class import FlowSolutions as FS

class MassLoss:
    def __init__(self, params):
        """
        Initialize the MassLoss class with model parameters.
        :param params: An instance of the ModelParams class or a similar object containing constants.
        """
        self.params = params

    ### Mass loss ###
        
    def compute_mdot_only(self, cs, REUV, m_planet):
        """
        Direct computation of the mass-loss rate (Mdot).
        """
        G, sigma_EUV, mmw_H, m_H = self.params.G, self.params.sigma_EUV, self.params.mmw_H, self.params.m_H
        RS_flow = G * m_planet / (2. * cs**2)


        if RS_flow >= REUV:
            r = np.logspace(np.log10(REUV), np.log10(max(5 * RS_flow, 5 * REUV)), 250)
            u = FS.get_parker_wind(r, cs, RS_flow)
            rho = (RS_flow / r)**2 * (cs / u)
            tau = np.fabs(np.trapz(rho[::-1], r[::-1]))
            rho_s = 1. / ((sigma_EUV / (mmw_H * m_H / 2.)) * tau)
            rho *= rho_s
            Mdot = 4 * np.pi * REUV**2 * rho[0] * u[0]
        else:
            r = np.logspace(np.log10(REUV), np.log10(max(5 * RS_flow, 5 * REUV)), 250)
            constant = (1. - 4. * np.log(REUV / RS_flow) - 4. * (RS_flow / REUV) - 1e-13)
            u = FS.get_parker_wind_const(r, cs, RS_flow, constant)
            rho = (RS_flow / r)**2 * (cs / u)
            tau = np.fabs(np.trapz(rho[::-1], r[::-1]))
            rho_s = 1. / ((sigma_EUV / (mmw_H * m_H / 2.)) * tau)
            rho *= rho_s
            Mdot = 4 * np.pi * REUV**2 * rho[0] * u[0]
        return Mdot

    def compute_sound_speed(self, REUV, m_planet):
        """
        Calculate the energy-limited mass loss rate and corresponding sound speed.
        """
        G, FEUV, eff = self.params.G, self.params.FEUV, self.params.eff
        Mdot_EL = eff * np.pi * REUV**3 / (4 * G * m_planet) * FEUV

        lower_bound_initial, upper_bound = 2e5, 1e13

        def mdot_difference(cs):
            return (self.compute_mdot_only(cs, REUV, m_planet) - Mdot_EL) / (Mdot_EL + 1)

        f1 = mdot_difference(lower_bound_initial)
        if f1 < 0:
            return brentq(mdot_difference, lower_bound_initial, upper_bound)
        else:
            while f1 > 0:
                lower_bound_initial *= 0.95
                f1 = mdot_difference(lower_bound_initial)
            return brentq(mdot_difference, lower_bound_initial, upper_bound / 10)

    ### Momentum balance ###
        
    def compare_densities_EL(self, REUV, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon):
        """
        Momentum balance for the energy-limited (EL) case.
        """
        G = self.params.G
        alpha_rec = self.params.alpha_rec

        rho_EUV = rho_photo * np.exp((G * m_planet / cs_eq**2) * (1. / REUV - 1. / r_planet))
        cs_use = self.compute_sound_speed(REUV, m_planet)
        cs_use = min(cs_use, 1.2e6) # Limit sound speed to T ~ 10^4 K
        Mdot = self.compute_mdot_only(cs_use, REUV, m_planet)
        Rs = G * m_planet / (2. * cs_use**2)

        # Determine outflow parameters
        if REUV <= Rs:
            u_launch = FS.get_parker_wind_single(REUV, cs_use, Rs)
            u_launch_for_grad = FS.get_parker_wind_single(1.001 * REUV, cs_use, Rs)
            Hflow = 0.001 * REUV / (np.log(1. / REUV**2 / u_launch) - np.log(1. / (1.001 * REUV)**2 / u_launch_for_grad))
        else:
            u_launch = cs_use
            Hflow = REUV

        # Timescales
        time_scale_flow = Hflow / u_launch
        time_scale_recom = np.sqrt(Hflow / (FEUV_photon * alpha_rec))
        time_scale_ratio = time_scale_recom / time_scale_flow
        if cs_use < 1.2e6:
            time_scale_ratio = 10.

        # Momentum balance
        rho_flow = Mdot / (4 * np.pi * r_planet**2 * u_launch)
        diff = (rho_EUV * cs_eq**2 - rho_flow * (u_launch**2 + cs_use**2)) / (2. * rho_EUV * cs_eq**2)
        return diff, time_scale_ratio

    def find_REUV_solution_EL(self, r_planet, m_planet, rho_photo, cs_eq, FEUV_photon):
        """
        REUV solution for the energy-limited (EL) case.
        """
        REUV_lower_bound = r_planet * 1.001
        REUV_upper_bound = r_planet * 6

        def root_function(REUV):
            diff, _ = self.compare_densities_EL(REUV, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon)
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
        _, time_scale_ratio = self.compare_densities_EL(REUV_solution, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon)
        return REUV_solution, time_scale_ratio

    def compare_densities_RL(self, REUV, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon):
        """
        Momentum balance for the radiation-limited (RL) case.
        """
        G = self.params.G
        m_H = self.params.m_H
        alpha_rec = self.params.alpha_rec

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

        # Momentum balance
        diff = (rho_EUV * cs_eq**2 - rho_flow * (u_RL**2 + cs_RL**2)) / (2. * rho_EUV * cs_eq**2)
        return diff

    def find_REUV_solution_RL(self, r_planet, m_planet, rho_photo, cs_eq, FEUV_photon):
        """
        REUV solution for the radiation-limited (RL) case.
        """
        REUV_lower_bound = r_planet * 1.001
        REUV_upper_bound = r_planet * 6

        def root_function(REUV):
            return self.compare_densities_RL(REUV, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon)

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
        return REUV_solution

    ### Run mass loss model ###
    def compute_mass_loss_parameters(self, m_planet, r_planet, teq):
        """
        Compute REUV, Mdot, cs, and classify regimes for mass loss.
        Also filters solutions where RS_flow < REUV.
        """
        results = []
        for m_planet, r_planet, teq in zip(m_planet, r_planet, teq):
            try:
                FEUV = self.params.FEUV
                alpha_rec = self.params.alpha_rec
                E_photon = self.params.E_photon
                FEUV_photon = FEUV / E_photon
                G = self.params.G
                k_b = self.params.k_b
                mmw_eq = self.params.mmw_eq
                m_H = self.params.m_H
                kappa_p = self.params.kappa_p

                g = G * m_planet / r_planet**2
                Fbol = 4. * 5.6705e-5 * teq**4
                cs_eq = np.sqrt((k_b * teq) / (m_H * mmw_eq))
                rho_photo = g / (kappa_p * cs_eq**2)

                result = {'mass': m_planet, 'radius': r_planet, 'teq': teq}

                # Energy-limited (EL) regime calculations
                REUV_solution_EL, time_scale_ratio = self.find_REUV_solution_EL(r_planet, m_planet, rho_photo, cs_eq, FEUV_photon)
                cs_use = self.compute_sound_speed(REUV_solution_EL, m_planet)
                cs_use = min(cs_use, 1.2e6)
                Mdot_EL = self.compute_mdot_only(cs_use, REUV_solution_EL, m_planet)
                RS_flow = G * m_planet / (2. * cs_use**2)

                result.update({
                    'REUV': REUV_solution_EL,
                    'cs': cs_use,
                    'Mdot': Mdot_EL,
                    'RS_flow': RS_flow,
                    'time_scale_ratio': time_scale_ratio,
                    'regime': 'EL',
                })

                # Recombination-limited (RL) regime check
                if time_scale_ratio < 1:
                    REUV_solution_RL = self.find_REUV_solution_RL(r_planet, m_planet, rho_photo, cs_eq, FEUV_photon)
                    cs_use_RL = 1.2e6
                    Hflow_RL = min(REUV_solution_RL / 3, cs_use_RL**2 * REUV_solution_RL**2 / (2 * G * m_planet))
                    nb = np.sqrt(FEUV_photon / (alpha_rec * Hflow_RL))
                    Rs_RL = G * m_planet / (2. * cs_use_RL**2)
                    u_RL = (FS.get_parker_wind_single(REUV_solution_RL, cs_use_RL, Rs_RL)
                        if REUV_solution_RL <= Rs_RL
                        else cs_use_RL)
                    Mdot_RL = 4 * np.pi * REUV_solution_RL**2 * m_H * nb * u_RL

                    result.update({
                        'REUV': REUV_solution_RL,
                        'cs': cs_use_RL,
                        'Mdot': Mdot_RL,
                        'RS_flow': Rs_RL,
                        'regime': 'RL',
                    })

                # Only move on with results if photoevaporation happens: RS_flow < REUV
                if result['RS_flow'] >= result['REUV']:
                    results.append(result)
                else:
                    print(
                        f"Excluded: RS_flow ({result['RS_flow']:.2e}) < REUV ({result['REUV']:.2e}) "
                        f"for planet mass={m_planet/self.params.mearth:.2f} M_earth, "
                        f"radius={r_planet/self.params.rearth:.2f} R_earth."
                    )

            except Exception as e:
                print(f"Error for planet mass={m_planet}, planet radius={r_planet}: {e}")
        
        return results