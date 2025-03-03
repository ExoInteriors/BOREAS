import numpy as np
from Parameters import ModelParams

class Fractionation:
    def __init__(self, params):
        """
        Initialize the Fractionation class with model parameters.
        :param params: An instance of the ModelParams class or similar object containing constants.
        """
        self.params = params

    def compute_T_outflow(self, cs):
        """
        Calculate the temperature T_outflow that corresponds to the REUV radius and the outflow region, with a specific sound speed cs.
        """
        mmw_outflow = self.params.mmw_outflow
        m_H = self.params.m_H
        T_outflow = (cs**2 * m_H * mmw_outflow) / self.params.k_b # Kelvin

        return T_outflow
    
    def compute_fractionation_params(self, cs, REUV, Mdot):
        """
        Compute all necessary parameters for fractionation.
        """
        T_outflow = self.compute_T_outflow(cs)
        b_i = 4.8e17 * T_outflow ** 0.75 # cm^-1 s^-1
        mass_difference = self.params.m_O - self.params.m_H # grams
        flux_total = Mdot / (4 * np.pi * REUV**2) # g cm^-2 s^-1

        N_H = 1                         # Hydrogen reservoir
        N_O = N_H / 2                   # For H2O, the reservoir of O is half of the reservoir of H
        reservoir_ratio = N_O / N_H     # needed ratio if we solve for φ_k in eq.2 Zahnle & Kasting 1986
    
        return b_i, mass_difference, flux_total, reservoir_ratio

    def iterative_fractionation(self, flux_total, REUV, m_planet, T_outflow, b_i, mass_difference, reservoir_ratio, tol=1e-30, max_iter=20):
        """
        Iteratively solve for phi_O given physical constraints.
        Ensure phi_O + phi_H = flux_total at the end.
        """
        phi_O = flux_total / 2  # initial guess

        for iteration in range(max_iter):
            phi_H = flux_total - phi_O
            if phi_H <= 0:
                print(f"Iteration {iteration+1}: Non-physical phi_H = {phi_H}. Stopping.")
                return None, None, None

            G, k_b = self.params.G, self.params.k_b
            numerator = ((phi_H * REUV**2) / b_i) - ((G * m_planet * mass_difference) / (k_b * T_outflow))
            denominator = numerator * np.exp(-numerator / REUV)
            
            x_O = numerator / denominator  # Zahnle & Kasting equation 14
            if x_O <= 0:
                print(f"Iteration {iteration+1}: Non-physical x_O = {x_O}. Stopping.")
                return None, None, None

            phi_O_new = x_O * phi_H * reservoir_ratio

            # Check convergence
            if abs(phi_O_new - phi_O) < tol:
                phi_H = flux_total - phi_O_new
                if abs(phi_H + phi_O_new - flux_total) > tol:
                    print(f"Iteration {iteration+1}: Final phi_O + phi_H does not match flux_total. Adjusting.")
                    phi_O_new = flux_total - phi_H
                # print(f"Iteration {iteration+1}: Converged with flux total = {flux_total}, phi_O = {phi_O_new}, phi_H = {phi_H}.")
                return phi_O_new, phi_H, x_O

            phi_O = phi_O_new

        print("Maximum iterations reached without convergence.")
        return None, None, None
    

    def execute_fractionation(self, mass_loss_results, misc):
        params = ModelParams()

        for result in mass_loss_results:
            m_planet = result['m_planet']
            r_planet = result['r_planet']
            Teq = result['Teq']

            REUV = result['REUV']
            cs = result['cs']
            Mdot = result['Mdot']

            # ---------- T & P at R_EUV, and Bondi radius ----------
            T_outflow = self.compute_T_outflow(cs)
            P_EUV = misc.calculate_pressure_ideal_gas(result['rho_EUV'], T_outflow)
            R_b = misc.calculate_R_b(m_planet, cs)

            if Teq >= T_outflow:
                print(f"Excluded: T_eq ({Teq:.2f} K) > T_outflow ({T_outflow:.2f} K) for planet mass={m_planet/params.mearth:.2f} M_earth.]")
                continue

            # ---------- Fractionation model ----------
            b_i, mass_diff, flux_total, reservoir_ratio = self.compute_fractionation_params(cs, REUV, Mdot)
            phi_O, phi_H, x_O = self.iterative_fractionation(flux_total, REUV, m_planet, T_outflow, b_i, mass_diff, reservoir_ratio)

            if phi_O is None or phi_H is None:
                print(f"Skipping planet {m_planet}: Invalid fractionation values.")
                continue

            print(f"Planet with mass={m_planet/params.mearth:.2f} M_earth results: Mdot = {Mdot}")
            print(f"Planet with mass={m_planet/params.mearth:.2f} M_earth results: phi_O = {phi_O}, phi_H = {phi_H}, x_O = {x_O}")

            result.update({
                'T_outflow': T_outflow,
                'P_EUV': P_EUV,
                'R_b': R_b,
                'phi_O': phi_O,
                'phi_H': phi_H,
                'x_O': x_O
            })

        return mass_loss_results

    def execute_self_consistent_fractionation(self, mass_loss_results, mass_loss, misc, params, tol=1e-5, max_iter=20):
        """
        Iteratively update mmw_outflow until convergence.
        """
        for result in mass_loss_results:
            m_planet = result['m_planet']
            r_planet = result['r_planet']
            Teq = result['Teq']
            REUV = result['REUV']
            cs = result['cs']
            Mdot = result['Mdot']
            
            mmw_outflow = params.get_param('mmw_outflow') # start with value from parameter file
            prev_mmw_outflow = None # track for convergence
            
            for iteration in range(max_iter):
                # ---------- T & P at R_EUV, and Bondi radius ----------
                T_outflow = self.compute_T_outflow(cs)
                P_EUV = misc.calculate_pressure_ideal_gas(result['rho_EUV'], T_outflow)
                R_b = misc.calculate_R_b(m_planet, cs)

                if Teq >= T_outflow:
                    print(f"Excluded: T_eq ({Teq:.2f} K) > T_outflow ({T_outflow:.2f} K) for planet mass={m_planet/params.mearth:.2f} M_earth.")
                    continue
                
                # ---------- Fractionation model ----------
                b_i, mass_diff, flux_total, reservoir_ratio = self.compute_fractionation_params(cs, REUV, Mdot)
                phi_O, phi_H, x_O = self.iterative_fractionation(flux_total, REUV, m_planet, T_outflow, b_i, mass_diff, reservoir_ratio)

                if phi_O is None or phi_H is None:
                    print(f"Skipping planet {m_planet}: Invalid fractionation values.")
                    continue

                # ---------- Make it self consistent ----------
                # **Update mmw_outflow using O/H ratio**
                X_H = phi_H / (phi_H + phi_O)
                X_O = phi_O / (phi_H + phi_O)
                mmw_outflow = X_H * params.am_h + X_O * params.am_o
                # OR
                mmw_outflow = (phi_H * params.am_h + phi_O * params.am_o) / (phi_H + phi_O)

                # **Check Convergence**
                if prev_mmw_outflow is not None and abs(mmw_outflow - prev_mmw_outflow) / prev_mmw_outflow < tol:
                    print(f"Converged in {iteration+1} iterations for planet {m_planet/params.mearth:.1f} M_earth.")
                    print(f"Final mmw_outflow = {mmw_outflow:.1f}")
                    break  # stop iterating if mmw_outflow has converged

                prev_mmw_outflow = mmw_outflow  # store for next iteration

                 # **Update mmw_outflow in params**
                params.update_param('mmw_outflow', mmw_outflow)
                
                # **Recompute Mdot with updated mmw_outflow**
                Mdot = mass_loss.compute_mdot_only(cs, REUV, m_planet)

                # **Recompute cs until Mdot matches Mdot_EL**
                cs = mass_loss.compute_sound_speed(REUV, m_planet)

            result.update({
                'T_outflow': T_outflow,
                'P_EUV': P_EUV,
                'R_b': R_b,
                'phi_O': phi_O,
                'phi_H': phi_H,
                'x_O': x_O,
                'mmw_outflow': mmw_outflow, # final mmw_outflow
                'Mdot': Mdot, # final mass loss rate
                'cs': cs # final sound speed
            })

        return mass_loss_results





    def mmw_outflow_residual(self, mmw_guess, mass_loss, cs, REUV, m_planet, r_planet, Teq, fractionation, params):
        """
        Residual function for the root finder.
        
        Given a guess for mmw_outflow, update the fractionation model and return
        the difference between the updated mmw_outflow and the guess.
        """
        params.update_param("mmw_outflow", mmw_guess)
        
        updated_mmw, _, _ = fractionation.update_mmw_outflow(mass_loss, cs, REUV, m_planet, r_planet, Teq)

        return updated_mmw - mmw_guess


    def update_mmw_outflow(self, mass_loss, cs, REUV, m_planet, r_planet, teq, tol_mmw=1e-3, tol_REUV=1e-3, max_iter=20):
        """
        Iteratively update mmw_outflow based on computed escape fractions.
        """
        self.params.update_param("mmw_outflow", (2 * self.params.am_h + self.params.am_o) / 9) # initial guess
        # the initial guess is the mean molecular weight (H and O) assuming full dissociation of water
        
        for iteration in range(max_iter):
            T_outflow = self.compute_T_outflow(cs)
            cs = np.sqrt(self.params.k_b * T_outflow / (self.params.m_H * self.params.get_param("mmw_outflow")))
            Mdot = mass_loss.compute_mdot_only(cs, REUV, m_planet)
            b_i, mass_difference, flux_total, reservoir_ratio = self.compute_fractionation_params(cs, REUV, Mdot)
            phi_O, phi_H, _ = self.iterative_fractionation(flux_total, REUV, m_planet, T_outflow, b_i, mass_difference, reservoir_ratio)

            if phi_O is None or phi_H is None:
                print("Fractionation failed to converge. Stopping iteration.")
                return None, cs, REUV

            X_H = phi_H / (phi_H + phi_O) # escaping H fraction
            X_O = phi_O / (phi_H + phi_O) # escaping H fraction
            new_mmw_outflow = X_H * self.params.am_h + X_O * self.params.am_o

            self.params.update_param("mmw_outflow", new_mmw_outflow)
            # print(f"Iteration {iteration+1}: Updated mmw_outflow = {self.params.get_param('mmw_outflow'):.3f}")

            FEUV = self.params.FEUV
            E_photon = self.params.E_photon
            FEUV_photon = FEUV / E_photon
            G = self.params.G
            k_b = self.params.k_b
            mmw_eq = self.params.mmw_eq
            m_H = self.params.m_H
            kappa_p = self.params.kappa_p
            g = G * m_planet / r_planet**2
            cs_eq = np.sqrt((k_b * teq) / (m_H * mmw_eq)) # H2O in bolometrically heated region
            rho_photo = g / (kappa_p * cs_eq**2)
            new_REUV, _, _, _ = mass_loss.find_REUV_solution_EL(r_planet, m_planet, rho_photo, cs_eq, FEUV_photon)

            if (abs(new_mmw_outflow - self.params.get_param("mmw_outflow")) < tol_mmw) and (abs(new_REUV - REUV) < tol_REUV):
                print(f"Planet x with mass {(m_planet/self.params.mearth):.2f} M_earth: Converged after {iteration+1} iterations with mmw_outflow = {self.params.get_param('mmw_outflow'):.1f}, REUV = {new_REUV:.1e}")
                return new_mmw_outflow, cs, new_REUV
        
            REUV = new_REUV

        print("Warning: mmw_outflow did not converge within max iterations.")
        return None, cs, REUV