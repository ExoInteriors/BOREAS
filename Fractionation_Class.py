import numpy as np

class FractionationPhysics:
    """
    Physical models for oxygen fractionation; used by Fractionation.
    """

    def __init__(self, params):
        self.params = params

    def compute_xO_H2O_zahnle1986(self, phi_H, REUV, b_i, m_planet, T_outflow, mass_diff_O_H):
        """Compute x_O when losing H and O using equation 14 from Zahnle & Kasting 1986."""
        G, k_b = self.params.G, self.params.k_b
        numerator = ((phi_H * REUV**2) / b_i) - ((G * m_planet * mass_diff_O_H) / (k_b * T_outflow))
        denominator = numerator * np.exp(-numerator / REUV)
        if denominator == 0:
            print("Denominator in x_O calculation is zero. Returning x_O = 0.")
            return 0
        x_O = numerator / denominator

        return np.clip(x_O, 0.0, 1.0)

    def compute_xO_H2O_zahnle1986_reduced(self, phi_H, REUV, b_i, m_planet, T_outflow, mass_diff_O_H):
        """Compute x_O when losing H and O using equation 18 from Zahnle & Kasting 1986."""
        G, k_b = self.params.G, self.params.k_b
        numerator = G * m_planet * mass_diff_O_H * b_i
        denominator = phi_H * REUV**2 * k_b * T_outflow
        x_O = 1 - (numerator / denominator)

        return np.clip(x_O, 0.0, 1.0)

class Fractionation:
    """
    Self-consistent fractionation loop; applies only for H2O or mixed atmospheres.
    """
    def __init__(self, params):
        self.params = params
        self.physics = FractionationPhysics(params)

    def compute_T_outflow(self, cs):
        """
        Calculate the temperature T_outflow that corresponds to the REUV radius and the outflow region. Assumes ideal gas law.
        """
        mode = self.params.outflow_mode
        m_H = self.params.m_H
        if mode == 'H2O':
            μ = self.params.mmw_H2O_outflow
        elif mode == 'HHe_H2O':
            μ = self.params.mmw_HHe_H2O_outflow
        else:
            raise ValueError(f"Mode '{mode}' does not support fractionation")
        return cs**2 * m_H * μ / self.params.k_b
    
    def compute_fractionation_params(self, cs, REUV, Mdot):
        """
        Compute all necessary parameters for fractionation.
        """
        T_outflow = self.compute_T_outflow(cs)
        b_i = 4.8e17 * T_outflow ** 0.75                    # cm^-1 s^-1, diffusion coefficient for O in H from Zahnle & Kasting 1986
        mass_diff = self.params.m_O - self.params.m_H       # grams
        flux_total = Mdot / (4 * np.pi * REUV**2)           # g cm^-2 s^-1

        mode = self.params.outflow_mode
        if mode == 'H2O':
            N_H, N_O = 1.0, 0.5 # Reservoirs, free particle numbers (per unit mass). For H2O, the reservoir of O is half of the reservoir of H
        elif mode == 'HHe_H2O':
            X_HHe, X_H2O = self.params.X_HHe, self.params.X_H2O
            N_HHe = X_HHe / self.params.mmw_HHe_outflow
            N_H2O_total = X_H2O / self.params.mmw_H2O_outflow
            N_H = N_HHe + (2/3) * N_H2O_total
            N_O = (1/3) * N_H2O_total
        else:
            raise ValueError(f"Mode '{mode}' does not support fractionation")
        
        reservoir_ratio = N_O / N_H

        return b_i, mass_diff, flux_total, reservoir_ratio

    def iterative_fractionation(self, flux_total, REUV, m_planet, T_outflow, b_i, mass_diff_O_H, reservoir_ratio,
                                method="H_O_zahnle1986", tol=1e-5, max_iter=1000):
        """
        Iteratively solve for phi_O given physical constraints. Ensure phi_O + phi_H = flux_total at the end.
        """
        phi_O = flux_total / 2 # initial guess

        for iteration in range(max_iter):
            phi_H = flux_total - phi_O
            if phi_H <= 0:
                print(f"Iteration {iteration+1}: Non-physical phi_H = {phi_H}. Stopping.")
                return None, None, None

            if method == "H_O_zahnle1986":
                x_O = self.physics.compute_xO_H2O_zahnle1986(phi_H, REUV, b_i, m_planet, T_outflow, mass_diff_O_H)
            elif method == "H_O_zahnle_reduced":
                x_O = self.physics.compute_xO_H2O_zahnle1986_reduced(phi_H, REUV, b_i, m_planet, T_outflow, mass_diff_O_H)
            else:
                raise ValueError(f"Invalid method '{method}'. Check spelling for methods.")

            if x_O <= 0:
                print(f"Iteration {iteration+1}: Non-physical x_O = {x_O}. Stopping.")
                return None, None, None

            phi_O_new = x_O * phi_H * reservoir_ratio

            # Check convergence
            if abs(phi_O_new - phi_O) < tol:
                phi_H = flux_total - phi_O_new
                return phi_O_new, phi_H, x_O

            phi_O = phi_O_new

        print("Maximum iterations reached without convergence of phi_O + phi_H.")
        return None, None, None

    def execute_self_consistent_fractionation(self, mass_loss_results, mass_loss, misc, params, 
                                              tol=1e-5, max_iter=1000):
        """
        Iteratively update mmw_outflow until convergence.
        """
        mode = self.params.outflow_mode

        if mode == 'HHe':
            return mass_loss_results
        
        results = []

        for sol in mass_loss_results:
            # skip cases where equilibrium temperature exceeds outflow temperature
            T_eq = sol.get('Teq', None)
            cs_init = sol['cs']
            T_out_init = self.compute_T_outflow(cs_init)
            if T_eq is not None and T_eq >= T_out_init:
                print(f"Excluded: T_eq ({T_eq:.2f} K) >= T_outflow ({T_out_init:.2f} K) for planet mass={sol['m_planet']/self.params.mearth:.2f} M_earth.")
                continue
            
            # initialize mmw_outflow
            if mode == 'H2O':
                init = self.params.mmw_H2O_outflow; key='mmw_H2O_outflow'
            elif mode == 'HHe_H2O':
                init = self.params.mmw_HHe_H2O_outflow; key='mmw_HHe_H2O_outflow'
            else:
                raise ValueError(f"Unknown outflow_mode '{mode}'")
            
            params.update_param(key, init)

            prev_mmw = None
            m_p, Teq, REUV, cs, Mdot = sol['m_planet'], sol['Teq'], sol['REUV'], sol['cs'], sol['Mdot']

            for iteration in range(max_iter):
                T_outflow = self.compute_T_outflow(cs)
                b_i, mass_diff, flux_tot, reservoir_ratio = self.compute_fractionation_params(cs, REUV, Mdot)
                phi_O, phi_H, x_O = self.iterative_fractionation(flux_tot, REUV, m_p, T_outflow, b_i, mass_diff, reservoir_ratio)
                if phi_O is None or phi_H is None:
                    print(f"Skipping planet {m_p/params.mearth:.1f} M_earth: Invalid fractionation values.")
                    continue
                new_mmw = (phi_H * self.params.am_h + phi_O * self.params.am_o)/(phi_H + phi_O)
                if prev_mmw and abs(new_mmw-prev_mmw)/prev_mmw<tol:
                    print(f"Converged in {iteration+1} iterations for planet {m_p/params.mearth:.1f} M_earth.")
                    print(f"Final mmw_outflow = {new_mmw:.1f}")
                    break
                prev_mmw = new_mmw
                params.update_param(key, new_mmw)
                Mdot = mass_loss.compute_mdot_only(cs, REUV, m_p)
                cs   = mass_loss.compute_sound_speed(REUV, m_p)
                
            # final recompute
            T_out = self.compute_T_outflow(cs)
            P_EUV = misc.calculate_pressure_ideal_gas(sol['rho_EUV'], T_out)
            sol.update({'T_outflow':T_out,'P_EUV':P_EUV,'phi_O':phi_O,'phi_H':phi_H,'x_O':x_O,'mmw_outflow':prev_mmw,'Mdot':Mdot,'cs':cs})
            results.append(sol)

            print(f"Planet with mass={m_p/params.mearth:.2f} M_earth results: Mdot = {Mdot}")
            print(f"Planet with mass={m_p/params.mearth:.2f} M_earth results: phi_O = {phi_O}, phi_H = {phi_H}, x_O = {x_O}\n")

        return results