import numpy as np
from Parameters import ModelParams

class FractionationPhysics:
    """
    Contains various physical models for fractionation. This includes different methods for calculating x_O.

    In this model, we differentiate between a pure H2O, and a mix of H/He-H2O fractionation in 3 different places. 
    They are pointed out by arrows in comment form. Make sure to change each time.
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

        if x_O < 0:
            print(f"x_O < 0, setting to 0. (Original x_O = {x_O:.2e})")
            x_O = 0
        elif x_O > 1:
            print(f"x_O > 1, setting to 1. (Original x_O = {x_O:.2e})")
            x_O = 1
        
        return x_O

    def compute_xO_H2O_zahnle1986_reduced(self, phi_H, REUV, b_i, m_planet, T_outflow, mass_diff_O_H):
        """Compute x_O when losing H and O using equation 18 from Zahnle & Kasting 1986."""
        G, k_b = self.params.G, self.params.k_b
        numerator = G * m_planet * mass_diff_O_H * b_i
        denominator = phi_H * REUV**2 * k_b * T_outflow

        x_O = 1 - (numerator / denominator)

        if x_O < 0:
            print(f"x_O < 0, setting to 0. (Original x_O = {x_O:.2e})")
            x_O = 0
        elif x_O > 1:
            print(f"x_O > 1, setting to 1. (Original x_O = {x_O:.2e})")
            x_O = 1

        return x_O
    

class Fractionation:
    """
    Contains functions for fractionation framework.
    """
    def __init__(self, params):
        self.params = params
        self.physics = FractionationPhysics(params)

    def compute_T_outflow(self, cs):
        """
        Calculate the temperature T_outflow that corresponds to the REUV radius and the outflow region, with a specific sound speed cs.
        Assumes ideal gas law.
        """
        # mmw_outflow = self.params.mmw_H2O_outflow       # <----------- for dissociated H2O atmosphere
        mmw_outflow = self.params.mmw_HHe_H2O_outflow   # <----------- for dissociated HHe + H2O atmosphere
        m_H = self.params.m_H
        T_outflow = (cs**2 * m_H * mmw_outflow) / self.params.k_b # Kelvin

        return T_outflow
    
    def compute_fractionation_params(self, cs, REUV, Mdot):
        """
        Compute all necessary parameters for fractionation.
        """
        T_outflow = self.compute_T_outflow(cs)
        b_i = 4.8e17 * T_outflow ** 0.75                    # cm^-1 s^-1, diffusion coefficient for O in H from Zahnle & Kasting 1986
        mass_diff_O_H = self.params.m_O - self.params.m_H   # grams
        flux_total = Mdot / (4 * np.pi * REUV**2)           # g cm^-2 s^-1

        # ----- Pure H2O case (dissociated -> 2H + O)       <--------------------------
        # N_H = 1                                             # Hydrogen reservoir, free particle numbers (per unit mass)
        # N_O = N_H / 2                                       # For H2O, the reservoir of O is half of the reservoir of H
        # reservoir_ratio = N_O / N_H                         # needed ratio if we solve for φ_k in eq.2 Zahnle & Kasting 1986

        # ----- For dissociated H/He and dissociated H2O    <--------------------------
        X_HHe = self.params.X_HHe                           # H/He mass fraction  (dissociates to free H; max mmw_HHe_outflow = 1)
        X_H2O = self.params.X_H2O                           # Water mass fraction (dissociates to 2H + O; max mmw_H2O_outflow = 6)
        N_H_HHe = X_HHe / self.params.mmw_HHe_outflow       # eg 0.9 / 1 = 0.9.     free particle numbers (per unit mass). H/He contributes free hydrogen (ignore He)
        N_total_H2O = X_H2O / self.params.mmw_H2O_outflow   # eg 0.1 / 6 = 0.01667. total number of free particles from water
        # Water dissociates as: 2 H + 1 O, so partition the free particles:
        N_H_H2O = (2/3) * N_total_H2O                       # hydrogen from water, e.g. ≈ 0.01111
        N_O_H2O = (1/3) * N_total_H2O                       # oxygen from water, e.g. ≈ 0.00556
        # Total free hydrogen in the outflow is from both components
        N_H_total = N_H_HHe + N_H_H2O                       # e.g. 0.9 + 0.01111 ≈ 0.91111
        # The oxygen comes solely from water
        N_O_total = N_O_H2O                                 # e.g. ≈ 0.00556
        # And finally
        reservoir_ratio = N_O_total / N_H_total             # e.g. ~0.00556 / 0.91111 ≈ 0.0061
    
        return b_i, mass_diff_O_H, flux_total, reservoir_ratio

    def iterative_fractionation(self, flux_total, REUV, m_planet, T_outflow, b_i, mass_diff_O_H, reservoir_ratio, method="H_O_zahnle1986", tol=1e-30, max_iter=1000):
        """
        Iteratively solve for phi_O given physical constraints.
        Ensure phi_O + phi_H = flux_total at the end.
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
                if abs(phi_H + phi_O_new - flux_total) > tol:
                    print(f"Iteration {iteration+1}: Final phi_O + phi_H does not match flux_total. Adjusting.")
                    phi_O_new = flux_total - phi_H
                # print(f"Iteration {iteration+1}: Converged with flux total = {flux_total}, phi_O = {phi_O_new}, phi_H = {phi_H} for planet with mass {m_planet/params.mearth:.2f} M_earth.")
                return phi_O_new, phi_H, x_O

            phi_O = phi_O_new

        print("Maximum iterations reached without convergence of phi_O + phi_H.")
        return None, None, None

    def execute_fractionation(self, mass_loss_results, misc):
        if params is None:
            params = self.params

        for result in mass_loss_results:
            m_planet = result['m_planet']
            Teq = result['Teq']
            REUV = result['REUV']
            cs = result['cs']
            Mdot = result['Mdot']

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
                print(f"Skipping planet {m_planet/params.mearth:.1f} M_earth: Invalid fractionation values.")
                continue

            print(f"Planet with mass={m_planet/params.mearth:.2f} M_earth results: Mdot = {Mdot}")
            print(f"Planet with mass={m_planet/params.mearth:.2f} M_earth results: phi_O = {phi_O}, phi_H = {phi_H}, x_O = {x_O}\n")

            result.update({
                'T_outflow': T_outflow,
                'P_EUV': P_EUV,
                'R_b': R_b,
                'phi_O': phi_O,
                'phi_H': phi_H,
                'x_O': x_O
            })

        return mass_loss_results

    def execute_self_consistent_fractionation(self, mass_loss_results, mass_loss, misc, params, tol=1e-5, max_iter=1000):
        """
        Iteratively update mmw_outflow until convergence.
        """
        for result in mass_loss_results:
            m_planet = result['m_planet']
            Teq = result['Teq']
            REUV = result['REUV']
            cs = result['cs']
            Mdot = result['Mdot']

            T_outflow = self.compute_T_outflow(cs)
            if Teq >= T_outflow:
                print(f"Excluded: T_eq ({Teq:.2f} K) > T_outflow ({T_outflow:.2f} K) for planet mass={m_planet/params.mearth:.2f} M_earth.")
                continue
            
            # mmw_outflow = params.get_param('mmw_H2O_outflow')       # <----------- H2O, start with value from parameter file
            mmw_outflow = params.get_param('mmw_HHe_H2O_outflow')   # <----------- HHe & H2O, start with value from parameter file
            prev_mmw_outflow = None # track for convergence
            
            for iteration in range(max_iter): # iterative loop to self-consistently update mmw_outflow
                # ---------- P at R_EUV, and Bondi radius ----------
                T_outflow = self.compute_T_outflow(cs)
                P_EUV = misc.calculate_pressure_ideal_gas(result['rho_EUV'], T_outflow)
                R_b = misc.calculate_R_b(m_planet, cs)
                
                # ---------- Fractionation model ----------
                b_i, mass_diff, flux_total, reservoir_ratio = self.compute_fractionation_params(cs, REUV, Mdot)
                phi_O, phi_H, x_O = self.iterative_fractionation(flux_total, REUV, m_planet, T_outflow, b_i, mass_diff, reservoir_ratio)

                if phi_O is None or phi_H is None:
                    print(f"Skipping planet {m_planet/params.mearth:.1f} M_earth: Invalid fractionation values.")
                    continue

                # ---------- Make it self consistent ----------
                # **Update mmw_outflow using O/H ratio**
                mmw_outflow = (phi_H * params.am_h + phi_O * params.am_o) / (phi_H + phi_O)
                
                # **Check Convergence**
                if prev_mmw_outflow is not None and abs(mmw_outflow - prev_mmw_outflow) / prev_mmw_outflow < tol:
                    print(f"Converged in {iteration+1} iterations for planet {m_planet/params.mearth:.1f} M_earth.")
                    print(f"Final mmw_outflow = {mmw_outflow:.1f}")
                    break  # stop iterating if mmw_outflow has converged

                prev_mmw_outflow = mmw_outflow  # store for next iteration

                 # **Update mmw_outflow in params**
                params.update_param('mmw_HHe_H2O_outflow', mmw_outflow)
                
                # **Recompute Mdot with updated mmw_outflow, and cs until Mdot matches Mdot_EL**
                Mdot = mass_loss.compute_mdot_only(cs, REUV, m_planet)
                cs = mass_loss.compute_sound_speed(REUV, m_planet)

            if phi_O is None or phi_H is None:
                continue

            print(f"Planet with mass={m_planet/params.mearth:.2f} M_earth results: Mdot = {Mdot}")
            print(f"Planet with mass={m_planet/params.mearth:.2f} M_earth results: phi_O = {phi_O}, phi_H = {phi_H}, x_O = {x_O}\n")

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
                # updated RXUV
            })

        return mass_loss_results