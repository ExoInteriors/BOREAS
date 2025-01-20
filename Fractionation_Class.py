import numpy as np

class Fractionation:
    def __init__(self, params):
        """
        Initialize the Fractionation class with model parameters.
        :param params: An instance of the ModelParams class or similar object containing constants.
        """
        self.params = params

    def preprocess_fractionation_params(self, cs, REUV, Mdot, Teq):
        """
        Preprocess parameters for the fractionation calculation.
        """
        mmw_H = self.params.mmw_H
        m_H = self.params.m_H
        m_O = self.params.m_O
        T_REUV = (cs**2 * m_H * mmw_H) / self.params.k_b # Kelvin

        if Teq > T_REUV:
            raise ValueError(f"Solution excluded: T_eq ({Teq} K) exceeds T_REUV ({T_REUV} K).")

        b_i = 4.8e17 * T_REUV ** 0.75   # cm^-1 s^-1
        mass_difference = m_O - m_H     # grams
        flux_total = Mdot / (4 * np.pi * REUV**2) # g cm^-2 s^-1

        N_H = 1                         # Hydrogen reservoir
        N_O = N_H / 2                   # For H2O, the reservoir of O is half of the reservoir of H
        reservoir_ratio = N_O / N_H     # needed ratio if we solve for φ_k in eq.2 Zahnle & Kasting 1986
    
        return T_REUV, b_i, mass_difference, flux_total, reservoir_ratio


    def iterative_fractionation(self, flux_total, REUV, m_planet, T_REUV, b_i, mass_difference, reservoir_ratio, tol=1e-30, max_iter=20):
        """
        Iteratively solve for phi_O given physical constraints.
        Ensure phi_O + phi_H = flux_total at the end.
        """
        phi_O = flux_total / 2  # Initial guess for phi_O

        for iteration in range(max_iter):
            phi_H = flux_total - phi_O
            if phi_H <= 0:
                print(f"Iteration {iteration}: Non-physical phi_H = {phi_H}. Stopping.")
                return None, None, None

            G, k_b = self.params.G, self.params.k_b
            numerator = ((phi_H * REUV**2) / b_i) - ((G * m_planet * mass_difference) / (k_b * T_REUV))
            denominator = numerator * np.exp(-numerator / REUV)
            
            x_O = numerator / denominator  # Zahnle & Kasting equation 14
            if x_O <= 0:
                print(f"Iteration {iteration}: Non-physical x_O = {x_O}. Stopping.")
                return None, None, None

            phi_O_new = x_O * phi_H * reservoir_ratio  # Update phi_O

            # Check convergence
            if abs(phi_O_new - phi_O) < tol:
                phi_H = flux_total - phi_O_new  # Recalculate phi_H
                if abs(phi_H + phi_O_new - flux_total) > tol:
                    print(f"Iteration {iteration}: Final phi_O + phi_H does not match flux_total. Adjusting.")
                    phi_O_new = flux_total - phi_H  # Enforce constraint
                print(f"Iteration {iteration}: Converged with flux total = {flux_total}, phi_O = {phi_O_new}, phi_H = {phi_H}.")
                return phi_O_new, phi_H, x_O

            phi_O = phi_O_new  # Update for the next iteration

        print("Maximum iterations reached without convergence.")
        return None, None, None