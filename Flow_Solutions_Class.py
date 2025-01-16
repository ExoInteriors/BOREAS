import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw


class FlowSolutions:
    def __init__(self):
        """
        Initialize the FlowSolutions class. 
        Currently, no parameters are required to be stored as instance variables.
        """
        pass

    ### Isothermal Flow Routines ###

    @staticmethod
    def get_parker_wind(r, cs, Rs):
        """
        Calculates the velocity structure (in cgs) of the isothermal Parker wind solution.
        """
        f_r = 3.0 - 4.0 * Rs / r
        brackets = -(Rs / r)**4 * np.exp(f_r)
        u = np.zeros(r.size)

        u[r <= Rs] = np.sqrt(-np.real(lambertw(brackets[r <= Rs], 0))) * cs
        u[r > Rs] = np.sqrt(-np.real(lambertw(brackets[r > Rs], -1))) * cs

        return u

    @staticmethod
    def get_parker_wind_const(r, cs, Rs, const):
        """
        Calculates the velocity structure of the isothermal Parker wind solution with a constant parameter.
        """
        f_r = -const - 4.0 * Rs / r
        brackets = -(Rs / r)**4 * np.exp(f_r)
        u = np.zeros(r.size)

        u[r <= Rs] = np.sqrt(-np.real(lambertw(brackets[r <= Rs], 0))) * cs
        u[r > Rs] = np.sqrt(-np.real(lambertw(brackets[r > Rs], -1))) * cs

        return u

    @staticmethod
    def get_parker_wind_single(r, cs, Rs):
        """
        Calculates the velocity structure of the isothermal Parker wind solution for a single radius.
        """
        f_r = 3.0 - 4.0 * Rs / r
        brackets = -(Rs / r)**4 * np.exp(f_r)

        if r <= Rs:
            return np.sqrt(-np.real(lambertw(brackets, 0))) * cs
        else:
            return np.sqrt(-np.real(lambertw(brackets, -1))) * cs

    @staticmethod
    def get_parker_wind_single_const(r, cs, Rs, c):
        """
        Calculates the velocity structure of the isothermal Parker wind solution for a single radius with a constant parameter.
        """
        f_r = -c - 4.0 * Rs / r
        brackets = -(Rs / r)**4 * np.exp(f_r)

        if r <= Rs:
            return np.sqrt(-np.real(lambertw(brackets, 0))) * cs
        else:
            return np.sqrt(-np.real(lambertw(brackets, -1))) * cs

    ### Polytropic Flow Routines ###

    @staticmethod
    def get_rmin(n):
        """
        Calculates the minimum radius where u -> 0 for a polytropic wind with n < -1.
        """
        return 4 * (1 + 1.0 / n) / (1 - 2 * n + 3 / n)

    @staticmethod
    def U_eqn(U, n, r_Rs):
        """
        Computes the energy equation for dimensionless velocity U.
        """
        lhs = 0.5 * U**2 + (1 + 1 / n)**(1 / (2 * n)) * (n + 1) * U**(-1 / n) * (1 / r_Rs)**(2 / n) - 2 * (1 + 1 / n) * (1 / r_Rs)
        rhs = 0.5 * (2 * n - 3 / n - 1)
        return lhs - rhs

    @staticmethod
    def getflow_fromU(r_Rs, U, n):
        """
        Computes flow variables (density, pressure, sound speed) from U as a function of r_Rs.
        """
        rho = np.sqrt(1 + 1 / n) / (U * r_Rs**2)
        P = rho**(1 + 1 / n)
        cs = np.sqrt(P / rho)
        return rho, P, cs

    def get_dimensionless_flow_sol(self, r_Rs_grid, n):
        """
        Solves for dimensionless flow variables over a grid of radii.
        """
        Uout = np.zeros(r_Rs_grid.size)

        for i, r_Rs in enumerate(r_Rs_grid):
            U_s = np.sqrt(1 + 1 / n)

            if r_Rs < 1.0:
                sol = brentq(self.U_eqn, 1e-100, U_s, args=(n, r_Rs), xtol=1e-100, rtol=8.9e-16)
            elif r_Rs > 1.0:
                sol = brentq(self.U_eqn, U_s, 100, args=(n, r_Rs), xtol=1e-12, rtol=1e-15)
            else:
                sol = U_s

            Uout[i] = sol

        rho, P, cs = self.getflow_fromU(r_Rs_grid, Uout, n)
        return Uout, rho, P, cs
