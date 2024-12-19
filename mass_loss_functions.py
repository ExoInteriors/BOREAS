import numpy as np
import Flow_solutions as FS
import params as prm
from scipy.optimize import brentq

G = prm.G          # Gravitational constant, cm3 g-1 s-2
mmw_H = prm.mmw_H            # Mean molecular weight (H/He-ish envelope)
m_H = prm.m_H     # Mass of hydrogen atom, g

FEUV = prm.FEUV             # received EUV flux, ergs cm-2 s-1
sigma_EUV = prm.sigma_EUV    # EUV cross-section (of H? H2?), cm2
alpha_rec = prm.alpha_rec     # Recombination coefficient, cm3 s-1
eff = prm.eff               # Mass-loss efficiency factor

def compute_mdot_only(cs, REUV, m_planet):
    '''mass-loss rate Mdot directly'''
    RS_flow = G * m_planet / (2. * cs**2.)
    #print(f'--- Sonic radius for cs={cs:.2e}:', RS_flow)
    if RS_flow >= REUV:
        r = np.logspace(np.log10(REUV), np.log10(max(5 * RS_flow, 5 * REUV)), 250) # radius
        u = FS.get_parker_wind(r, cs, RS_flow) # outflow velocity
        rho = (RS_flow / r)**2 * (cs / u) # density such that density at sonic point is 1
        # now scale such that optical depth to EUV is 1
        tau = np.fabs(np.trapz(rho[::-1], r[::-1]))
        rho_s = 1. / ((sigma_EUV / (mmw_H * m_H / 2.)) * tau) # division by 2 suggests we are accounting for diatomic H2
        rho = rho * rho_s
        Mdot = 4 * np.pi * REUV**2 * rho[0] * u[0]
    else:
        r = np.logspace(np.log10(REUV), np.log10(max(5 * RS_flow, 5 * REUV)), 250)
        constant = (1. - 4. * np.log(REUV / RS_flow) - 4. * (RS_flow / REUV) - 1e-13)
        u = FS.get_parker_wind_const(r, cs, RS_flow, constant)
        rho = (RS_flow / r)**2 * (cs / u)
        tau = np.fabs(np.trapz(rho[::-1], r[::-1]))
        rho_s = 1. / ((sigma_EUV / (mmw_H * m_H / 2.)) * tau)
        rho = rho * rho_s
        Mdot = 4 * np.pi * REUV**2 * rho[0] * u[0]
    return Mdot


def compute_mdot(cs, REUV, m_planet, Mdot_want):
    '''used for root finding'''
    Mdot = compute_mdot_only(cs, REUV, m_planet)
    return (Mdot - Mdot_want) / (Mdot_want + 1.)


def compute_sound_speed(REUV, m_planet):
    '''calculate the energy-limited mass loss rate
    It's the theoretical upper limit on the mass loss rate based on the energy input and efficiency factor.'''
    Mdot_EL = eff * np.pi * REUV**3. / (4. * G * m_planet) * FEUV # EL = energy limited, after equation 17
    #print(f'- Mdot_EL for REUV={REUV:.2e}:', Mdot_EL)

    lower_bound_initial = 2e5 # Initial guess for the lower bound of sound speed
    upper_bound = 1e13

    def mdot_difference(cs):
        return compute_mdot(cs, REUV, m_planet, Mdot_EL)
   
    f1 = mdot_difference(lower_bound_initial)

    # Use root-finding to find the sound speed that matches Mdot_EL
    if f1 < 0:
        cs_use = brentq(mdot_difference, lower_bound_initial, upper_bound)
    else:
        # print ("reducing lower_bound")
        lower_bound = lower_bound_initial
        while f1 > 0:
            lower_bound = 0.95*lower_bound # adjust lower bound
            f1 = mdot_difference(lower_bound)
        cs_use = brentq(mdot_difference, lower_bound, upper_bound / 10)
    
    return cs_use

#
# momentum balance functions 
#

def compare_densities_EL(REUV, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon):
    '''momentum balance for EL case'''
    rho_EUV = rho_photo * np.exp((G * m_planet / cs_eq**2) * (1. / REUV - 1. / r_planet))
    cs_use = compute_sound_speed(REUV, m_planet)
    cs_use = min(cs_use, 1.2e6)  # Limit cs_use to T ~ 10^4 K
    Mdot = compute_mdot_only(cs_use, REUV, m_planet)
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
    return diff, time_scale_ratio

def find_REUV_solution_EL(r_planet, m_planet, rho_photo, cs_eq, FEUV_photon):
    '''REUV solution for EL case'''
    REUV_lower_bound = r_planet * 1.001
    REUV_upper_bound = r_planet * 6

    def root_function(REUV):
        diff, _ = compare_densities_EL(REUV, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon)
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
    _, time_scale_ratio = compare_densities_EL(REUV_solution, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon)
    return REUV_solution, time_scale_ratio

def compare_densities_RL(REUV, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon):
    '''momentum balance for RL case'''
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
    return diff

def find_REUV_solution_RL(r_planet, m_planet, rho_photo, cs_eq, FEUV_photon):
    '''REUV solution for RL case'''
    REUV_lower_bound = r_planet * 1.001
    REUV_upper_bound = r_planet * 6

    def root_function(REUV):
        return compare_densities_RL(REUV, rho_photo, r_planet, m_planet, cs_eq, FEUV_photon)
    
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
    REUV_solution_RL = brentq(root_function, REUV_lower_bound, REUV_upper_bound)
    return REUV_solution_RL