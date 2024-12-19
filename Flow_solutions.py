import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw

## This routine contains various evaluation functions to find isothermal and polytropic flow solutions


#### ----- ISOTHERMAL ROUTINES

## critical flows
## these functions are following equation B2 from the appendix B in Owen & Schlichting 2023

def get_parker_wind(r,cs,Rs):
    
    # this function calculates the velocity structure (in cgs) of the isothermal Parker wind solution
    # c.f. Cranmer 2004
    
    f_r = 3. - 4.*Rs/r
    
    brackets = - (Rs/r)**4.*np.exp(f_r)
    
    u = np.zeros(np.size(r))
    
    u[r<=Rs] = np.sqrt(-np.real(lambertw(brackets[r<=Rs],0)))*cs # this means r/Rs<1, so the wind is subsonic. for r/Rs=1, we are at the sonic radius/we have sonic wind
    u[r>Rs]  = np.sqrt(-np.real(lambertw(brackets[r>Rs],-1)))*cs # this means r/Rs>1, so the wind is supersonic
    
    return u

def get_parker_wind_const(r,cs,Rs,const):
    
    # this function calculates the velocity structure (in cgs) of the isothermal Parker wind solution
    # c.f. Cranmer 2004
    
    f_r = -const - 4.*Rs/r
    
    brackets = - (Rs/r)**4.*np.exp(f_r)
    
    u = np.zeros(np.size(r))
    
    u[r<=Rs] = np.sqrt(-np.real(lambertw(brackets[r<=Rs],0)))*cs
    u[r>Rs]  = np.sqrt(-np.real(lambertw(brackets[r>Rs],-1)))*cs
    
    return u
    
def get_parker_wind_single(r,cs,Rs):
    
    # as above but for single value of r
    
    # this function calculates the velocity structure (in cgs) of the isothermal Parker wind solution
    # c.f. Cranmer 2004
    
    f_r = 3. - 4.*Rs/r
    
    brackets = - (Rs/r)**4.*np.exp(f_r)
    
    if (r <=Rs):
        u = np.sqrt(-np.real(lambertw(brackets,0)))*cs
    else:
        u = np.sqrt(-np.real(lambertw(brackets,-1)))*cs
    
    return u

## non-critical flows where c chooses the solution
def get_parker_wind_single_const(r,cs,Rs,c):
    
    # as above but for single value of r
    
    # this function calculates the velocity structure (in cgs) of the isothermal Parker wind solution
    # c.f. Cranmer 2004
    
    f_r = -c - 4.*Rs/r
    
    brackets = - (Rs/r)**4.*np.exp(f_r)
    
    if (r <=Rs):
        u= np.sqrt(-np.real(lambertw(brackets,0)))*cs
    else:
        u  = np.sqrt(-np.real(lambertw(brackets,-1)))*cs
    
    return u


#### ----- POLYTROPIC ROUTINES

def get_rmin(n):
    # calculates the minimum radius where u--> 0 for a polytropic wind with n<-1 
    
    r_Rs = 4* (1+1./n) / (1-2*n+3/n)
    
    return r_Rs

def U_eqn(U,n,r_Rs):
    
    LHS = 0.5*U**2.+(1+1/n)**(1./(2.*n))*(n+1)*U**(-1./n)*(1./r_Rs)**(2./n)-2*(1.+1/n)*(1./r_Rs)
    RHS = 0.5*(2.*n-3./n-1)
    
    return LHS-RHS

def getflow_fromU(r_Rs,U,n):
    
    # computes the flow variables from U as a function of r_Rs
    
    # get rho from mass conservation =1 at sonic point in dimensionless units
    
    rho = np.sqrt(1.+1./n)/(U*r_Rs**2.)
    
    P = rho**(1.+1./n)
    
    cs = np.sqrt(P/rho)
    
    return rho, P, cs

def get_dimensionless_flow_sol(r_Rs_grid,n):
    
    Uout = np.zeros(np.size(r_Rs_grid))

    for i in range (np.size(r_Rs_grid)):
        U_s = np.sqrt(1.+1./n)
    
        if (r_Rs_grid[i] < 1.):
            sol = brentq(U_eqn,1e-100,U_s,args=(n,r_Rs_grid[i]),xtol=1e-100, rtol=8.9e-16)
            Uout[i] = sol
        
        elif (r_Rs_grid[i] > 1.):
            sol = brentq(U_eqn,U_s,100,args=(n,r_Rs_grid[i]),xtol=1e-12, rtol=1e-15)
            Uout[i] = sol
        
        elif (r_Rs_grid[i]==1):
            Uout[i] = U_s
        
    
    rho, P, cs = getflow_fromU(r_Rs_grid,Uout,n)
    
    return Uout, rho, P, cs