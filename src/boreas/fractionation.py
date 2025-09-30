import numpy as np

class FractionationPhysics:
    def __init__(self, params):
        self.p = params

    def T_outflow_from_cs(self, cs, mu_outflow):
        return (cs**2) * self.p.m_H * mu_outflow / self.p.k_b

    def mu_eff_from_fluxes(self, phi_H_num, phi_O_num, phi_C_num=0.0, phi_N_num=0.0, phi_S_num=0.0):
        total = max(phi_H_num + phi_O_num + phi_C_num + phi_N_num + phi_S_num, 1e-300)
        return (self.p.am_h*phi_H_num + self.p.am_o*phi_O_num
                + self.p.am_c*phi_C_num + self.p.am_n*phi_N_num
                + self.p.am_s*phi_S_num) / total
    
    @staticmethod
    def atomic_counts_from_X(p):
        """Return per-species atomic counts per bulk mass at the XUV base (fully dissociated)."""
        X = p.get_X_tuple()
        # reuse N_* code
        (X_H2, X_H2O, X_O2, X_CO2, X_CO, X_CH4, X_N2, X_NH3, X_H2S, X_SO2, X_S2) = X

        N_H2  = X_H2  / p.mmw_H2_outflow  if X_H2  > 0 else 0.0
        N_H2O = X_H2O / p.mmw_H2O_outflow if X_H2O > 0 else 0.0
        N_O2  = X_O2  / p.mmw_O2_outflow  if X_O2  > 0 else 0.0
        N_CO2 = X_CO2 / p.mmw_CO2_outflow if X_CO2 > 0 else 0.0
        N_CO  = X_CO  / p.mmw_CO_outflow  if X_CO  > 0 else 0.0
        N_CH4 = X_CH4 / p.mmw_CH4_outflow if X_CH4 > 0 else 0.0
        N_N2  = X_N2  / p.mmw_N2_outflow  if X_N2  > 0 else 0.0
        N_NH3 = X_NH3 / p.mmw_NH3_outflow if X_NH3 > 0 else 0.0
        N_H2S = X_H2S / p.mmw_H2S_outflow if X_H2S > 0 else 0.0
        N_SO2 = X_SO2 / p.mmw_SO2_outflow if X_SO2 > 0 else 0.0
        N_S2  = X_S2  / p.mmw_S2_outflow  if X_S2  > 0 else 0.0

        N = dict(H = 2*N_H2 + 2*N_H2O + 4*N_CH4 + 3*N_NH3 + 2*N_H2S,
                O = 1*N_H2O + 2*N_O2  + 2*N_CO2 + 1*N_CO  + 2*N_SO2,
                C = 1*N_CO2 + 1*N_CO  + 1*N_CH4,
                N = 2*N_N2  + 1*N_NH3,
                S = 1*N_H2S + 1*N_SO2 + 2*N_S2)
        return N

    @staticmethod
    def choose_light_and_heavy_major(p, RXUV, T_outflow, m_planet, allow_dynamic_light_major=True, forced_light_major='H', eps=1e-20):
        """Pick (i, j) given current base composition and outflow T."""
        # atomic inventories (fully dissociated, per bulk mass)
        N = FractionationPhysics.atomic_counts_from_X(p)

        # ---- masses ----
        # Use GRAMS for all physics below (F_crit etc.)
        mass_g = {"H": p.m_H, "C": p.m_C, "N": p.m_N, "O": p.m_O, "S": p.m_S}
        
        # ---- pick light major i ----
        if allow_dynamic_light_major:
            # lightest species by mass, with non-negligible abundance (use mass_order or mass_g; both give same ordering)
            candidates = [s for s in ('H','C','N','O','S') if N[s] > eps]
            if not candidates:
                raise ValueError("No atomic species present at the XUV base.")
            i = min(candidates, key=lambda s: mass_g[s])
        else:
            i = forced_light_major.upper()
            if N.get(i, 0.0) <= eps:
                raise ValueError(f"Forced light major {i} absent at base (N_{i}≈0).")

        # ---- base mixing ratios relative to i ---- 
        # build mixing ratios f_s = N_s / N_i
        f = {s: (N[s]/max(N[i], eps)) for s in N.keys()}

        # ---- choose heavy major j among heavier species present ----
        # among heavier species (m_j > m_i) with f_j>0, choose j maximizing F_i,crit^(j)
        g = p.G * m_planet / (RXUV**2)
        best = None
        for s in ('H','C','N','O','S'):
            if s == i:
                continue
            if mass_g[s] <= mass_g[i]: # compare in grams
                continue
            if f[s] <= eps:
                continue
            # need b_{i,s}(T)
            b_ij = p.b_pair(i, s, T_outflow)
            # IMPORTANT: use GRAM masses in Fcrit, not amu
            Fcrit = g * (mass_g[s] - mass_g[i]) * b_ij / (p.k_b * T_outflow * (1.0 + f[s]))

            if (best is None) or (Fcrit > best[2]):
                best = (s, b_ij, Fcrit)
                
        if best is None:
            # no heavier species with significant f: no heavy major
            j = None
        else:
            j = best[0]
        return i, j, f

class GeneralizedFractionation:
    """
    Odert2018-style fractionation generalized to a dynamic light major i and heavy major j.
    Species considered: H, C, N, O, S (extendable).
    All phi_* returned are NUMBER fluxes.
    """
    def __init__(self, params):
        self.p  = params
        self.G  = params.G
        self.kB = params.k_b
        self.reg = params.species_registry()

    @staticmethod
    def _clamp01(x):
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))

    def compute_fluxes(self, flux_total_mass, RXUV, T_outflow, m_planet,
                       allow_dynamic_light_major=True, forced_light_major='H',
                       tol=1e-6, max_iter=100, eps=1e-20):
        """
        Returns:
          {
            'i': i, 'j': j,
            'phi': {'H':..., 'C':..., 'N':..., 'O':..., 'S':...},  # NUMBER fluxes
            'x':   {'C':..., 'N':..., 'O':..., 'S':...},           # entrainment fractions (x_i ≡ 1)
            'f':   {'H':..., 'C':..., 'N':..., 'O':..., 'S':...},  # base mixing ratios relative to i
            'mode': 'energy-limited' or 'diffusion-limited'
          }
        """
        p = self.p
        # choose i, j and compute f_s relative to i
        i, j, f = FractionationPhysics.choose_light_and_heavy_major(p, RXUV, T_outflow, m_planet, allow_dynamic_light_major, forced_light_major, eps)
        # present species
        species = ['H','C','N','O','S']
        # masses
        # m = {s: self.reg[s]['m'] for s in species} # amus!
        m = {"H": p.m_H, "C": p.m_C, "N": p.m_N, "O": p.m_O, "S": p.m_S} # grams!

        # initialize x_s (x_i ≡ 1 by definition)
        x = {s: 1.0 for s in species if s != i}

        # fixed-point loop on x's
        g = self.G * m_planet / (RXUV**2)

        for _ in range(max_iter):
            # effective grams per escaping i-particle in denominator
            denom_g_per_i = m[i] + sum(m[s]*f[s]*x.get(s,1.0) for s in species if s != i)
            denom_g_per_i = max(denom_g_per_i, 1e-300)
            Fi = flux_total_mass / denom_g_per_i # number flux of i

            # if we have a heavy major j, update x_j first (Eq. 4)
            if j is not None:
                b_ij = p.b_pair(i, j, T_outflow)
                xj_new = 1.0 - ( g * (m[j]-m[i]) * b_ij ) / ( max(Fi,1e-300) * self.kB * T_outflow * (1.0 + f[j]) )
                
                # if xj_new <= 0.0:
                #     # diffusion-limited branch against j
                #     Fcrit = g * (m[j]-m[i]) * b_ij / ( self.kB * T_outflow * (1.0 + f[j]) )
                #     phi = {s: 0.0 for s in species}
                #     phi[i] = Fcrit
                #     return {'i': i, 'j': j, 'phi': phi, 'x': {s:(0.0 if s!=i else 1.0) for s in species},
                #             'f': f, 'mode': 'diffusion-limited'}
                if xj_new <= 0.0:
                    # Heavy major j stalls. The actual i-flux is the MIN of EL supply and diffusion limit.
                    Fcrit  = g * (m[j]-m[i]) * b_ij / ( self.kB * T_outflow * (1.0 + f[j]) )  # cm^-2 s^-1
                    Fi_EL  = flux_total_mass / m[i]                                           # cm^-2 s^-1 (only i escapes)
                    phi_i  = Fcrit if (Fcrit < Fi_EL) else Fi_EL
                    mode   = 'diffusion-limited' if (phi_i is Fcrit) else 'energy-limited (j stalled)'

                    phi = {s: 0.0 for s in species}
                    phi[i] = phi_i
                    # x: j and all heavier minors are 0 when j stalls
                    x_out = {s: (1.0 if s == i else 0.0) for s in species}
                    return {'i': i, 'j': j, 'phi': phi, 'x': x_out, 'f': f, 'mode': mode}

                x[j] = self._clamp01(xj_new)

            # update minors (Eq. 5) for all s != i and s != j
            changed = False
            for k in species:
                if k == i or k == j:
                    continue
                if f[k] <= eps:
                    x[k] = 0.0
                    continue
                b_ik = p.b_pair(i, k, T_outflow)
                # terms in Eq. 5:
                base = 1.0 - g * (m[k]-m[i]) * b_ik / ( max(Fi,1e-300) * self.kB * T_outflow )
                if j is not None:
                    b_ij = p.b_pair(i, j, T_outflow)
                    b_jk = p.b_pair(j, k, T_outflow)
                    num = base + (b_ik/b_ij)*f[j]*(1.0 - x[j]) + (b_ik/b_jk)*f[j]*x[j]
                    den = 1.0 + (b_ik/b_jk)*f[j]
                else:
                    # no heavy major: Eq. 5 reduces to trace against i only
                    num = base
                    den = 1.0
                xk_new = self._clamp01( num / max(den, 1e-300) )
                changed |= (abs(xk_new - x.get(k,1.0)) > tol)
                x[k] = xk_new

            if not changed:
                break

        # final flux split (number flux)
        denom_g_per_i = m[i] + sum(m[s]*f[s]*x.get(s,1.0) for s in species if s != i)
        denom_g_per_i = max(denom_g_per_i, 1e-300)
        Fi = flux_total_mass / denom_g_per_i
        phi = {s: 0.0 for s in species}
        phi[i] = Fi
        for s in species:
            if s == i:
                continue
            phi[s] = Fi * f[s] * x.get(s, 1.0)

        # temporary: mass-flux self-consistency guard
        Fphi = sum(m[s] * phi.get(s, 0.0) for s in species) # g cm^-2 s^-1
        rel_err = abs(Fphi - flux_total_mass) / max(flux_total_mass, 1e-300)
        if rel_err > 1e-6:
            raise RuntimeError(f"Mass-flux mismatch in fractionation: rel_err={rel_err:.3e}")
        # hand back the exact mass flux that was used (for debugging/comparison)
        result = {'i': i, 'j': j, 'phi': phi, 'x': x, 'f': f, 'mode': 'energy-limited'}
        result['Fmass_in'] = float(flux_total_mass) # g cm^-2 s^-1
        
        return result

# -----------------------------------------
# Controller that runs the iteration for each solution
# -----------------------------------------
class Fractionation:
    """
    Four-species orchestrator:
      1) start from reservoir mu_outflow,
      2) run hydro (RXUV, c_s, Mdot),
      3) compute 5-species fluxes (H,O,C,N,S),
      4?) update mu_eff from escaping mixture, iterate to convergence?
    """
    def __init__(self, params):
        self.params  = params
        self.phys    = FractionationPhysics(params)
        self.general = GeneralizedFractionation(params)

    def execute(self, mass_loss_results, mass_loss, tol=1e-5, max_iter=100, allow_dynamic_light_major=True, forced_light_major='H', debug=False):
        
        out = []
        for sol in mass_loss_results:
            Mp, Rp, Teq = sol['m_planet'], sol['r_planet'], sol['Teq']
            mu_eff  = self.params.get_mu_outflow_current()
            mu_prev = None

            # keep the latest results so we can return even if we converge at it=0
            final_hydro = dict(sol)
            final_res   = None
            
            for _ in range(max_iter):
                # 1) make current μ available to hydro transport
                self.params.update_param('mmw_outflow_eff', mu_eff)
                
                # 2) run hydro-loss with RL disabled to get a consistent EL geometry for the current mu and chi_xuv
                probe = mass_loss.compute_mass_loss_parameters(np.array([Mp]), np.array([Rp]), np.array([Teq]), rl_policy='never')[0]
                RXUV_EL, cs_EL, Mdot_EL = probe['RXUV'], probe['cs'], probe['Mdot']
                t_ratio_EL = probe.get('time_scale_ratio', 10.0) # default >>1

                # fast selection of (i,j) without computing phi yet
                T_out_EL = self.phys.T_outflow_from_cs(cs_EL, mu_eff)
                i_guess, j_guess, _ = FractionationPhysics.choose_light_and_heavy_major(self.params, RXUV_EL, T_out_EL, Mp)

                # 3) if i==H and EL says RL regime, recompute hydro once in RL. Else keep EL.
                if (i_guess == 'H') and (t_ratio_EL < 1.0):
                    hydro = mass_loss.compute_mass_loss_parameters(np.array([Mp]), np.array([Rp]), np.array([Teq]), rl_policy='if_H', light_major='H')[0]
                else:
                    hydro = probe

                RXUV, cs, Mdot = hydro['RXUV'], hydro['cs'], hydro['Mdot']
    
                Fmass = Mdot / (4.0*np.pi*RXUV**2) # g cm^-2 s^-1
                
                # 3.5 RL/H temperature override just for fractionation physics
                # Find the current light-major i (independent of T, so a provisional T is fine):
                i_now, _, _ = FractionationPhysics.choose_light_and_heavy_major(self.params, RXUV, self.phys.T_outflow_from_cs(cs, mu_eff), Mp)

                if (hydro['regime'] == 'RL') or (i_now == 'H'):
                    # Use ~1e4 K for the outflow when H controls the thermodynamics (RL or i=H)
                    T_out = 1.0e4  # K
                else:
                    # EL case or non-H light-major: keep your usual mapping from cs & μ
                    T_out = self.phys.T_outflow_from_cs(cs, mu_eff)

                # 4) fractionation with this final geometry and T_out
                res = self.general.compute_fluxes(Fmass, RXUV, T_out, Mp, allow_dynamic_light_major=allow_dynamic_light_major, forced_light_major=forced_light_major)
 
                # 5) update mu from number fluxes
                phi = res['phi']
                mu_new = self.phys.mu_eff_from_fluxes(phi.get('H',0.0), phi.get('O',0.0), phi.get('C',0.0), phi.get('N',0.0), phi.get('S',0.0))

                # store latest in case we converge now
                final_hydro = hydro
                final_res   = res
                
                # 6) convergence on mu
                if (mu_prev is not None) and (abs(mu_new - mu_prev) <= tol*max(mu_prev, 1e-12)):
                    mu_eff = mu_new
                    break
                mu_prev = mu_eff = mu_new
            
            # ---- pack outputs for this planet ----
            RXUV, cs, Mdot = final_hydro['RXUV'], final_hydro['cs'], final_hydro['Mdot']
            T_out_final    = self.phys.T_outflow_from_cs(cs, mu_eff)

            # fallbacks if (pathologically) final_res is None
            if final_res is None:
                # Single-shot; run once to populate fields
                Fmass = Mdot / (4.0*np.pi*RXUV**2)
                final_res = self.general.compute_fluxes(
                    Fmass, RXUV, self.phys.T_outflow_from_cs(cs, mu_eff), Mp,
                    allow_dynamic_light_major=allow_dynamic_light_major,
                    forced_light_major=forced_light_major
                )

            phi = final_res['phi']
            x   = final_res['x']
            f   = final_res['f']

            # augment the hydro dict with fractionation results
            final_hydro.update({
                # number fluxes:
                'phi_H_num': phi.get('H',0.0), 'phi_O_num': phi.get('O',0.0),
                'phi_C_num': phi.get('C',0.0), 'phi_N_num': phi.get('N',0.0), 'phi_S_num': phi.get('S',0.0),
                # entrainment x and base ratios f:
                'x_O': x.get('O', 1.0), 'x_C': x.get('C', 1.0), 'x_N': x.get('N', 1.0), 'x_S': x.get('S', 1.0),
                'f_O': f['O'], 'f_C': f['C'], 'f_N': f['N'], 'f_S': f['S'],
                # who is i / j:
                'light_major_i': final_res['i'], 'heavy_major_j': final_res['j'],
                # thermodynamics:
                'mmw_outflow': mu_eff, 'T_outflow': T_out_final,
                # regimes:
                'fractionation_mode': final_res['mode'], # Odert branch (energy- vs diffusion-limited in the multi-species sense)
                # 'regime' stays whatever mass_loss reported (EL/RL)
            })

            out.append(final_hydro)

            # clear temporary μ so next planet starts clean
            self.params.update_param('mmw_outflow_eff', None)

        return out