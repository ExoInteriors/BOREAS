import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.tri as mtri
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors

from scipy.interpolate import griddata

from boreas import ModelParams
_M_EARTH = ModelParams().mearth
_R_EARTH = ModelParams().rearth
_mH      = ModelParams().m_H
_mO      = ModelParams().m_O
_kB      = ModelParams().k_b
_G       = ModelParams().G

class Plots:
    @staticmethod
    def _ensure_ok(df):
        """Return a copy filtered to rows that are not SKIPPED."""
        if "regime" not in df.columns:
            return df.copy()
        return df[df["regime"] != "SKIPPED"].copy()

    @staticmethod
    def regime_scatter(df):
        """
        Scatter plot of regime vs FXUV and planet mass.
        - X: FXUV (log)
        - Y: planet mass (Earth masses; linear)
        - Edge color encodes regime (EL=black, RL=red).
        """
        df = Plots._ensure_ok(df)
        if df.empty:
            print("[regime_scatter] No non-SKIPPED rows to plot.")
            return

        df = df[df['regime'].notna()].copy()
        df['mass_earth'] = df['m_planet'] / _M_EARTH
        edge = df['regime'].map({'EL':'black', 'RL':'red'}).fillna('gray')

        plt.figure(figsize=(4, 3.2))
        plt.scatter(df['FXUV'], df['mass_earth'], s=20, c='none', edgecolors=edge, linewidths=0.8)
        plt.xlabel("F$_{\\mathrm{XUV}}$ (erg cm$^{-2}$ s$^{-1}$)")
        plt.ylabel("Planet Mass (M$_\\oplus$)")
        plt.xscale('log')
        plt.yscale('linear')

        legend_elems = [
            mlines.Line2D([], [], marker='o', linestyle='None', markeredgecolor='black', markerfacecolor='none', markersize=6, label='EL'),
            mlines.Line2D([], [], marker='o', linestyle='None', markeredgecolor='red', markerfacecolor='none', markersize=6, label='RL')
        ]
        plt.legend(handles=legend_elems, loc='best', frameon=False, fontsize=9)
        plt.tight_layout()
        plt.show()



    @staticmethod
    def Mdot_FXUV(df, show_regime=True,
                  planet_types=('super-Earths (3% WMF)',
                                'sub-Neptunes (3% AMF, H/He)',
                                'sub-Neptunes (10% H2O)',
                                'sub-Neptunes (50% H2O)')):
        df = Plots._ensure_ok(df)
        df = df[df['planet_type'].isin(planet_types)].copy()
        if df.empty:
            print("[Mdot_FXUV] No rows for selected planet_types or all SKIPPED.")
            return

        unique_Teq = sorted(df['Teq'].unique())
        teq_palette = ['gold', 'darkorange', 'chocolate', 'maroon']
        teq_color = {t: teq_palette[i % len(teq_palette)] for i, t in enumerate(unique_Teq)}

        if show_regime:
            edge_map = {'EL': 'black', 'RL': 'red'}
        else:
            edge_map = {r: 'gray' for r in df['regime'].unique()}
            
        def subsample_by_flux_and_mass(group, desired=6):
            g = group.sort_values('m_planet')
            n = len(g)
            if n <= desired:
                return g
            idx_min = 0; idx_max = n - 1
            n_middle = desired - 2
            if n_middle > 0:
                middle_idxs = np.linspace(1, n - 2, n_middle, dtype=int)
                picks = [idx_min] + middle_idxs.tolist() + [idx_max]
            else:
                picks = [idx_min, idx_max]
            return g.iloc[picks]

        fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
        axes = axes.flatten()

        for ax, ptype in zip(axes, planet_types):
            sub = df[df['planet_type'] == ptype]
            if sub.empty:
                ax.text(0.5, 0.5, f"No data for\n{ptype}", ha='center', va='center')
                continue

            # Subsample within each FXUV to avoid overplotting
            sub_s = sub.groupby('FXUV', group_keys=False).apply(subsample_by_flux_and_mass)

            local_min = sub['m_planet'].min()
            local_max = sub['m_planet'].max()
            scale = max(local_max - local_min, 1.0)

            for _, row in sub_s.iterrows():
                ax.scatter(
                    row['FXUV'], row['Mdot'],
                    s = max(30.0, 200.0 * (row['m_planet'] - local_min) / scale),
                    marker = 'o',
                    facecolor = teq_color.get(row['Teq'], 'gray'),
                    edgecolor = edge_map.get(row.get('regime','EL'), 'gray'),
                    linewidths = 0.8,
                    alpha = 0.9
                )

            # Annotate min/max mass at a mid FXUV, if possible
            for Teq_value in sorted(sub['Teq'].unique()):
                feuv_list = sorted(sub.loc[sub['Teq'] == Teq_value, 'FXUV'].unique())
                if len(feuv_list) < 3:
                    continue
                third = feuv_list[2]
                slice_df = sub[(sub['Teq'] == Teq_value) & (sub['FXUV'] == third)]
                if slice_df.empty: 
                    continue
                row_min = slice_df.loc[slice_df['m_planet'].idxmin()]
                row_max = slice_df.loc[slice_df['m_planet'].idxmax()]
                ax.text(row_min['FXUV'] * 0.9, row_min['Mdot'] * 1.1, f"{row_min['m_planet']/_M_EARTH:.1f} M$_\\oplus$", fontsize=10)
                ax.text(row_max['FXUV'] * 1.1, row_max['Mdot'] * 0.9, f"{row_max['m_planet']/_M_EARTH:.1f} M$_\\oplus$", fontsize=10)

            ax.text(0.02, 0.98, ptype, transform=ax.transAxes, va='top', ha='left', fontsize=13)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.tick_params(axis='both', which='major', labelsize=11)

        fig.text(0.5, 0.02, "F$_{\\mathrm{XUV}}$ (erg cm$^{-2}$ s$^{-1}$)", ha='center', fontsize=14)
        fig.text(0.02, 0.5, "Mass-loss rate (g s$^{-1}$)", va='center', rotation='vertical', fontsize=14)

        # Legend for Teq and regime
        handles = [mpatches.Patch(color=teq_color[t], label=f"{int(t)} K") for t in unique_Teq[:8]]
        if show_regime:
            for r, ec in {'EL':'black', 'RL':'red'}.items():
                handles.append(mlines.Line2D([], [], marker='o', linestyle='None',
                                             markerfacecolor='none', markeredgecolor=ec, markersize=7,
                                             label=f"{r} regime"))
        fig.legend(handles=handles, loc='lower center', ncol=min(4, len(handles)), fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.03))
        plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.10, hspace=0.08, wspace=0.08)
        plt.show()
        
        
    
    @staticmethod
    def radius_FXUV_oxygen_contour(df_combined):
        '''
        Radius vs FXUV (2 panels) with log10(x_O) colormap, clamped below -10.
        Adapts columns to the new schema (uses FXUV, not FEUV). Skips SKIPPED rows.
        '''
        # Filter + required columns
        req = ['FXUV','r_planet','x_O','regime','planet_type','Teq']
        df = df_combined.dropna(subset=[c for c in req if c in df_combined.columns]).copy()
        if 'regime' in df.columns:
            df = df[df['regime'] != 'SKIPPED'].copy()
        if df.empty:
            print('[radius_FXUV_oxygen_contour] No plottable rows.')
            return

        # Units + markers
        df['R_earth'] = df['r_planet'] / _R_EARTH
        df['marker']  = df['Teq'].apply(lambda T: 'o' if T <= 500 else 's')
        df['ecolor']  = df['regime'].map({'EL':'k','RL':'red'}).fillna('k')

        # Color scaling for log10(x_O)
        clamp_log = -4.0
        xmin = df['FXUV'].min()
        xmax = df['FXUV'].max()
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin <= 0:
            print('[radius_FXUV_oxygen_contour] Non-positive or invalid FXUV range.')
            return

        # Grid
        xi = np.logspace(np.log10(xmin), np.log10(xmax), 220)
        yi = np.linspace(df['R_earth'].min(), df['R_earth'].max(), 220)
        X, Y = np.meshgrid(xi, yi)

        # Panels: Super-Earths vs Sub-Neptunes
        panels = [
            ("Super-Earths (3% WMF)", df['planet_type'].str.contains('super-Earths', na=False)),
            ("Sub-Neptunes (3% AMF, 10-90% WMF)", df['planet_type'].str.contains('sub-Neptunes', na=False)),
        ]

        # Colormap
        cmap = plt.get_cmap('coolwarm')
        cmap.set_under(cmap(0))
        norm = mcolors.Normalize(vmin=clamp_log, vmax=0.0)
        levels = np.linspace(clamp_log, 0.0, 11)

        fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)

        cf = None
        for ax, (title, mask) in zip(axes, panels):
            sub = df[mask].copy()
            ax.set_title(title, fontsize=15)

            if sub.empty:
                ax.axis('off')
                continue

            # Interpolate raw x_O, then log10 with clamp to avoid -inf
            Z_raw = griddata((sub['FXUV'].values, sub['R_earth'].values),
                             sub['x_O'].values, (X, Y), method='linear')
            Z_raw = np.ma.masked_invalid(Z_raw)
            with np.errstate(divide='ignore'):
                Z = np.log10(np.clip(Z_raw, 10.0**clamp_log, None))

            cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, extend='min', alpha=0.85)

            # Overlay scatter, encoded by log10(x_O)
            zvals = np.log10(np.clip(sub['x_O'].values, 10.0**clamp_log, None))
            for (regime, marker), grp in sub.groupby(['regime','marker']):
                gvals = np.log10(np.clip(grp['x_O'].values, 10.0**clamp_log, None))
                ax.scatter(grp['FXUV'], grp['R_earth'], c=gvals, cmap=cmap, norm=norm,
                           marker=marker, edgecolor=grp['ecolor'].iloc[0], linewidth=0.7, alpha=0.6, s=20)

            ax.set_xscale('log')
            ax.tick_params(axis='both', which='major', labelsize=13)

        fig.text(0.5, 0.04, "F$_{\\mathrm{XUV}}$ (erg cm$^{-2}$ s$^{-1}$)", ha='center', fontsize=16)
        fig.text(0.02, 0.5, "Planet radius (R$_\\oplus$)", va='center', rotation='vertical', fontsize=16)

        # Colorbar
        fig.subplots_adjust(left=0.12, right=0.86, bottom=0.12, top=0.94, wspace=0.08)
        if cf is not None:
            cax = fig.add_axes([0.88, 0.15, 0.02, 0.70])
            cbar = fig.colorbar(cf, cax=cax, extend='min')
            cbar.set_label('log$_{10}$ x$_{\\mathrm{O}}$', fontsize=13)
            ticks = np.linspace(clamp_log, 0.0, 5)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t:.0f}" for t in ticks])
            cbar.ax.tick_params(labelsize=12)

        # Legend for Teq marker shapes
        handles = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=7, label='T$_{eq} \\leq 500$ K'),
            Line2D([0],[0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=7, label='T$_{eq} > 500$ K'),
        ]
        fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=12)

        plt.show()

    @staticmethod
    def mass_FXUV_oxygen_contour(df_combined):
        req = ['FXUV','m_planet','x_O','regime','planet_type','Teq']
        df = df_combined.dropna(subset=[c for c in req if c in df_combined.columns]).copy()
        if 'regime' in df.columns:
            df = df[df['regime'] != 'SKIPPED'].copy()
        if df.empty:
            print('[mass_FXUV_oxygen_contour] No plottable rows.')
            return

        df['M_earth'] = df['m_planet'] / _M_EARTH
        df['marker']  = df['Teq'].apply(lambda T: 'o' if T <= 500 else 's')
        df['ecolor']  = df['regime'].map({'EL':'k','RL':'red'}).fillna('k')

        # clamp_log = -6.0
        xmin = df['FXUV'].min()
        xmax = df['FXUV'].max()
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin <= 0:
            print('[mass_FXUV_oxygen_contour] Non-positive or invalid FXUV range.')
            return

        xi = np.logspace(np.log10(xmin), np.log10(xmax), 220)
        yi = np.linspace(df['M_earth'].min(), df['M_earth'].max(), 220)
        X, Y = np.meshgrid(xi, yi)

        panels = [
            ("Super-Earths (3% WMF)", df['planet_type'].str.contains('super-Earths', na=False)),
            ("Sub-Neptunes (3% AMF, 10-90% WMF)", df['planet_type'].str.contains('sub-Neptunes', na=False)),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

        cf_for_cbar = None # keep a handle for colorbar from any panel that has positives

        cf = None
        for ax, (title, mask) in zip(axes, panels):
            sub = df[mask].copy()
            ax.set_title(title, fontsize=15)

            if sub.empty:
                ax.axis('off')
                continue

            Z_raw = griddata((sub['FXUV'].values, sub['M_earth'].values), sub['x_O'].values, (X, Y), method='linear')
            
            # Build mask for exactly-zero values (allow tiny numerical noise)
            zero_mask = np.isfinite(Z_raw) & (np.abs(Z_raw) == 0.0)

            # Positive values -> log10 scaling (no clipping/clamping)
            pos_mask = np.isfinite(Z_raw) & (Z_raw > 0.0)
            if np.any(pos_mask):
                Z_log = np.full_like(Z_raw, np.nan, dtype=float)
                Z_log[pos_mask] = np.log10(Z_raw[pos_mask])

                # 10 levels spanning actual data range
                vmin = np.nanmin(Z_log[pos_mask])
                vmax = np.nanmax(Z_log[pos_mask])
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    # fallback to a tiny span if degenerate
                    vmin, vmax = vmin - 1e-6, vmin + 1e-6
                levels = np.linspace(vmin, vmax, 10)

                cmap = plt.get_cmap('plasma')
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

                # Draw the positive field first (so black zeros can sit on top cleanly)
                cf = ax.contourf(X, Y, Z_log, levels=levels, cmap=cmap, norm=norm, alpha=0.9)
                cf_for_cbar = cf if cf_for_cbar is None else cf_for_cbar
            else:
                cf = None
                levels = None
                cmap = plt.get_cmap('plasma')
                norm = None

            # Overlay zeros in pure black, exactly where x_O == 0
            if np.any(zero_mask):
                # Use a binary contourf to paint zeros black
                ax.contourf(X, Y, zero_mask.astype(float), levels=[0.5, 1.5], colors='grey', alpha=0.2)

            # Scatter the original points:
            #  - black for x_O == 0
            #  - colormap for x_O > 0 (log10 scaled with the same norm)
            for (regime, marker), grp in sub.groupby(['regime','marker']):
                xo = grp['x_O'].values
                fx = grp['FXUV'].values
                me = grp['M_earth'].values

                # build RGBA colors per point
                facecolors = []
                if np.any(pos_mask) and norm is not None:
                    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                    for val in xo:
                        if val == 0.0:
                            facecolors.append(mcolors.to_rgba('grey'))
                        elif np.isfinite(val) and (val > 0.0):
                            facecolors.append(sm.to_rgba(np.log10(val)))
                        else:
                            # NaN/negative: make transparent (or choose a fallback)
                            facecolors.append((0, 0, 0, 0))
                else:
                    # no positive mapping available; paint zeros black, others transparent
                    for val in xo:
                        facecolors.append((0, 0, 0, 1) if val == 0.0 else (0, 0, 0, 0))

                ax.scatter(fx, me, c=facecolors, marker=marker, edgecolor=grp['ecolor'].iloc[0], linewidth=0.7, alpha=0.5, s=20)

            ax.set_xscale('log')
            ax.tick_params(axis='both', which='major', labelsize=13)

        fig.text(0.5, 0.01, "F$_{\\mathrm{XUV}}$ (erg cm$^{-2}$ s$^{-1}$)", ha='center', fontsize=15)
        fig.text(0.05, 0.5, "Planet mass (M$_\\oplus$)", va='center', rotation='vertical', fontsize=15)

        fig.subplots_adjust(left=0.12, right=0.86, bottom=0.12, top=0.94, wspace=0.08)

        # Colorbar for the positive field only (if any panel had positives)
        if cf_for_cbar is not None:
            cax = fig.add_axes([0.88, 0.15, 0.02, 0.70])
            cbar = fig.colorbar(cf_for_cbar, cax=cax)
            cbar.set_label('log$_{10}$ x$_{\\mathrm{O}}$ (x$_{\\mathrm{O}}$ > 0)', fontsize=15)
            # 10 ticks = the levels used
            cbar.set_ticks(cf_for_cbar.levels)
            cbar.set_ticklabels([f"{t:.2f}" for t in cf_for_cbar.levels])
            cbar.ax.tick_params(labelsize=15)
            
        plt.show()
        
        # separate figure for legend
        fig_leg = plt.figure(figsize=(8, 3))
        ax_leg = fig_leg.add_subplot(111)
        ax_leg.axis('off')  # hide axes
        
        handles = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor='white', markeredgecolor='k', markersize=7, label='T$_{eq} = 300, \, 400$ K'),
            Line2D([0],[0], marker='s', color='w', markerfacecolor='white', markeredgecolor='k', markersize=7, label='T$_{eq} = 1000, \,2000$ K'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='white', markeredgecolor='k', markersize=7, label='energy-limited regime'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='white', markeredgecolor='red', markersize=7, label='recombination-limited regime'),
            Line2D([0],[0], marker='s', color='w', markerfacecolor='grey',  markersize=7, label='x$_{\\mathrm{O}}=0$'),
        ]
        
        leg = fig_leg.legend(handles=handles, loc='center', frameon=False, fontsize=14, ncol=2)
        plt.show()


    @staticmethod
    def oxygen_indicators(df, flux_unit="number"):
        """
        flux_unit : {'number','mass'}
            - 'number' uses 'phi_O_num' [cm^-2 s^-1]
            - 'mass'   multiplies by m_O to show [g cm^-2 s^-1]
        """
        d = Plots._ensure_ok(df)
        
        types_with_water = [
            "super-Earths (3% WMF)",
            "sub-Neptunes (10% H2O)",
            "sub-Neptunes (20% H2O)",
            "sub-Neptunes (50% H2O)",
            "sub-Neptunes (70% H2O)",
            "sub-Neptunes (90% H2O)"
        ]
        
        d = d[d['planet_type'].isin(types_with_water)].copy()
        if d.empty:
            print("[oxygen_indicators_planets] No matching rows.")
            return

        # choose flux column
        if flux_unit == "number":
            flux_col = "phi_O_num"
            ylab = "$\phi_{\mathrm{O}}$ (cm$^{-2}$ s$^{-1}$)"
            if flux_col not in d.columns:
                raise KeyError("Expected 'phi_O_num' in dataframe for number flux plotting.")
            d["_phiO_plot"] = d[flux_col]
        elif flux_unit == "mass":
            # convert number flux to mass flux
            src = "phi_O_num"
            if src not in d.columns:
                raise KeyError("Expected 'phi_O_num' in dataframe to compute mass flux.")
            d["_phiO_plot"] = d[src] * _mO
            ylab = "$\phi_{\mathrm{O}}$ (g cm$^{-2}$ s$^{-1}$)"
        else:
            raise ValueError("flux_unit must be 'number' or 'mass'.")

        color_dict = {
            "super-Earths (3% WMF)": "darkorange",
            "sub-Neptunes (10% H2O)": "powderblue",
            "sub-Neptunes (20% H2O)": "lightskyblue",
            "sub-Neptunes (50% H2O)": "deepskyblue",
            "sub-Neptunes (70% H2O)": "royalblue",
            "sub-Neptunes (90% H2O)": "mediumblue"
        }

        plt.figure(figsize=(6, 4))
        for ptype in types_with_water:
            subset = d[d['planet_type'] == ptype]
            if subset.empty:
                continue

            # valid positive pairs for log-log fit
            valid = (subset['x_O'] > 0) & (subset['_phiO_plot'] > 0)
            slope_text = ""
            if valid.sum() > 1:
                x_vals = np.log10(subset.loc[valid, 'x_O'].values)
                y_vals = np.log10(subset.loc[valid, '_phiO_plot'].values)
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                slope_text = f", slope={slope:.2f}"
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_line = slope * x_line + intercept
                plt.plot(10**x_line, 10**y_line, linestyle='--',
                         color=color_dict[ptype], label='_nolegend_')

            plt.scatter(subset['x_O'], subset['_phiO_plot'],label=ptype + slope_text,
                        marker='o', color=color_dict[ptype], alpha=0.85, edgecolor='k', linewidths=0.6)

        plt.xscale('log'); plt.yscale('log')
        plt.xlabel("$x_{\mathrm{O}}$", fontsize=15)
        plt.ylabel(ylab, fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.legend(loc='best', fontsize=15, frameon=False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def atomic_escape(df):
        """
        number fluxes:
          - N_H/s = phi_H_num * 4π RXUV^2
          - N_O/s = phi_O_num * 4π RXUV^2
          - ratio = (N_O/s) / (N_H/s)
        Add water-mixing limiting lines by supplied WMFs.
        """
        d = Plots._ensure_ok(df)
        need = ['phi_H_num','phi_O_num','RXUV','planet_type','f_O','regime']
        for c in need:
            if c not in d.columns:
                raise KeyError(f"[atomic_escape] Missing '{c}'.")

        d = d.dropna(subset=['phi_H_num','phi_O_num','RXUV']).copy()
        area        = 4*np.pi*d['RXUV']**2
        d['_NH']    = d['phi_H_num'] * area
        d['_ratio'] = (d['phi_O_num'] / d['phi_H_num']).replace([np.inf,-np.inf], np.nan)

        # Planet types, water fractions & colors
        types = [
            ("super-Earths (3% WMF)"),
            ("sub-Neptunes (10% H2O)"),
            ("sub-Neptunes (20% H2O)"),
            ("sub-Neptunes (50% H2O)"),
            ("sub-Neptunes (70% H2O)"),
            ("sub-Neptunes (90% H2O)"),
        ]
        colors = {
            "super-Earths (3% WMF)": "darkorange",
            "sub-Neptunes (10% H2O)": "powderblue",
            "sub-Neptunes (20% H2O)": "lightskyblue",
            "sub-Neptunes (50% H2O)": "deepskyblue",
            "sub-Neptunes (70% H2O)": "royalblue",
            "sub-Neptunes (90% H2O)": "mediumblue",
        }

        plt.figure(figsize=(8, 9))
        for ptype in types:
            sub = d[d['planet_type'] == ptype]
            if sub.empty:
                continue

            # points (RL with red edge)
            edge = np.where(sub['regime'].values=='RL','red','black')
            plt.scatter(sub['_NH'], sub['_ratio'], c=colors[ptype], edgecolors=edge, linewidths=0.5, s=25, alpha=0.8, label=ptype)

            # hard ceiling from the model: median f_O for that class
            fO_med = np.nanmedian(sub['f_O'].values)
            plt.axhline(fO_med, color=colors[ptype], linestyle='--', label=f"limit ≈ median $f_O$ ({ptype})")

        plt.xscale('log'); plt.yscale('log')
        plt.xlabel("Hydrogen escape rate $\dot N_{\mathrm{H}}$ (atoms s$^{-1}$)", fontsize=16)
        plt.ylabel("Atomic ratio $\dot N_{\mathrm{O}}/\dot N_{\mathrm{H}}$", fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=13)
        plt.legend(fontsize=9, loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)
        plt.subplots_adjust(bottom=0.25)
        plt.show()


        
    @staticmethod
    def water_loss(df):
        d = Plots._ensure_ok(df)
        if 'planet_type' not in d.columns:
            raise KeyError("[water_loss] Missing 'planet_type'.")
        d = d[d['planet_type'].str.contains('super-Earths', na=False)].copy()
        need = ['phi_O_num','RXUV','Teq','m_planet','FXUV']
        for c in need:
            if c not in d.columns:
                raise KeyError(f"[water_loss] Missing required column '{c}'.")

        if d.empty:
            print("[water_loss] No super-Earth rows.")
            return

        sec_per_year = 365.25 * 24 * 3600.0
        sec_per_Myr  = 1e6 * sec_per_year
        ocean_mass   = 1.4e24  # g (1 Earth ocean)

        area                = 4.0 * np.pi * d['RXUV']**2
        d['phiO_gs']        = d['phi_O_num'] * _mO * area            # g s^-1 (global O escape)
        d['phiH2O_gs']      = d['phiO_gs'] * (18.0/16.0)             # g s^-1 water-equivalent (O-limited)
        d['EO_water_1Myr']  = d['phiH2O_gs'] * sec_per_Myr / ocean_mass
        d['EO_water_200Myr']= d['EO_water_1Myr'] * 200.0

        # initial bulk water mass: 3% of planet mass
        initial_water_mass  = 0.03 * d['m_planet']
        initial_water_EO    = initial_water_mass / ocean_mass
        d['water_depleted'] = d['EO_water_200Myr'] >= initial_water_EO

        # color by Teq
        teq = d['Teq']
        cmap = mcolors.LinearSegmentedColormap.from_list('teq_cmap', ['maroon', 'chocolate'])
        norm = mcolors.Normalize(vmin=np.nanmin(teq), vmax=np.nanmax(teq))
        teq_colors = cmap(norm(teq))

        masses = d['m_planet'] / _M_EARTH
        sizes  = np.sqrt(np.clip(masses, 1e-6, None)) * (25.0**1.2)

        fig, ax = plt.subplots(figsize=(8,6))

        ymax = d['EO_water_200Myr'].max() * 1e10
        ax.axhspan(1, ymax, color='#B3E5FC', alpha=0.3)
        
        # scatter
        for i, row in d.iterrows():
            ax.scatter(row['FXUV'], row['EO_water_200Myr'],
                       s=sizes[i], c=[teq_colors[i]],
                       edgecolors='red' if row.get('regime','EL')=='RL' else 'black',
                       marker='*' if row['water_depleted'] else 'o', alpha=0.85)

        # legends
        uniq_teq = sorted(d['Teq'].dropna().unique())
        teq_handles = [Line2D([0],[0], marker='o', color='w',
                              markerfacecolor=cmap(norm(t)), markersize=12,
                              label=f'{t:.0f} K') for t in uniq_teq]

        # mass handles
        min_mass = masses.min(); max_mass = masses.max()
        size_handles = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
                   markersize=max(6, np.sqrt(np.clip(min_mass,1e-6,None))* (25.0**1.2) ** 0.5),
                   label=f'{min_mass:.1f} M$_\oplus$'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
                   markersize=max(6, np.sqrt(np.clip(max_mass,1e-6,None))* (25.0**1.2) ** 0.5),
                   label=f'{max_mass:.1f} M$_\oplus$')
        ]
        depletion_handles = [Line2D([0],[0], marker='*', color='w', markerfacecolor='none',
                                    markeredgecolor='black', markersize=14, label='Water-depleted')]

        ax.legend(handles=teq_handles+depletion_handles+size_handles, loc='upper left', fontsize=11, frameon=False)

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('F$_{\mathrm{XUV}}$ (erg cm$^{-2}$ s$^{-1}$)', fontsize=16)
        ax.set_ylabel('Water loss per 200 Myr (Earth oceans)', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=13)

        def to_per_myr(y):    return y/200.0
        def to_per_200myr(y): return y*200.0
        ax2 = ax.secondary_yaxis('right', functions=(to_per_myr, to_per_200myr))
        ax2.set_yscale('log')
        ax2.set_ylabel('Water loss per 1 Myr (Earth oceans)', fontsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=12)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def metallicity_change(df, delta=0.02):
        d = Plots._ensure_ok(df)
        mask = (d['planet_type'].str.contains('sub-Neptunes', na=False) &
                d['planet_type'].str.contains('(10% H2O|20% H2O)', na=False))
        d = d[mask].copy()
        need = ['phi_O_num','phi_H_num','RXUV','m_planet','planet_type','FXUV','regime']
        for c in need:
            if c not in d.columns:
                raise KeyError(f"[metallicity_change] Missing required column '{c}'.")
        if d.empty:
            print("[metallicity_change] No rows for 10%/20% H2O sub-Neptunes.")
            return

        sec_per_year = 365.25 * 24 * 3600.0
        sec200       = 200e6 * sec_per_year
        area         = 4.0 * np.pi * d['RXUV']**2

        # Adopt atmosphere mass = 3% of planet mass
        d['M_atm']   = 0.03 * d['m_planet']

        # Initial metallicity from label
        d['Z_init']  = d['planet_type'].str.extract('(10|20)% H2O')[0].astype(float)/100.0

        # Initial inventories
        d['M_H2O_init'] = d['Z_init'] * d['M_atm']
        d['M_O_init']   = d['M_H2O_init'] * (16.0/18.0)
        d['M_H_init']   = d['M_atm'] - d['M_O_init']

        # Escaped masses over 200 Myr from number fluxes
        d['_MO_esc'] = d['phi_O_num'] * _mO * area * sec200
        d['_MH_esc'] = d['phi_H_num'] * _mH * area * sec200

        # Remainders (non-negative)
        d['M_O_rem'] = np.clip(d['M_O_init'] - d['_MO_esc'], 0.0, None)
        d['M_H_rem'] = np.clip(d['M_H_init'] - d['_MH_esc'], 0.0, None)

        # Max possible H2O from each
        d['M_H2O_from_O'] = d['M_O_rem'] * (18.0/16.0)
        d['M_H2O_from_H'] = d['M_H_rem'] * (18.0/2.0)

        # Actual water formed (limiting reagent)
        d['M_H2O_final'] = np.minimum(d['M_H2O_from_O'], d['M_H2O_from_H'])

        # Atoms consumed in final water
        d['used_O'] = d['M_H2O_final'] * (16.0/18.0)
        d['used_H'] = d['M_H2O_final'] * (2.0/18.0)

        # Leftovers
        d['leftover_O'] = d['M_O_rem'] - d['used_O']
        d['leftover_H'] = d['M_H_rem'] - d['used_H']

        # Final atmosphere
        d['M_atm_final'] = d['M_H2O_final'] + d['leftover_O'] + d['leftover_H']

        # Regimes
        d['water_regime'] = np.select(
            [
                d['leftover_H'] <= 0,                     # zero H remaining
                d['M_H2O_from_O'] <= d['M_H2O_from_H'],   # O-limited water
            ],
            [
                'Oxygen-dominated (pure O)',
                'Water-enriched (excess H)',
            ],
            default='H_limited'  # excess O after combining all H
        )

        # Final Z
        d['Z_final'] = np.select(
            [
                d['water_regime']=='Water-enriched (excess H)',
                d['water_regime']=='Oxygen-dominated (pure O)',
                d['water_regime']=='H_limited'  # excess O but some water formed
            ],
            [
                d['M_H2O_final'] / np.maximum(d['M_atm_final'], 1e-300),
                0.0,
                d['M_H2O_final'] / np.maximum(d['M_atm_final'], 1e-300),
            ]
        )

        # Keep significant changes
        d['dZ'] = d['Z_final'] - d['Z_init']
        keep = (np.abs(d['dZ']) > delta) | (d['water_regime']=='H_limited')
        d = d.loc[keep].copy()

        d['mass_em']  = np.round(d['m_planet']/_M_EARTH, 1)
        mass_vals     = np.sort(d['mass_em'].unique())
        n_mass        = len(mass_vals)
        if n_mass == 0:
            print("[metallicity_change] Nothing to plot after filtering.")
            return

        cmap   = plt.cm.get_cmap('inferno', n_mass)
        levels = np.arange(-1, n_mass, 1)
        norm   = mcolors.BoundaryNorm(levels, n_mass)

        regime_markers = {
            'Water-enriched (excess H)' : '^',    
            'H_limited'                 : 's',    
            'Oxygen-dominated (pure O)' : 'd',
        }

        fig, axes = plt.subplots(nrows=2, figsize=(6, 10), sharex=True, constrained_layout=True)
        fractions = ['10% H2O', '20% H2O']

        for ax, frac in zip(axes, fractions):
            sub = d[d['planet_type'].str.contains(frac, na=False)].copy()
            if sub.empty:
                ax.text(0.5, 0.5, f"No data for {frac}", ha='center', va='center')
                continue

            # mass index (for color)
            sub['mass_idx'] = sub['mass_em'].apply(lambda v: int(np.where(mass_vals==v)[0][0]))

            # Shade where Z_final >= Zmin (initial water fraction threshold)
            Zmin = float(frac.split('%')[0]) / 100.0
            mask = sub['Z_final'] >= Zmin
            if mask.any() and (len(sub[mask]) >= 3):
                tri = mtri.Triangulation(sub.loc[mask, 'Z_final'], sub.loc[mask, 'FXUV'])
                ax.tricontourf(tri, sub.loc[mask,'mass_idx'], levels=levels, cmap=cmap, norm=norm, alpha=0.25)

            # Scatter initial vs final along FXUV
            for _, row in sub.iterrows():
                col = cmap(row['mass_idx'])
                edge_col = {'EL':'k','RL':'r'}.get(row.get('regime','EL'), 'k')
                # initial (open circle) at Z_init
                ax.scatter(row['Z_init'], row['FXUV'], marker='o', s=55, facecolors='none', edgecolors='k', linewidths=1.0, alpha=0.9)
                # final (filled marker by regime)
                mk = regime_markers.get(row['water_regime'], '^')
                ax.scatter(row['Z_final'], row['FXUV'], marker=mk, s=75, facecolors=col, edgecolors=edge_col, linewidths=1.0, alpha=0.8)
                # arrow
                ax.annotate('', xy=(row['Z_final'], row['FXUV']), xytext=(row['Z_init'], row['FXUV']),
                            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.6), annotation_clip=False)

            ax.set_yscale('log')
            ax.set_ylim(d['FXUV'].min()*0.8, d['FXUV'].max()*1.2)
            ax.set_xlim(-0.05, 0.95)
            ax.tick_params(labelsize=12)
            ax.set_title(f"Initial {frac} sub-Neptunes", fontsize=13)

        # Legends
        present_regimes = d['water_regime'].unique()
        regime_handles = [Line2D([0],[0], marker=regime_markers[name], color='k', linestyle='None',
                                 markerfacecolor='w', markersize=9, label=name)
                          for name in present_regimes if name in regime_markers]

        mass_handles = [Line2D([0],[0], color=cmap(i), marker='^', linestyle='None',
                                markeredgecolor='k', markersize=9, label=f"{mv:.1f} M$_\oplus$")
                        for i, mv in enumerate(mass_vals)][::-1]

        fig.legend(handles=regime_handles, title='Regime', loc='upper left',
                   bbox_to_anchor=(1.02, 0.85), borderaxespad=0, fontsize=11, title_fontsize=12, frameon=False)
        fig.legend(handles=mass_handles, title='Planet mass', loc='upper left',
                   bbox_to_anchor=(1.02, 0.55), borderaxespad=0, fontsize=11, title_fontsize=12, frameon=False)

        axes[-1].set_xlabel('Metallicity $Z$ (M$_{\mathrm{H_2O}}$/M$_{\mathrm{atm}}$)', fontsize=14)
        fig.text(0.0, 0.5, 'F$_{\mathrm{XUV}}$ (erg cm$^{-2}$ s$^{-1}$)', va='center', rotation='vertical', fontsize=14)

        plt.show()
        
    # ---- diagnostics 
    
    @staticmethod
    def oxygen_fractionation_diagnostics(df, color_by="FXUV", size_by="m_planet", cmap="viridis", min_size=15, max_size=80, alpha=0.6):

        def b_HO(T):
            return 4.8e17 * (T ** 0.75)

        # base mask: keep finite numbers
        cols_needed = ['regime','phi_O_num','phi_H_num','f_O','x_O','T_outflow','m_planet','RXUV']
        miss = [c for c in cols_needed if c not in df.columns]
        if miss:
            raise KeyError(f"Missing required columns: {miss}")
        d = df.copy()

        # compute the two diagnostics
        with np.errstate(divide='ignore', invalid='ignore'):
            y_left = d['phi_O_num'] / (d['phi_H_num'] * d['f_O'])
            g      = (_G * d['m_planet']) / (d['RXUV'] ** 2)
            Fcrit  = g * (_mO - _mH) * b_HO(d['T_outflow']) / (_kB * d['T_outflow'] * (1.0 + d['f_O']))
            x_right= d['phi_H_num'] / Fcrit

        # finite/positive masks
        m_left  = np.isfinite(y_left) & (y_left > 0) & np.isfinite(d['x_O']) & (d['x_O'] > 0)
        is_H_major = (d.get('light_major_i') == 'H') if 'light_major_i' in d.columns else np.ones(len(d), dtype=bool)
        m_left  = m_left & is_H_major
        m_right = np.isfinite(x_right) & (x_right > 0) & np.isfinite(d['x_O']) & (d['x_O'] > 0)
        if not (m_left.any() and m_right.any()):
            raise ValueError("Not enough valid points to plot.")

        # Color map setup if column exists
        if color_by is not None and color_by in d.columns:
            cvals = d[color_by].astype(float)
            cmin, cmax = np.nanpercentile(cvals, [2, 98])
            if np.all((cvals[m_left|m_right]) > 0):
                norm = mcolors.LogNorm(vmin=max(np.nanmin(cvals[cvals>0]), 1e-300), vmax=max(cmax, 1e-299))
            else:
                norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
            pt_colors = mpl.cm.get_cmap(cmap)(norm(cvals))
        else:
            pt_colors = np.array([[0.3,0.5,0.9,1.0]] * len(d))
            norm = None

        # Size mapping
        if size_by is not None and size_by in d.columns:
            svals = d[size_by].astype(float)
            smin, smax = np.nanpercentile(svals, [2, 98])
            if smax <= smin:
                sizes = np.full(len(d), (min_size+max_size)/2.0)
            else:
                sizes = min_size + (np.clip(svals, smin, smax) - smin) / (smax - smin) * (max_size - min_size)
        else:
            sizes = np.full(len(d), (min_size+max_size)/2.0)

        edges = np.where(d['regime'].values=='RL', 'red', 'black')

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

        # left panel: identity check
        ax = axes[0]
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.scatter(d.loc[m_left,'x_O'], y_left[m_left], s=sizes[m_left], c=pt_colors[m_left], edgecolors=edges[m_left], linewidths=0.6, alpha=alpha)
        lo = 1e-3; hi = 1.1
        ax.plot([lo,1],[lo,1],'k--',lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(r'$x_{\mathrm{O}}$')
        ax.set_ylabel(r'$\phi_{\mathrm{O}} / (\phi_{\mathrm{H}} f_{\mathrm{O}})$')
        ax.set_title('Flux identity (H light-major)')

        # right panel: drag law
        ax = axes[1]
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.scatter(x_right[m_right], d.loc[m_right,'x_O'], s=sizes[m_right], c=pt_colors[m_right],
                edgecolors=edges[m_right], linewidths=0.6, alpha=alpha)
        rr = np.logspace(-2, 3, 300)
        ax.plot(rr, np.clip(1 - 1/rr, 1e-4, 1), 'k--', lw=1, label='trace-O theory')
        ax.set_xlabel(r'$\phi_{\mathrm{H}} / F_{\mathrm{crit}}$')
        ax.set_ylabel(r'$x_{\mathrm{O}}$')
        ax.set_ylim(1e-3, 1.2)
        ax.legend(loc='lower right', frameon=False)
        ax.set_title('Drag vs diffusion')

        if color_by is not None and color_by in d.columns and norm is not None:
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, shrink=0.9, pad=0.02)
            cbar.set_label(color_by)

        legend_handles = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor='none', markeredgecolor='black', label='EL', markersize=8),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='none', markeredgecolor='red',   label='RL', markersize=8),
        ]
        axes[0].legend(handles=legend_handles, loc='lower right', frameon=False)

        n0_left  = np.sum((d['x_O'] == 0) | (d['phi_O_num'] == 0))
        n0_right = np.sum((d['x_O'] == 0))
        print(f"[diag] dropped zeros (left,right) = ({n0_left},{n0_right})")

        return fig, axes
    
    @staticmethod
    def phiH_Fcrit(df):
        def b_HO(T): 
            return 4.8e17 * (T**0.75)

        d = df.copy()
        m  = (d['regime']!='SKIPPED') & np.isfinite(d['phi_H_num']) & (d['phi_H_num']>0) \
            & np.isfinite(d['T_outflow']) & (d['T_outflow']>0)
            
        g      = (_G*d.loc[m,'m_planet'])/(d.loc[m,'RXUV']**2)
        Fcrit  = g*(_mO-_mH)*b_HO(d.loc[m,'T_outflow'])/(_kB*d.loc[m,'T_outflow']*(1+d.loc[m,'f_O']))
        ratio  = d.loc[m,'phi_H_num']/Fcrit

        plt.figure(figsize=(5,3.5))
        plt.hist(np.log10(ratio), bins=40, color='gray')
        plt.axvline(0, c='r', ls='--', lw=1)
        plt.xlabel(r'$\log_{10}(\phi_{\rm H}/F_{\rm crit})$')
        plt.ylabel('count')
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def phiH_Fcrit2(df):
        bins = np.linspace(-3,3,25)
        d = df.query("regime!='SKIPPED'").copy()
        
        def b_HO(T): return 4.8e17 * (T**0.75)

        d = df.copy()
        m  = (d['regime']!='SKIPPED') & np.isfinite(d['phi_H_num']) & (d['phi_H_num']>0) \
            & np.isfinite(d['T_outflow']) & (d['T_outflow']>0)
            
        g      = (_G*d.loc[m,'m_planet'])/(d.loc[m,'RXUV']**2)
        Fcrit  = g*(_mO-_mH)*b_HO(d.loc[m,'T_outflow'])/(_kB*d.loc[m,'T_outflow']*(1+d.loc[m,'f_O']))
        ratio  = d.loc[m,'phi_H_num']/Fcrit

        d["lr"] = np.log10(ratio)
        m = d[np.isfinite(d.lr) & (d.lr>-5) & (d.lr<5) & (d.x_O>0)]
        grp = m.groupby(pd.cut(m.lr, bins))
        x_med = grp.x_O.median()
        lr_mid = 0.5*(bins[1:]+bins[:-1])
        plt.figure(figsize=(5,3.5))
        plt.semilogy(lr_mid, x_med, 'ko-')
        plt.axvline(0, ls='--', c='r')
        plt.xlabel(r'$\log_{10}(\phi_H/F_{\rm crit})$')
        plt.ylabel(r'median $x_O$')
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def g_Rp_RXUV(df):
        d = df.copy()
        g_Rp   = (_G*d.m_planet)/(d.r_planet**2)
        g_RXUV = (_G*d.m_planet)/(d.RXUV**2)
        plt.figure(figsize=(5,3.5))
        plt.loglog(g_Rp, g_RXUV, '.', alpha=0.3)
        plt.plot([g_Rp.min(), g_Rp.max()], [g_Rp.min(), g_Rp.max()],'k--')
        plt.xlabel(r'$g(R_p)$')
        plt.ylabel(r'$g(R_{\rm XUV})$')
        plt.tight_layout()
        
    @staticmethod
    def RXUV_over_Rp_scatter(df):
        d = df.copy()
        d = d[d['regime']!='SKIPPED'].dropna(subset=['RXUV','r_planet'])

        # absolute radii (for the x-axis) and the ratio (for the y-axis)
        Rp_abs   = d['r_planet'] / _R_EARTH          # R_p in R_earth
        RXUV_abs = d['RXUV']     / _R_EARTH          # R_XUV in R_earth
        ratio    = RXUV_abs / Rp_abs                 # dimensionless R_XUV/R_p

        # colors/edges by regime
        edge = np.where(d['regime'].values=='RL', 'r', 'k')

        plt.figure(figsize=(6,4))
        plt.scatter(Rp_abs, ratio, s=20, c='tab:blue', edgecolors=edge, linewidths=0.6, alpha=0.6)
        plt.axhline(1.0, ls='--', c='gray', lw=1)
        plt.xlabel(r'$R_p\;(\mathrm{R}_\oplus)$')
        plt.ylabel(r'$R_{\rm XUV}/R_p$')
        plt.tight_layout()
        plt.show()


    @staticmethod
    def RXUV_over_Rp_hist(df, xmax_zoom=2.5):
        d = df.copy()
        d = d[d['regime']!='SKIPPED'].dropna(subset=['RXUV','r_planet'])
        ratio = (d['RXUV'] / d['r_planet']).clip(lower=1.0)  # guard tiny numerical <1

        # quick textual summary (useful sanity check)
        q = np.quantile(ratio, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        print("[R_XUV/R_p] percentiles (1,5,25,50,75,95,99%):", np.round(q, 3))
        print("min/max:", float(ratio.min()), float(ratio.max()))

        fig, axes = plt.subplots(1, 2, figsize=(10,4), gridspec_kw={'wspace':0.25})

        # Zoomed panel (linear)
        r_zoom = ratio[(ratio >= 0.98) & (ratio <= xmax_zoom)]
        bins_zoom = np.linspace(0.98, xmax_zoom, 40)
        axes[0].hist(r_zoom, bins=bins_zoom, color='0.6')
        axes[0].axvline(1.0, ls='--', c='k', lw=1)
        axes[0].set_xlabel(r'$R_{\rm XUV}/R_p$ (zoom)')
        axes[0].set_ylabel('count')

        # Tail panel (log x)
        r_full = ratio[ratio > 0.98]
        rmin = max(0.98, r_full.min())
        rmax = r_full.max()
        # keep bins sensible even if rmax ~ 1
        if rmax/rmin < 1.05:
            rmax = rmin*1.05
        bins_full = np.logspace(np.log10(rmin), np.log10(rmax), 50)
        axes[1].hist(r_full, bins=bins_full, color='0.6')
        axes[1].set_xscale('log')
        axes[1].axvline(1.0, ls='--', c='k', lw=1)
        axes[1].set_xlabel(r'$R_{\rm XUV}/R_p$ (full, log)')
        axes[1].set_ylabel('count')

        plt.tight_layout()
        plt.show()
