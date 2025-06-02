import numpy as np
import re
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from matplotlib.lines import Line2D
from matplotlib import colors
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap
from scipy.interpolate import griddata

from Parameters import ModelParams

class ResultsHandler:
        
    @staticmethod
    def regime_scatter(df_combined):
        """Scatter plot showing which regime each planet is in."""
        params = ModelParams()
        mearth = params.mearth

        df = df_combined[df_combined['regime'].notna()].copy()
        df['mass_earth'] = df['m_planet'] / mearth
        df['color'] = df['regime'].apply(lambda x: 'black' if x == 'EL' else 'red') # "RL" planets will be red, others ("EL")blue.

        plt.figure(figsize=(3, 3))
        plt.scatter(df['FEUV'], df['mass_earth'], c=df['color'], alpha=0.8)
        plt.xlabel("F$_{XUV}$ (erg/cm$^2$/s)")
        plt.ylabel("Planet Mass (M$_{\oplus}$)")
        plt.yscale('linear')
        plt.xscale('log')

        legend_elems = [
            Line2D([0], [0], marker='o', color='w', label='EL', markerfacecolor='black', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='RL', markerfacecolor='red', markersize=8)
        ]
        plt.legend(handles=legend_elems, loc='best')

        plt.tight_layout()
        plt.show()

    #
    ### ----------- Mass loss with FEUV -----------
    #
        
    @staticmethod
    def Mdot_FEUV(df, show_regime=True):
        # filter to desired planet types
        selected = [
            'super-Earths (3% WMF)', 
            'sub-Neptunes (3% AMF, H/He)',
            'sub-Neptunes (10% H2O)', 
            'sub-Neptunes (50% H2O)'
        ]
        df = df[df['planet_type'].isin(selected)].copy()

        # markers & colors
        marker = 'o'
        unique_Teq = sorted(df['Teq'].unique())
        teq_palette = ['gold', 'darkorange', 'chocolate', 'maroon']
        teq_color = {t: teq_palette[i % len(teq_palette)] for i, t in enumerate(unique_Teq)}
        edge_map = {'EL': 'black', 'RL': 'red'} if show_regime else {r: 'gray' for r in df['regime'].unique()}

        # subsampling helper
        def subsample_by_flux_and_mass(group, desired=5):
            g = group.sort_values('m_planet')
            if len(g) <= desired:
                return g
            idx = np.linspace(0, len(g)-1, desired, dtype=int)
            return g.iloc[idx]

        # build 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        axes = axes.flatten()

        max_m = df['m_planet'].max()
        for ax, ptype in zip(axes, selected):
            sub = df[df['planet_type'] == ptype]
            sub_s = sub.groupby('FEUV', group_keys=False).apply(subsample_by_flux_and_mass)

            for _, row in sub_s.iterrows():
                ax.scatter(
                    row['FEUV'], row['Mdot'],
                    s = row['m_planet']/max_m * 200,
                    marker = marker,
                    facecolor = teq_color[row['Teq']],
                    edgecolor = edge_map.get(row['regime'], 'gray'),
                    linewidths = 1,
                    alpha = 0.8
                )
            # place planet type label inside
            ax.text(0.02, 0.98, ptype, transform=ax.transAxes, va='top', ha='left', fontsize=12)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.tick_params(axis='both', which='major', labelsize=15)

        # common axes labels with spacing
        fig.text(0.5, 0.01, "F$_{XUV}$ (erg/cm$^2$/s)", ha='center', fontsize=15)
        fig.text(0.01, 0.5, "Mass loss rate (g/s)", va='center', rotation='vertical', fontsize=15)

        # legend handles (no marker shapes for planet types)
        handles = []
        # Teq color patches
        for t in unique_Teq:
            handles.append(mpatches.Patch(color=teq_color[t], label=f"{t} K"))
        # regime markers
        if show_regime:
            for r, ec in edge_map.items():
                handles.append(mlines.Line2D([], [], marker='o', linestyle='None', markerfacecolor='none', markeredgecolor=ec, markersize=8, label=f"{r} Regime"))

        # adjust margins to fit labels/legend
        plt.subplots_adjust(left=0.1, bottom=0.1, top=0.95, right=0.95, hspace=0.1, wspace=0.1)
        
        fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize='10', frameon=False, bbox_to_anchor=(0.5, -0.07))
        plt.show()

    #
    ### ----------- Oxygen escape contour maps -----------
    #

    @staticmethod
    def radius_FEUV_oxygen_contour_by_planet_type(df_combined):
        '''Radius vs FEUV plots for however many types of planets are available'''
        params = ModelParams()
        rearth = params.rearth

        # clean and normalize
        df = (df_combined.dropna(subset=['FEUV','r_planet','x_O','regime','planet_type','Teq']).copy())
        df['R_earth'] = df['r_planet'] / rearth

        # classify low/high Teq
        df['marker'] = df['Teq'].apply(lambda T: 'o' if T <= 400 else 's')
        df['ecolor'] = df['regime'].map({'EL':'k','RL':'red'})

        # grid for contours
        xi = np.logspace(np.log10(df['FEUV'].min()), np.log10(df['FEUV'].max()), 200)
        yi = np.linspace(df['R_earth'].min(), df['R_earth'].max(), 200)
        X, Y = np.meshgrid(xi, yi)

        # clamp threshold
        clamp = -10

        # prepare colormap: coolwarm, with the "under" color = the very bottom of the map
        cmap = plt.get_cmap('coolwarm')
        cmap.set_under(cmap(0))

        # norm over the full data range for ticks, etc.
        norm = colors.Normalize(vmin=clamp, vmax=0)
        # subplot grid
        types = df['planet_type'].unique()
        n = len(types)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, (ax, ptype) in enumerate(zip(axes, types)):
            sub = df[df['planet_type'] == ptype]
            if sub.empty:
                ax.axis('off')
                continue

            # interpolate raw x_O, then log10
            Z_raw = griddata((sub['FEUV'], sub['R_earth']), sub['x_O'], (X, Y), method='linear')
            Z_raw = np.ma.masked_invalid(Z_raw)
            Z = np.log10(Z_raw)

            # contour only between clamp→0
            levels = np.linspace(clamp, 0, 13)
            cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, extend='min', alpha=0.7)

            # scatter: log10 then clip at clamp
            for (_, marker), grp in sub.groupby(['regime','marker']):
                vals = np.log10(grp['x_O'].values)
                vals = np.clip(vals, clamp, None)
                ax.scatter(grp['FEUV'], grp['R_earth'], c=vals, cmap=cmap, norm=norm, marker=marker, edgecolor=grp['ecolor'].iloc[0], linewidth=0.8, alpha=0.9)

            ax.set_xscale('log')
            ax.set_title(ptype, fontsize=10)

            # only left‐col ylabels and bottom‐row xlabels
            row, col = divmod(idx, ncols)
            if col == 0:
                ax.set_ylabel("Planet Radius (R$_\oplus$)", fontsize=11)
            if row == nrows - 1:
                ax.set_xlabel("F$_{XUV}$ (erg/cm$^2$/s)", fontsize=11)

        # hide any extra axes
        for extra_ax in axes[n:]:
            extra_ax.axis('off')

        # reserve left 80% for subplots
        fig.subplots_adjust(right=0.80)

        # explicit colorbar axes in the remaining 20%
        cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(cf, cax=cbar_ax, extend='min')
        cbar.set_label('log$_{10}$ x$_O$', fontsize=11)

        # ticks from clamp→0, and label “under” as e.g. “< -20”
        ticks = np.linspace(clamp, 0, 5) # e.g. [-20, -15, -10, -5, 0]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.0f}" for t in ticks])

            # shared legend for Teq‐shape and regime‐edgecolor
        legend_handles = [
            Line2D([0],[0], marker='o', color='w', label='Low T$_{eq}$ (≤400 K)', markerfacecolor='gray', markeredgecolor='k', markersize=8),
            Line2D([0],[0], marker='s', color='w', label='High T$_{eq}$ (>400 K)', markerfacecolor='gray', markeredgecolor='k', markersize=8),
            Line2D([0],[0], marker='o', color='w', label='EL regime', markerfacecolor='white', markeredgecolor='k', markersize=8),
            Line2D([0],[0], marker='o', color='w', label='RL regime', markerfacecolor='white', markeredgecolor='red', markersize=8)
        ]
        fig.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=9)

        # tighten subplots within left 80%
        fig.tight_layout(rect=[0, 0.05, 0.80, 1.0])
        plt.show()

    @staticmethod
    def mass_FEUV_oxygen_contour_by_planet_type(df_combined):
        '''Mass vs FEUV plots with x_O clamped below -20 in colorbar.'''
        params = ModelParams()
        mearth = params.mearth

        # clean & normalize
        df = (df_combined.dropna(subset=['FEUV','r_planet','m_planet','x_O','regime','planet_type','Teq']).copy())
        df['M_earth'] = df['m_planet'] / mearth
        df['marker'] = df['Teq'].apply(lambda T: 'o' if T <= 500 else 's')
        df['ecolor'] = df['regime'].map({'EL':'k','RL':'red'})

        # clamp threshold in log10(x_O)
        clamp = -10
        cmap  = plt.get_cmap('coolwarm'); cmap.set_under(cmap(0))
        norm  = colors.Normalize(vmin=clamp, vmax=0)
        levels = np.linspace(clamp, 0, 13)

        # grid for contours
        xi = np.logspace(np.log10(df['FEUV'].min()), np.log10(df['FEUV'].max()), 200)
        yi = np.linspace(df['M_earth'].min(), df['M_earth'].max(), 200)
        X, Y = np.meshgrid(xi, yi)

        # set up subplots
        types = df['planet_type'].unique()
        n, ncols = len(types), 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows),
                                 sharex=True, sharey=True)
        axes = axes.flatten()

        for ax, ptype in zip(axes, types):
            sub = df[df['planet_type']==ptype]
            if sub.empty:
                ax.axis('off'); continue

            Z_raw = griddata((sub['FEUV'], sub['M_earth']), sub['x_O'], (X, Y), method='linear')
            Z_raw = np.ma.masked_invalid(Z_raw)
            Z = np.log10(Z_raw)

            # contour
            cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, extend='min', alpha=0.7)

            # scatter (clamp below clamp)
            for (_, marker), grp in sub.groupby(['regime','marker']):
                vals = np.log10(grp['x_O'].values)
                vals = np.clip(vals, clamp, None)
                ax.scatter(
                    grp['FEUV'], grp['M_earth'],
                    c=vals, cmap=cmap, norm=norm,
                    marker=marker, edgecolor=grp['ecolor'].iloc[0],
                    linewidth=0.8, alpha=0.9
                )

            ax.set_xscale('log')
            ax.set_title(ptype, fontsize=10)

        # hide extras, labels
        for extra in axes[len(types):]:
            extra.axis('off')

        fig.subplots_adjust(right=0.80)
        cax = fig.add_axes([0.82,0.15,0.02,0.7])
        cbar = fig.colorbar(cf, cax=cax, extend='min')
        cbar.set_label('log$_{10}$ x$_O$', fontsize=11)
        ticks = np.linspace(clamp, 0, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.0f}" for t in ticks])

        # legend
        handles = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label='Low T$_{eq}$ (≤500 K)'),
            Line2D([0],[0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label='High T$_{eq}$ (>500 K)'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='white', markeredgecolor='k', markersize=8, label='EL regime'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='white', markeredgecolor='red', markersize=8, label='RL regime'),
        ]
        fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=9)
        fig.tight_layout(rect=[0,0.05,0.80,1.0])
        plt.show()


    @staticmethod
    def radius_FEUV_oxygen_contour(df_combined):
        '''Radius vs FEUV 2x2 grid with x_O clamped below -10.'''
        params = ModelParams()
        rearth = params.rearth

        df = (df_combined.dropna(subset=['FEUV','r_planet','x_O','regime','planet_type','Teq']).copy())
        df['R_earth'] = df['r_planet'] / rearth
        df['marker'] = df['Teq'].apply(lambda T: 'o' if T <= 500 else 's')
        df['ecolor'] = df['regime'].map({'EL':'k','RL':'red'})

        clamp = -10
        cmap  = plt.get_cmap('coolwarm'); cmap.set_under(cmap(0))
        norm  = colors.Normalize(vmin=clamp, vmax=0)
        levels = np.linspace(clamp, 0, 13)

        xi = np.logspace(np.log10(df['FEUV'].min()), np.log10(df['FEUV'].max()), 200)
        yi = np.linspace(df['R_earth'].min(), df['R_earth'].max(), 200)
        X, Y = np.meshgrid(xi, yi)

        masks = [
            ("Super-Earths (3% WMF)", df['planet_type'].str.contains('super-Earths')),
            ("Sub-Neptunes (3% AMF, 10-90% WMF)", df['planet_type'].str.contains('sub-Neptunes'))
        ]
        panels = [
            (masks[0][0], masks[0][1], True),
            (masks[1][0], masks[1][1], True),
            (masks[0][0], masks[0][1], False),
            (masks[1][0], masks[1][1], False),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, (ax, (title, mask, do_scatter)) in enumerate(zip(axes, panels)):
            sub = df[mask]
            if sub.empty:
                if i < 2:
                    ax.set_title(title, fontsize=15)
                ax.axis('off')
                continue

            Z_raw = griddata((sub['FEUV'], sub['R_earth']), sub['x_O'], (X, Y), method='linear')
            Z_raw = np.ma.masked_invalid(Z_raw)
            Z = np.log10(Z_raw)

            cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, extend='min', alpha=0.8)

            if do_scatter:
                for (_, marker), grp in sub.groupby(['regime', 'marker']):
                    vals = np.log10(grp['x_O'].values)
                    vals = np.clip(vals, clamp, None)
                    ax.scatter(grp['FEUV'], grp['R_earth'], c=vals, cmap=cmap, norm=norm,
                            marker=marker, edgecolor=grp['ecolor'].iloc[0],
                            linewidth=0.8, alpha=0.5)

            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.set_xscale('log')
            if i < 2:
                ax.set_title(title, fontsize=15)

        fig.text(0.45, 0.02, "F$_{XUV}$ (erg/cm$^2$/s)", ha='center', fontsize=15)
        fig.text(0.0001, 0.5, "Planet radius (R$_\oplus$)", va='center', rotation='vertical', fontsize=15)

        fig.subplots_adjust(left=0.2, right=0.80)
        cax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(cf, cax=cax, extend='min')
        cbar.set_label('log$_{10}$ x$_O$', fontsize=15)
        ticks = np.linspace(clamp, 0, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.0f}" for t in ticks])
        cbar.ax.tick_params(labelsize=15)

        handles = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label='Low T$_{eq}$'),
            Line2D([0],[0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label='High T$_{eq}$'),
        ]

        fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=15)
        fig.tight_layout(rect=[0,0.05,0.80,1.0])
        plt.show()

    @staticmethod
    def mass_FEUV_oxygen_contour(df_combined):
        '''Mass vs FEUV 2x2 grid with x_O clamped below -10.'''
        params = ModelParams()
        mearth = params.mearth

        df = (df_combined.dropna(subset=['FEUV','m_planet','x_O','regime','planet_type','Teq']).copy())
        df['M_earth'] = df['m_planet'] / mearth
        df['marker']  = df['Teq'].apply(lambda T: 'o' if T <= 500 else 's')
        df['ecolor']  = df['regime'].map({'EL':'k','RL':'red'})

        clamp = -10
        cmap  = plt.get_cmap('coolwarm'); cmap.set_under(cmap(0))
        norm  = colors.Normalize(vmin=clamp, vmax=0)
        levels = np.linspace(clamp, 0, 13)

        xi = np.logspace(np.log10(df['FEUV'].min()), np.log10(df['FEUV'].max()), 200)
        yi = np.linspace(df['M_earth'].min(), df['M_earth'].max(), 200)
        X, Y = np.meshgrid(xi, yi)

        masks = [
            ("Super-Earths (3% WMF)", df['planet_type'].str.contains('super-Earths')),
            ("Sub-Neptunes (3% AMF, 10-90% WMF)", df['planet_type'].str.contains('sub-Neptunes'))
        ]
        panels = [
            (masks[0][0], masks[0][1], True),
            (masks[1][0], masks[1][1], True),
            (masks[0][0], masks[0][1], False),
            (masks[1][0], masks[1][1], False),
        ]

        fig, axes = plt.subplots(2,2, figsize=(10,8), sharex=True, sharey=True)
        axes = axes.flatten()


        for i, (ax, (title, mask, do_scatter)) in enumerate(zip(axes, panels)):
            sub = df[mask]
            if sub.empty:
                if i < 2:
                    ax.set_title(title, fontsize=15)
                ax.axis('off')
                continue

            Z_raw = griddata((sub['FEUV'], sub['M_earth']), sub['x_O'], (X, Y), method='linear')
            Z_raw = np.ma.masked_invalid(Z_raw)
            Z = np.log10(Z_raw)

            cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, extend='min', alpha=0.8)

            if do_scatter:
                for (_,marker), grp in sub.groupby(['regime','marker']):
                    vals = np.log10(grp['x_O'].values)
                    vals = np.clip(vals, clamp, None)
                    ax.scatter(grp['FEUV'], grp['M_earth'], c=vals, cmap=cmap, norm=norm, marker=marker, edgecolor=grp['ecolor'].iloc[0], linewidth=0.8, alpha=0.5)
            
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.set_xscale('log')
            if i < 2:
                ax.set_title(title, fontsize=15)

        fig.text(0.45, 0.02, "F$_{XUV}$ (erg/cm$^2$/s)", ha='center', fontsize=15)
        fig.text(0.0001, 0.5, "Planet mass (M$_\oplus$)", va='center', rotation='vertical', fontsize=15)
                
        fig.subplots_adjust(left=0.2, right=0.80)
        cax = fig.add_axes([0.82,0.15,0.02,0.7])
        cbar = fig.colorbar(cf, cax=cax, extend='min')
        cbar.set_label('log$_{10}$ x$_O$', fontsize=15)
        ticks = np.linspace(clamp, 0, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.0f}" for t in ticks])
        cbar.ax.tick_params(labelsize=15)

        handles = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label='Low T$_{eq}$'),
            Line2D([0],[0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label='High T$_{eq}$'),
        ]
        fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=15)
        fig.tight_layout(rect=[0,0.05,0.80,1.0])
        plt.show()

    #
    ### ----------- Oxygen escape fluxes and fractionation factor -----------
    #

    @staticmethod
    def oxygen_indicators_planets(df):
        types_with_water = [
            "super-Earths (3% WMF)",
            "sub-Neptunes (10% H2O)",
            "sub-Neptunes (20% H2O)",
            "sub-Neptunes (50% H2O)",
            "sub-Neptunes (70% H2O)",
            "sub-Neptunes (90% H2O)"
        ]
        df_filtered = df[df['planet_type'].isin(types_with_water)].copy()

        color_dict = {
            "super-Earths (3% WMF)": "darkorange",
            "sub-Neptunes (10% H2O)": "powderblue",
            "sub-Neptunes (20% H2O)": "lightskyblue",
            "sub-Neptunes (50% H2O)": "deepskyblue",
            "sub-Neptunes (70% H2O)": "royalblue",
            "sub-Neptunes (90% H2O)": "mediumblue"
        }

        plt.figure(figsize=(7, 5))
        for ptype in types_with_water:
            subset = df_filtered[df_filtered['planet_type'] == ptype]
            if subset.empty:
                continue

            # power-law fit in log-log space
            valid = (subset['x_O'] > 0) & (subset['phi_O'] > 0)
            slope_text = ""
            if valid.sum() > 1:
                x_vals = np.log10(subset.loc[valid, 'x_O'])
                y_vals = np.log10(subset.loc[valid, 'phi_O'])
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                slope_text = f", slope={slope:.2f}"

                # plot trend line but no legend entry
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_line = slope * x_line + intercept
                plt.plot(
                    10**x_line,
                    10**y_line,
                    linestyle='--',
                    color=color_dict[ptype],
                    label='_nolegend_' # <-- this hides the line in the legend
                )

            # now plot the scatter, with the slope in its label
            plt.scatter(subset['x_O'], subset['phi_O'], label=ptype + slope_text,
                # marker=marker_dict[ptype],
                marker='o', color=color_dict[ptype], alpha=0.8, edgecolor='k')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("$x_O$", fontsize=15)
        plt.ylabel("$\phi_O$ (g cm$^{-2}$ s$^{-1}$)", fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def atomic_escape(df):
        # Constants
        amu = 1.660539e-24  # g
        m_H = 1 * amu
        m_O = 16 * amu

        # Convert to atomic fluxes and global escape rates
        df['Phi_H_atoms'] = df['phi_H'] / m_H
        df['Phi_O_atoms'] = df['phi_O'] / m_O
        df['N_H_s'] = df['Phi_H_atoms'] * 4 * np.pi * df['REUV']**2
        df['N_O_s'] = df['Phi_O_atoms'] * 4 * np.pi * df['REUV']**2
        df['ratio_OH_atoms'] = df['N_O_s'] / df['N_H_s']

        # Planet types, water fractions & colors
        types = [
            ("super-Earths (3% WMF)", 1.0),
            ("sub-Neptunes (10% H2O)", 0.10),
            ("sub-Neptunes (20% H2O)", 0.20),
            ("sub-Neptunes (50% H2O)", 0.50),
            ("sub-Neptunes (70% H2O)", 0.70),
            ("sub-Neptunes (90% H2O)", 0.90),
        ]
        color_dict = {
            "super-Earths (3% WMF)": "darkorange",
            "sub-Neptunes (10% H2O)": "powderblue",
            "sub-Neptunes (20% H2O)": "lightskyblue",
            "sub-Neptunes (50% H2O)": "deepskyblue",
            "sub-Neptunes (70% H2O)": "royalblue",
            "sub-Neptunes (90% H2O)": "mediumblue"
        }
        
        plt.figure(figsize=(8, 6))

        for ptype, f_mass in types:
            sub = df[df['planet_type'] == ptype]
            if sub.empty:
                continue

            # Scatter
            # edge_colors = ['red' if reg=='RL' else 'black' for reg in sub['regime']]
            # sizes = sub['m_planet'] / df['m_planet'].max() * 200  
            # sizes = sub['FEUV'] / df['FEUV'].max() * 500**1.1  
            plt.scatter(sub['N_H_s'], sub['ratio_OH_atoms'], marker='o', color=color_dict[ptype],
                        # s=sizes,
                        # edgecolor='darkgray', 
                        alpha=0.7, label=ptype)

            # Mixing-limited maximum O/H ratio
            n_ratio = (1 - f_mass) / f_mass * (18 / 2)
            bound = 1.0 / (2 * (1 + n_ratio))
            plt.axhline(bound, color=color_dict[ptype], linestyle='--', label=f"bulk {int(f_mass*100)}% WMF")

            # # after you compute `bound` for each ptype:
            # signif_half = 0.1 * bound
            # plt.axhline(signif_half, color=color_dict[ptype], linestyle=':', label=f"50% of bulk of {int(f_mass*100)}% WMF")

        # Axis scales and labels
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Hydrogen escape rate $\\dot N_H$ (atoms/s)", fontsize=15)
        plt.ylabel("Atomic escape flux ratio $\\dot N_O/\\dot N_H$", fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.legend(fontsize='small', loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.subplots_adjust(right=0.78)
        # plt.tight_layout()
        plt.show()

    @staticmethod
    def oxygen_indicators_groups(df):
        params = ModelParams()

        types_with_water = [
            "super-Earths (3% WMF)",
            "sub-Neptunes (10% H2O)",
            "sub-Neptunes (20% H2O)",
            "sub-Neptunes (50% H2O)",
            "sub-Neptunes (70% H2O)",
            "sub-Neptunes (90% H2O)"
        ]
        df_filtered = df[df['planet_type'].isin(types_with_water)].copy()

        # variables
        FEUV      = df_filtered['FEUV']
        mass      = df_filtered['m_planet'] / params.mearth
        Teq       = df_filtered['Teq']
        REUV_rel  = df_filtered['REUV'] / df_filtered['r_planet']
        regime    = df_filtered['regime']  # 'EL' or 'RL'
        ptype     = df_filtered['planet_type']

        # discrete colormap for Teq
        unique_temps = np.sort(Teq.unique())
        n_t = len(unique_temps)
        cmap_teq = ListedColormap(plt.cm.coolwarm(np.linspace(0,1,n_t)))
        bounds = np.r_[
            unique_temps[0] - (unique_temps[1]-unique_temps[0])/2,
            (unique_temps[:-1] + unique_temps[1:]) / 2,
            unique_temps[-1] + (unique_temps[-1]-unique_temps[-2])/2
        ]
        norm_teq = BoundaryNorm(bounds, ncolors=n_t)

        # plotting parameters per panel
        vars_to_plot = {
            'F$_{XUV}$ (erg s$^{-1}$ cm$^{-2}$)': {'values': FEUV,     'cmap': 'coolwarm', 'norm': LogNorm(vmin=FEUV.min(), vmax=FEUV.max())},
            'Planet Mass / M$_\oplus$':           {'values': mass,     'cmap': 'coolwarm', 'norm': None},
            'Equilibrium Temp (K)':               {'values': Teq,      'cmap': cmap_teq,  'norm': norm_teq},
            'R$_{XUV}$ (R$_p$)':                  {'values': REUV_rel, 'cmap': 'coolwarm', 'norm': None},
        }

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        # mask for regimes
        mask_RL = regime == 'RL'
        mask_EL = regime == 'EL'

        for ax, (label, pp) in zip(axes, vars_to_plot.items()):
            vals = pp['values']
            cmap = pp['cmap']
            norm = pp['norm']

            # plot EL points: no edge for sub-Neptunes, black for super-Earth
            mask_el_sn = mask_EL & (ptype.str.startswith('sub-Neptunes'))
            mask_el_se = mask_EL & (ptype.str.startswith('super-Earths'))

            sc_el_sn = ax.scatter(
                df_filtered.loc[mask_el_sn, 'x_O'],
                df_filtered.loc[mask_el_sn, 'phi_O'],
                s=50,
                c=vals.loc[mask_el_sn],
                cmap=cmap,
                norm=norm,
                alpha=0.7,
                edgecolors='none',
                zorder=1
            )
            sc_el_se = ax.scatter(
                df_filtered.loc[mask_el_se, 'x_O'],
                df_filtered.loc[mask_el_se, 'phi_O'],
                s=50,
                c=vals.loc[mask_el_se],
                cmap=cmap,
                norm=norm,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5,
                zorder=1
            )

            # plot RL points: red edge for both types
            sc_rl = ax.scatter(
                df_filtered.loc[mask_RL, 'x_O'],
                df_filtered.loc[mask_RL, 'phi_O'],
                s=50,
                c=vals.loc[mask_RL],
                cmap=cmap,
                norm=norm,
                alpha=0.9,
                edgecolors='red',
                linewidths=0.7,
                zorder=2
            )

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel("Oxygen Fractionation Factor")
            ax.set_ylabel("Oxygen Escape Flux")

            # ax.set_xlim(1e-3, 1.5e0)
            # ax.set_ylim(1e-15, 1e-7)

            # Colorbar ticks if discrete Teq panel
            ticks = unique_temps if 'Temp' in label else None
            cbar = plt.colorbar(sc_el_sn if ticks is None else sc_rl, ax=ax, ticks=ticks)
            if ticks is not None:
                cbar.set_ticklabels([f"{t:.0f}" for t in unique_temps])
            cbar.set_label(label)

        plt.tight_layout()
        plt.show()

    #
    ### ----------- Pressure at REUV and Fbol/Teq -----------
    #

    @staticmethod
    def Radius_FXUV_PEUV(df):
        params = ModelParams()
        rearth = params.rearth

        df = df[df['planet_type'].isin(['super-Earths (3% WMF)', 'sub-Neptunes (3% AMF, H/He)'])].copy()
        df['r_norm'] = df['r_planet'] / rearth
        df['P_Pa'] = df['P_EUV'] * 0.1 # cgs (dyn/cm²) to Pa (1 dyn/cm² = 0.1 Pa)

        min_positive_Pa = df.loc[df['P_Pa'] > 0, 'P_Pa'].min()
        max_Pa = df['P_Pa'].max()
        norm = LogNorm(vmin=min_positive_Pa, vmax=max_Pa)

        df_circle = df[df['Teq'] < 500]
        df_square = df[df['Teq'] >= 500]

        plt.figure()
        sc1 = plt.scatter(
            df_circle['FEUV'],
            df_circle['r_norm'],
            c=df_circle['P_Pa'],
            norm=norm,
            cmap='viridis',
            marker='o',
            s=20,
            edgecolor='k',
            label='300/400 K',
            alpha=0.8
        )
        sc2 = plt.scatter(
            df_square['FEUV'],
            df_square['r_norm'],
            c=df_square['P_Pa'],
            norm=norm,
            cmap='viridis',
            marker='s',
            s=20,
            edgecolor='k',
            label='1000/2000 K',
            alpha=0.8
        )
        cbar = plt.colorbar(sc1)
        cbar.set_label('P_EUV [Pa]')

        plt.xscale('log')
        plt.xlabel('XUV Flux F_EUV')
        plt.ylabel('Planet Radius [R$_\\oplus$]')
        plt.legend()
        plt.show()

    #
    ### ----------- Water loss -----------
    #
        
    @staticmethod
    def water_loss(df):
        params = ModelParams()
        # 1) Work on a copy
        d = df.copy()

        # 2) Exclude the pure H/He sub-Neptunes
        d = df[df['planet_type'].str.contains('super-Earths')].copy()

        # 3) Compute water-loss columns
        sec_per_year = 365.25 * 24 * 3600
        sec_per_Myr  = 1e6 * sec_per_year
        ocean_mass   = 1.4e24 # grams

        d['phiO_gs']        = d['phi_O'] * 4*np.pi*d['REUV']**2
        d['phiH2O_gs']      = d['phiO_gs'] * (18/16)
        d['EO_water_1Myr']  = d['phiH2O_gs'] * sec_per_Myr / ocean_mass
        d['EO_water_200Myr']= d['EO_water_1Myr'] * 200

        # 4) Teqs and masses
        teq = d['Teq']
        cmap = colors.LinearSegmentedColormap.from_list('teq_cmap', ['maroon', 'chocolate'])
        norm = colors.Normalize(vmin=teq.min(), vmax=teq.max())
        teq_colors = cmap(norm(teq))

        masses = d['m_planet'] / params.mearth
        sizes  = np.sqrt(masses) * 25**1.2

        # 5) Plot
        fig, ax = plt.subplots(figsize=(8,6))

        # shade region >1 EO/200 Myr
        ymax = d['EO_water_200Myr'].max() * 1.1
        ax.axhspan(1, ymax, color='#B3E5FC', alpha=0.3)

        edgecols = ['red' if r=='RL' else 'black' for r in d['regime']]
        sc = ax.scatter(d['FEUV'], d['EO_water_200Myr'], s=sizes, c=teq_colors, edgecolors=edgecols, alpha=0.8, marker='o')

        # 6) log–log formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('F$_{XUV}$ (erg cm$^{-2}$ s$^{-1}$)', fontsize=15)
        ax.set_ylabel('Water loss per 200 Myr (Earth Oceans)', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)

        # 7) secondary y-axis in exact 1/200 scaling
        def to_per_myr(y):    return y/200
        def to_per_200myr(y): return y*200

        ax2 = ax.secondary_yaxis('right', functions=(to_per_myr, to_per_200myr))
        ax2.set_yscale('log')
        ax2.set_ylabel('Water loss per 1 Myr (Earth oceans)', fontsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=15)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def metallicity_combined(df, delta=0.03):
        params = ModelParams()

        # 1) select only 10% & 20% sub‐Neptunes
        mask = (df['planet_type'].str.contains('sub-Neptunes') & df['planet_type'].str.contains(r'(10% H2O|20% H2O)'))
        d = df[mask].copy()

        # 2) compute Z_init, Z_final (H‐sufficient only)
        sec_per_year     = 365.25*24*3600
        sec200           = 200e6 * sec_per_year

        d['M_atm']       = 0.03 * d['m_planet']
        d['Z_init']      = d['planet_type'].str.extract(r'(10|20)% H2O')[0].astype(float)/100

        # breakdown → loss → remaining → recombine (H‐sufficient only)
        d['M_H2O_init']  = d['Z_init'] * d['M_atm']
        d['M_O_init']    = d['M_H2O_init'] * (16/18)
        d['M_H_init']    = d['M_atm'] - d['M_O_init']

        area = 4*np.pi*d['REUV']**2
        d['M_O_rem']     = (d['M_O_init'] - d['phi_O']*area*sec200).clip(0)
        d['M_H_rem']     = (d['M_H_init'] - d['phi_H']*area*sec200).clip(0)

        can_recombine    = d['M_H_rem'] >= d['M_O_rem']/8
        d = d[can_recombine]

        d['M_H2O_final'] = d['M_O_rem'] * (18/16)
        d['M_H_used']    = d['M_H2O_final'] * (2/18)
        d['M_atm_final'] = d['M_O_rem'] + (d['M_H_rem'] - d['M_H_used'])
        d['Z_final']     = d['M_H2O_final'] / d['M_atm_final']

        # 3) filter ΔZ
        d['dZ'] = d['Z_final'] - d['Z_init']
        d = d[abs(d['dZ']) > delta]

        # 4) build the combined plot
        fig, ax = plt.subplots(figsize=(8,6))

        d_sorted = d.sort_values('m_planet', ascending=False)
        for _, row in d_sorted.iterrows():
            frac = '10%' if '10% H2O' in row['planet_type'] else '20%'
            edge = 'red' if row['regime']=='RL' else 'k'
            size = (row['m_planet']/params.mearth) * 10**1.2
            marker = '^' if '20% H2O' in row['planet_type'] else 'o'
            clr = 'royalblue' if '20% H2O' in row['planet_type'] else 'darkorange'

            # initial point (orange)
            ax.scatter(row['Z_init'], row['FEUV'], marker=marker, s=size, facecolors=clr, edgecolors=edge, linewidth=1.0, alpha=0.9)
            # final point (blue)
            ax.scatter(row['Z_final'], row['FEUV'], marker=marker, s=size, facecolors=clr, edgecolors=edge, linewidth=1.0, alpha=0.9)
            # connector
            ax.plot([row['Z_init'], row['Z_final']], [row['FEUV'], row['FEUV']], color=clr, linewidth=0.8, alpha=0.7)
            # tickmarks
            ax.tick_params(axis='both', which='major', labelsize=15)
            
        ax.set_xscale('linear')
        ax.set_yscale('log')
        ax.set_xlabel('Metallicity Z (M$_{{H_2}O}$/M$_{atm}$)', fontsize=15)
        ax.set_ylabel('F$_{XUV}$ (erg cm$^{-2}$ s$^{-1}$)', fontsize=15)

        handles = [
            Line2D([0],[0], marker='o', color='w', label='initial metallicity', markersize=8, markeredgecolor='k'),
            Line2D([0],[0], marker='^', color='w', label='final metallicity', markersize=8, markeredgecolor='k'),
            Line2D([0],[0], label='|ΔZ|>0.03', color='0.7', linewidth=0.8),
            ]
        ax.legend(handles=handles, loc='upper right', frameon=False)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def OH_vs_mass(df):
        # 1) select only sub-Neptunes with 10 and 20% H2O
        d = df.copy()
        mask = (d['planet_type'].str.contains('sub-Neptunes') & d['planet_type'].str.contains(r'(10% H2O|20% H2O)'))
        d = d[mask]

        # 2) compute initial and remaining O & H
        sec_per_year = 365.25 * 24 * 3600
        sec200       = 200e6 * sec_per_year

        d['M_atm']      = 0.03 * d['m_planet']
        frac = d['planet_type'].str.extract(r'(\d+)% H2O')[0].astype(float)/100
        d['M_H2O_init'] = frac * d['M_atm']
        d['M_O_init']   = d['M_H2O_init'] * (16/18)
        d['M_H_init']   = d['M_atm'] - d['M_O_init']
        d['M_O_rem']    = (d['phi_O'] * 4*np.pi*d['REUV']**2 * sec200).clip(lower=0)
        d['M_H_rem']    = (d['phi_H'] * 4*np.pi*d['REUV']**2 * sec200).clip(lower=0)

        # 3) two panels: O vs mass, H vs mass
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        for ax, comp in zip(axes, ['O', 'H']):
            init = d[f'M_{comp}_init']
            rem  = d[f'M_{comp}_rem']
            m_me = d['m_planet'] / ModelParams().mearth

            # initial points
            ax.scatter(
                init, m_me,
                edgecolors=['red' if r=='RL' else 'k' for r in d['regime']],
                facecolors='C0' if comp=='O' else 'C2',
                alpha=0.8,
                label=f'{comp} init'
            )
            # remaining points
            ax.scatter(
                rem, m_me,
                edgecolors=['red' if r=='RL' else 'k' for r in d['regime']],
                facecolors='C1' if comp=='O' else 'C3',
                alpha=0.8,
                label=f'{comp} rem'
            )

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(f'M$_{{{comp}}}$ (g)')
            ax.set_title({'O':'Oxygen','H':'Hydrogen'}[comp])
            if comp == 'O':
                ax.set_ylabel('Planet Mass Mₚ (M⊕)')
            ax.legend(frameon=False)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def OH_vs_FXUV(df):
        d = df.copy()
        mask = (d['planet_type'].str.contains('sub-Neptunes') & d['planet_type'].str.contains(r'(10% H2O|20% H2O)'))
        d = d[mask]

        sec_per_year = 365.25 * 24 * 3600
        sec200       = 200e6 * sec_per_year

        d['M_atm']      = 0.03 * d['m_planet']
        frac = d['planet_type'].str.extract(r'(\d+)% H2O')[0].astype(float)/100
        d['M_H2O_init'] = frac * d['M_atm']
        d['M_O_init']   = d['M_H2O_init'] * (16/18)
        d['M_H_init']   = d['M_atm'] - d['M_O_init']
        d['M_O_rem']    = (d['phi_O'] * 4*np.pi*d['REUV']**2 * sec200).clip(lower=0)
        d['M_H_rem']    = (d['phi_H'] * 4*np.pi*d['REUV']**2 * sec200).clip(lower=0)

        fig, axes = plt.subplots(1, 2, figsize=(12,6), sharey=True)
        for ax, comp in zip(axes, ['O', 'H']):
            init = d[f'M_{comp}_init']
            rem  = d[f'M_{comp}_rem']
            feuv = d['FEUV']

            ax.scatter(
                feuv, init,
                edgecolors=['red' if r=='RL' else 'k' for r in d['regime']],
                facecolors='C0' if comp=='O' else 'C2',
                alpha=0.8,
                label=f'{comp} init'
            )
            ax.scatter(
                feuv, rem,
                edgecolors=['red' if r=='RL' else 'k' for r in d['regime']],
                facecolors='C1' if comp=='O' else 'C3',
                alpha=0.8,
                label=f'{comp} rem'
            )

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('XUV Flux F$_{XUV}$')
            ax.set_title({'O':'Oxygen','H':'Hydrogen'}[comp])
            if comp == 'O':
                ax.set_ylabel(f'M$_{{{comp}}}$ (g)')
            ax.legend(frameon=False)

        plt.tight_layout()
        plt.show()