import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm

from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import LineCollection

from itertools import cycle
from scipy.interpolate import griddata

from Parameters import ModelParams

class ResultsHandler:
        
    @staticmethod
    def regime_scatter(df_combined):
        """Scatter plot showing which regime each planet is in."""
        params = ModelParams()
        mearth = params.mearth
        rearth = params.rearth

        df = df_combined[df_combined['regime'].notna()].copy()

        df['mass_earth'] = df['m_planet'] / mearth

        # Create a new column for plotting colors based on the regime.
        # "RL" planets will be red, others (assumed "EL") blue.
        df['color'] = df['regime'].apply(lambda x: 'black' if x == 'EL' else 'red')

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
    def Mdot_FEUV_all_planets(df, show_regime=True):
        planet_types = sorted(df['planet_type'].unique())
        # planet_types = ["super-Earth", "sub-Neptune (w/o H2O)", "sub-Neptune (10% H2O)"] # restrict to specific ones
        marker_list = ['o','s','D','^','p','v','<','>','*','h','H','X','d']
        marker_cycle = cycle(marker_list)
        marker_dict = {ptype: next(marker_cycle) for ptype in planet_types}

        # use a colormap to spread over unique Teq values
        unique_Teq = sorted(df['Teq'].unique())
        colors = ['gold', 'darkorange', 'chocolate', 'maroon']
        teq_color = {t: colors[i % len(colors)] for i, t in enumerate(unique_Teq)}

        if show_regime:
            edge_map = {'EL':'black', 'RL':'red'}
        else:
            edge_map = {r: 'gray' for r in df['regime'].unique()}

        def subsample_by_flux_and_mass(group, desired_samples=3):
            g = group.sort_values('m_planet')
            n = len(g)
            if n <= desired_samples:
                return g
            idx = np.linspace(0, n-1, desired_samples, dtype=int)
            return g.iloc[idx]

        df_sub = (df
            .groupby(['FEUV', 'planet_type'], group_keys=False)
            .apply(lambda g: subsample_by_flux_and_mass(g))
        )

        plt.figure(figsize=(7,5))
        for _, row in df_sub.iterrows():
            plt.scatter(
                row['FEUV'], row['Mdot'],
                s = row['m_planet']/df['m_planet'].max()*300,
                marker = marker_dict[row['planet_type']],
                facecolor = teq_color[row['Teq']],
                edgecolor = edge_map.get(row['regime'], 'gray'),
                linewidths = 1.5,
                alpha = 0.8,
                label = '_nolegend_'
            )

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("F$_{XUV}$ (erg/cm$^2$/s)")
        plt.ylabel("Mass loss rate (g/s)")

        handles = []
        # Planet‐type legend (filled marker with black edge)
        for p in planet_types:
            handles.append(
                mlines.Line2D([], [], marker=marker_dict[p], color='k',
                              linestyle='None', markersize=8,
                              label=f"{p}")
            )
        # Teq legend (color patches)
        for t in unique_Teq:
            handles.append(
                mpatches.Patch(color=teq_color[t], label=f"Teq = {t} K")
            )
        # Regime legend (empty‐face markers)
        if show_regime:
            for r, ec in edge_map.items():
                handles.append(
                    mlines.Line2D([], [], marker='o', color='w',
                                  markerfacecolor='none', markeredgecolor=ec,
                                  markersize=8, linestyle='None',
                                  label=f"Regime: {r}")
                )

        plt.legend(handles=handles, loc='lower right', fontsize='8', ncol=2)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def Mdot_FEUV_subplots_by_planet(df,
                                     planet_types=None,
                                     exclude_regimes=None,
                                     show_regime=True,
                                     num_midpoints=2,
                                     ncols=2):
        # 1) Determine planet types
        if planet_types is None:
            planet_types = sorted(df['planet_type'].unique())
        else:
            planet_types = [p for p in planet_types if p in df['planet_type'].unique()]

        # 2) Exclude regimes entirely if requested
        df_plot = df.copy()
        if exclude_regimes:
            df_plot = df_plot[~df_plot['regime'].isin(exclude_regimes)]

        # 3) Subsampling helper (always include min/max + num_midpoints in between)
        def subsample_by_flux(group, num_midpoints=0):
            g = group.sort_values('m_planet')
            n = len(g)
            if n <= 2 + num_midpoints:
                return g
            mids = np.linspace(1, n-2, num_midpoints, dtype=int)
            picks = np.unique(np.concatenate(([0], mids, [n-1])))
            return g.iloc[picks]

        # 4) Teq colors & regime edges
        unique_Teq = sorted(df_plot['Teq'].unique())
        teq_palette = ['gold', 'darkorange', 'chocolate', 'maroon']
        teq_color = {t: teq_palette[i % len(teq_palette)]
                     for i, t in enumerate(unique_Teq)}

        if show_regime:
            edge_map = {'EL':'black', 'RL':'red'}
        else:
            edge_map = {r:'gray' for r in df_plot['regime'].unique()}

        # 5) Figure & axes grid
        n = len(planet_types)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4*ncols, 4*nrows),
                                 sharex=True, sharey=True)
        axes_list = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for ax in axes_list[n:]:
            ax.set_visible(False)

        max_mass = df_plot['m_planet'].max()
        for ax, ptype in zip(axes_list, planet_types):
            sub = (
                df_plot[df_plot['planet_type'] == ptype]
                .groupby('FEUV', group_keys=False)
                .apply(lambda g: subsample_by_flux(g, num_midpoints=num_midpoints))
            )
            for _, row in sub.iterrows():
                ax.scatter(
                    row['FEUV'], row['Mdot'],
                    s=row['m_planet']/max_mass * 200,
                    marker='o',
                    facecolor=teq_color[row['Teq']],
                    edgecolor=edge_map.get(row['regime'], 'gray'),
                    linewidths=1.5,
                    alpha=0.8,
                    label='_nolegend_'
                )
            ax.set_title(ptype)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel("F$_{XUV}$ (erg/cm$^2$/s)")
            ax.set_ylabel("Mass loss rate (g/s)")

        # 6) Shared legend
        handles = []
        for t in unique_Teq:
            handles.append(
                mpatches.Patch(color=teq_color[t], label=f"Teq = {t} K")
            )
        if show_regime:
            for r, ec in edge_map.items():
                handles.append(
                    mlines.Line2D([], [], marker='o', color='w',
                                  markerfacecolor='none', markeredgecolor=ec,
                                  markersize=8, linestyle='None',
                                  label=f"Regime: {r}")
                )

        fig.legend(handles=handles,
                   loc='lower center',
                   ncol=len(handles),
                   fontsize='8')
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()

    #
    ### ----------- Mass loss with FEUV (ALTERNATIVE VERSIONS)-----------
    #
    
    @staticmethod
    def Mdot_FEUV_all_planets2(df, show_regime=True, max_size=300, cmap=plt.cm.viridis):
       # dynamic markers
        planet_types = sorted(df['planet_type'].unique())
        marker_list = ['o','s','D','^','p','v','<','>','*','h','H','X','d']
        marker_dict = {p: m for p, m in zip(planet_types, cycle(marker_list))}

        # regime edges
        if show_regime:
            edge_map = {'EL':'black','RL':'red'}
        else:
            edge_map = {r:'gray' for r in df['regime'].unique()}

        # normalize mass → colour
        mass_norm = Normalize(vmin=df['m_planet'].min(),
                              vmax=df['m_planet'].max())

        plt.figure(figsize=(7,5))
        for _, row in (
            df
            .groupby(['FEUV','planet_type'], group_keys=False)
            .apply(lambda g: g.sort_values('m_planet').iloc[::max(1,len(g)//3)])
            .iterrows()
        ):
            c = cmap(mass_norm(row['m_planet']))
            s = row['Teq'] / df['Teq'].max() * max_size
            plt.scatter(
                row['FEUV'], row['Mdot'],
                s=s,
                marker=marker_dict[row['planet_type']],
                facecolor=c,
                edgecolor=edge_map.get(row['regime'],'gray'),
                linewidths=1.2, alpha=0.8,
                label='_nolegend_'
            )

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("F$_{XUV}$ (erg/cm$^2$/s)")
        plt.ylabel("Mass loss rate (g/s)")

        # proper colorbar: bind to current Axes
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mass_norm)
        sm.set_array([])
        ax = plt.gca()
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Planet Mass")

        # size legend for Teq: show min/mid/max
        unique_T = sorted(df['Teq'].unique())
        samples = [unique_T[0]]
        if len(unique_T)>1:
            samples += [unique_T[len(unique_T)//2], unique_T[-1]]
        size_handles = []
        for T in samples:
            sz = T/df['Teq'].max()*max_size
            size_handles.append(
                mlines.Line2D([], [], marker='o', color='gray',
                              linestyle='None',
                              markersize=np.sqrt(sz),
                              label=f"Teq = {T} K")
            )
        # planet‐type legend
        pt_handles = [
            mlines.Line2D([], [], marker=marker_dict[p], color='k',
                          linestyle='None', markersize=8, label=p)
            for p in planet_types
        ]
        # regime legend
        reg_handles = []
        if show_regime:
            for r,ec in edge_map.items():
                reg_handles.append(
                    mlines.Line2D([], [], marker='o', color='w',
                                  markerfacecolor='none',
                                  markeredgecolor=ec,
                                  markersize=8, linestyle='None',
                                  label=f"Regime: {r}")
                )

        all_handles = pt_handles + size_handles + reg_handles
        plt.legend(handles=all_handles, loc='lower right', fontsize='8', ncol=2)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def Mdot_FEUV_subplots_by_planet2(df,
                                        planet_types=None,
                                        exclude_regimes=None,
                                        show_regime=True,
                                        num_midpoints=2,
                                        ncols=2,
                                        max_size=300,
                                        cmap=plt.cm.viridis):
        params = ModelParams()
        mearth = params.mearth

        # 1) pick types
        if planet_types is None:
            planet_types = sorted(df['planet_type'].unique())
        else:
            planet_types = [p for p in planet_types if p in df['planet_type'].unique()]

        # 2) drop regimes
        dfp = df.copy()
        if exclude_regimes:
            dfp = dfp[~dfp['regime'].isin(exclude_regimes)]

        # 3) normalizations (mass in Earth masses)
        mass_earth = dfp['m_planet'] / mearth
        mass_norm = Normalize(vmin=mass_earth.min(), vmax=mass_earth.max())
        max_T = dfp['Teq'].max()

        # 4) regime edges
        if show_regime:
            edge_map = {'EL':'black','RL':'red'}
        else:
            edge_map = {r:'gray' for r in dfp['regime'].unique()}

        # 5) grid shape
        n = len(planet_types)
        rows = math.ceil(n/ncols)
        fig, axes = plt.subplots(rows, ncols,
                                 figsize=(5*ncols,4*rows),
                                 sharex=True, sharey=True)
        axes_list = np.atleast_1d(axes).flatten()
        for ax in axes_list[n:]:
            ax.set_visible(False)

        # helper to pick safe indices
        def _pick_indices(n, num_midpoints):
            if n <= 2 + num_midpoints:
                return np.arange(n)
            mids = np.linspace(1, n-2, num_midpoints, dtype=int)
            return np.unique(np.concatenate(([0], mids, [n-1])))

        # 6) plot panels
        for ax, p in zip(axes_list, planet_types):
            sub = (
                dfp[dfp['planet_type']==p]
                  .groupby('FEUV', group_keys=False)
                  .apply(lambda g: g
                         .sort_values('m_planet')
                         .iloc[_pick_indices(len(g), num_midpoints)]
                  )
            )
            for _, row in sub.iterrows():
                mass_e = row['m_planet'] / mearth
                c = cmap(mass_norm(mass_e))
                s = row['Teq']/max_T * max_size
                ax.scatter(
                    row['FEUV'], row['Mdot'],
                    s=s, marker='o',
                    facecolor=c,
                    edgecolor=edge_map.get(row['regime'],'gray'),
                    linewidths=1.2, alpha=0.8,
                    label='_nolegend_'
                )
            ax.set_title(p)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel("F$_{XUV}$ (erg/cm$^2$/s)"); ax.set_ylabel("Mass loss rate (g/s)")

        # 7) colorbar & legends
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mass_norm)
        sm.set_array([])
        # place colorbar to right of all plots
        fig.subplots_adjust(right=0.82)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Planet Mass (Earth Masses)")

        # size legend for each Teq
        unique_T = sorted(dfp['Teq'].unique())
        size_handles = []
        for T in unique_T:
            sz = T/max_T * max_size
            size_handles.append(
                mlines.Line2D([], [], marker='o', color='gray',
                              linestyle='None',
                              markersize=np.sqrt(sz),
                              label=f"Teq = {T} K")
            )
        # regime legend
        reg_h = []
        if show_regime:
            for r,ec in edge_map.items():
                reg_h.append(
                    mlines.Line2D([], [], marker='o', color='w',
                                  markerfacecolor='none', markeredgecolor=ec,
                                  markersize=8, linestyle='None',
                                  label=f"Regime: {r}")
                )
        # combine legends
        fig.legend(handles=size_handles + reg_h,
                   loc='lower center', ncol=len(size_handles + reg_h),
                   fontsize='8')
        fig.tight_layout(rect=[0, 0.05, 0.82, 1])
        plt.show()

    #
    ### ----------- Oxygen escape fluxes and fractionation factor -----------
    #

    @staticmethod
    def oxygen_indicators(df):
        types_with_water = [
            "super-Earth",
            "sub-Neptune (10% H2O)",
            "sub-Neptune (20% H2O)",
            "sub-Neptune (70% H2O)",
            "sub-Neptune (90% H2O)"
        ]
        df_filtered = df[df['planet_type'].isin(types_with_water)].copy()

        marker_dict = {
            "super-Earth": "o",
            "sub-Neptune (10% H2O)": "v",
            "sub-Neptune (20% H2O)": "^",
            "sub-Neptune (70% H2O)": "s",
            "sub-Neptune (90% H2O)": "D"
        }
        color_dict = {
            "super-Earth": "tab:blue",
            "sub-Neptune (10% H2O)": "tab:orange",
            "sub-Neptune (20% H2O)": "tab:red",
            "sub-Neptune (70% H2O)": "tab:purple",
            "sub-Neptune (90% H2O)": "tab:green"
        }

        plt.figure(figsize=(7, 5))
        for ptype in types_with_water:
            subset = df_filtered[df_filtered['planet_type'] == ptype]
            if subset.empty:
                continue

            plt.scatter(
                subset['x_O'],
                subset['phi_O'],
                label=ptype,
                marker=marker_dict[ptype],
                color=color_dict[ptype],
                alpha=0.8,
                edgecolor='k'
            )

            # power-law fit in log-log space
            valid = (subset['x_O'] > 0) & (subset['phi_O'] > 0)
            if valid.sum() > 1:
                x_vals = np.log10(subset.loc[valid, 'x_O'])
                y_vals = np.log10(subset.loc[valid, 'phi_O'])
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_line = slope * x_line + intercept

                plt.plot(
                    10**x_line,
                    10**y_line,
                    linestyle='--',
                    color=color_dict[ptype],
                    label=f"{ptype} fit: slope={slope:.2f}"
                )

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Oxygen Fractionation Factor")
        plt.ylabel("Oxygen Escape Flux (g cm$^{-2}$ s$^{-1}$)")
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def radius_FEUV_oxygen_contour(df_combined):

              # ——— 1) normalize & classify ———
        params = ModelParams()
        rearth, mearth = params.rearth, params.mearth

        df = (df_combined
              .dropna(subset=['FEUV','r_planet','m_planet','x_O','regime','planet_type','Teq'])
              .copy())
        df['R_earth'] = df['r_planet'] / rearth
        df['M_earth'] = df['m_planet'] / mearth

        # low/high Teq ⇒ circle/square
        df['marker'] = df['Teq'].apply(lambda T: 'o' if T <= 400 else 's')
        # regime ⇒ edgecolor
        df['ecolor'] = df['regime'].map({'EL':'k','RL':'red'})

        # ——— 2) build global grid & color‐scale ———
        xi = np.logspace(np.log10(df['FEUV'].min()),
                         np.log10(df['FEUV'].max()), 200)
        yi = np.linspace(df['R_earth'].min(),
                         df['R_earth'].max(), 200)
        X, Y = np.meshgrid(xi, yi)
        vmin, vmax = df['x_O'].min(), df['x_O'].max()

        # ——— 3) set up 2×2 axes ———
        fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                                 sharex=True, sharey=True)
        axes = axes.flatten()

        # define our four panels: (title, mask, scatter_flag)
        panels = [
            ("Super-Earths (3% WMF)",               df['planet_type']=='super-Earth',               True),
            ("Sub-Neptunes (3% AMF, 10-90% water)", df['planet_type'].str.contains('sub-Neptune'),  True),
            ("Super-Earths (3% WMF)",               df['planet_type']=='super-Earth',               False),
            ("Sub-Neptunes (3% AMF, 10-90% water)", df['planet_type'].str.contains('sub-Neptune'),  False),
        ]

        for ax, (title, mask, do_scatter) in zip(axes, panels):
            sub = df[mask]
            if sub.empty:
                ax.set_title(title)
                ax.axis('off')
                continue

            # contour of log10(x_O)
            Z = griddata((sub['FEUV'], sub['R_earth']),
                         sub['x_O'], (X, Y), method='linear')
            cf = ax.contourf(
                X, Y, np.log10(Z),
                levels=20, cmap='viridis',
                vmin=np.log10(vmin), vmax=np.log10(vmax),
                alpha=0.7
            )

            if do_scatter:
                # overlay points
                for (regime, marker), grp in sub.groupby(['regime','marker']):
                    ax.scatter(
                        grp['FEUV'], grp['R_earth'],
                        c=grp['x_O'],
                        norm=LogNorm(vmin=vmin, vmax=vmax),
                        cmap='viridis',
                        s=(np.sqrt(grp['M_earth'])) * 20,
                        marker=marker,
                        edgecolor=grp['ecolor'].iloc[0],
                        linewidth=0.8,
                        alpha=0.9
                    )

            ax.set_xscale('log')
            ax.set_title(title, fontsize=12)

        # shared axis labels
        axes[2].set_xlabel("F$_{XUV}$ (erg/cm$^2$/s)", fontsize=11)
        axes[3].set_xlabel("F$_{XUV}$ (erg/cm$^2$/s)", fontsize=11)
        axes[0].set_ylabel("Planet Radius (R$_\oplus$)", fontsize=11)
        axes[2].set_ylabel("Planet Radius (R$_\oplus$)", fontsize=11)

        # ——— 4) carve out space for colorbar & legend ———
        fig.subplots_adjust(right=0.80)

        # colorbar axes: [left, bottom, width, height]
        cax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(cf, cax=cax)
        cbar.set_label('log$_{10}$ x$_O$', fontsize=11)

        # single legend for marker shape & edgecolor
        handles = [
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='gray', markeredgecolor='k', markersize=8,
                   label='Low T$_{eq}$ (≤400 K)'),
            Line2D([0],[0], marker='s', color='w',
                   markerfacecolor='gray', markeredgecolor='k', markersize=8,
                   label='High T$_{eq}$ (>400 K)'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='k', markersize=8,
                   label='EL regime'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='red', markersize=8,
                   label='RL regime'),
        ]
        fig.legend(handles=handles,
                   loc='upper right', frameon=False, fontsize=10)

        # ——— 5) final tight layout for subplots only ———
        fig.tight_layout(rect=[0, 0.05, 0.80, 1.0])
        plt.show()

    @staticmethod
    def mass_FEUV_oxygen_contour(df_combined):
        params = ModelParams()
        rearth, mearth = params.rearth, params.mearth

        # clean & normalize
        df = (df_combined
              .dropna(subset=['FEUV','r_planet','m_planet','x_O','regime','planet_type','Teq'])
              .copy())
        df['M_earth'] = df['m_planet'] / mearth
        df['marker'] = df['Teq'].apply(lambda T: 'o' if T <= 400 else 's')
        df['ecolor'] = df['regime'].map({'EL':'k','RL':'red'})

        # contour grid in (FEUV, M_earth)
        xi = np.logspace(np.log10(df['FEUV'].min()),
                         np.log10(df['FEUV'].max()), 200)
        yi = np.linspace(df['M_earth'].min(),
                         df['M_earth'].max(), 200)
        X, Y = np.meshgrid(xi, yi)
        vmin, vmax = df['x_O'].min(), df['x_O'].max()

        # 2×2 figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                                 sharex=True, sharey=True)
        axes = axes.flatten()

        masks = [
            ("Super-Earths (3% WMF)",               df['planet_type']=='super-Earth'),
            ("Sub-Neptunes (3% AMF, 10-90% water)", df['planet_type'].str.contains('sub-Neptune'))
        ]

        panels = [
            (masks[0][0], masks[0][1], True),
            (masks[1][0], masks[1][1], True),
            (masks[0][0], masks[0][1], False),
            (masks[1][0], masks[1][1], False),
        ]

        for ax, (title, mask, do_scatter) in zip(axes, panels):
            sub = df[mask]
            if sub.empty:
                ax.set_title(title)
                ax.axis('off')
                continue

            # contour
            Z = griddata((sub['FEUV'], sub['M_earth']),
                         sub['x_O'], (X, Y), method='linear')
            cf = ax.contourf(
                X, Y, np.log10(Z),
                levels=20, cmap='viridis',
                vmin=np.log10(vmin), vmax=np.log10(vmax),
                alpha=0.7
            )

            # scatter overlay?
            if do_scatter:
                for (regime, marker), grp in sub.groupby(['regime','marker']):
                    ax.scatter(
                        grp['FEUV'], grp['M_earth'],
                        c=grp['x_O'],
                        norm=LogNorm(vmin=vmin, vmax=vmax),
                        cmap='viridis',
                        s=(np.sqrt(grp['M_earth'])) * 20,
                        marker=marker,
                        edgecolor=grp['ecolor'].iloc[0],
                        linewidth=0.8,
                        alpha=0.9
                    )

            ax.set_xscale('log')
            ax.set_title(title, fontsize=12)

        # axis labels
        axes[2].set_xlabel("F$_{XUV}$ (erg/cm$^2$/s)", fontsize=11)
        axes[3].set_xlabel("F$_{XUV}$ (erg/cm$^2$/s)", fontsize=11)
        axes[0].set_ylabel("Planet Mass (M$_\oplus$)", fontsize=11)
        axes[2].set_ylabel("Planet Mass (M$_\oplus$)", fontsize=11)

        # colorbar on the right
        fig.subplots_adjust(right=0.80)
        cax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(cf, cax=cax)
        cbar.set_label('log$_{10}$ x$_O$', fontsize=11)

        # legend for shape & edgecolor
        handles = [
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='gray', markeredgecolor='k', markersize=8,
                   label='Low T$_{eq}$ (≤400 K)'),
            Line2D([0],[0], marker='s', color='w',
                   markerfacecolor='gray', markeredgecolor='k', markersize=8,
                   label='High T$_{eq}$ (>400 K)'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='k', markersize=8,
                   label='EL regime'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='red', markersize=8,
                   label='RL regime'),
        ]
        fig.legend(handles=handles,
                   loc='upper right', frameon=False, fontsize=9)

        fig.tight_layout(rect=[0, 0.05, 0.80, 1.0])
        plt.show()

    @staticmethod
    def radius_FEUV_oxygen_contour_by_planet_type(df_combined):
        '''Radius FXUV plots for however many types of planets are available'''
        params = ModelParams()
        rearth = params.rearth
        mearth = params.mearth

        # clean and normalize
        df = (df_combined
              .dropna(subset=['FEUV','r_planet','m_planet','x_O','regime','planet_type','Teq'])
              .copy())
        df['R_earth'] = df['r_planet'] / rearth
        df['M_earth'] = df['m_planet'] / mearth

        # classify low/high Teq
        df['marker'] = df['Teq'].apply(lambda T: 'o' if T <= 400 else 's')
        df['ecolor'] = df['regime'].map({'EL':'k','RL':'red'})

        # common grid for contours
        xi = np.logspace(np.log10(df['FEUV'].min()), np.log10(df['FEUV'].max()), 200)
        yi = np.linspace(df['R_earth'].min(), df['R_earth'].max(), 200)
        X, Y = np.meshgrid(xi, yi)

        vmin, vmax = df['x_O'].min(), df['x_O'].max()

        # subplot grid
        types = df['planet_type'].unique()
        n = len(types)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows),
                                 sharex=True, sharey=True)

        axes = axes.flatten()

        for idx, (ax, ptype) in enumerate(zip(axes, types)):
            sub = df[df['planet_type'] == ptype]
            if sub.empty:
                ax.axis('off')
                continue

            # contour
            Z = griddata((sub['FEUV'], sub['R_earth']),
                         sub['x_O'], (X, Y), method='linear')
            cf = ax.contourf(
                X, Y, np.log10(Z),
                levels=20, cmap='viridis',
                vmin=np.log10(vmin), vmax=np.log10(vmax),
                alpha=0.7
            )

            # scatter
            for (regime, marker), grp in sub.groupby(['regime','marker']):
                ax.scatter(
                    grp['FEUV'], grp['R_earth'],
                    c=grp['x_O'],
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                    cmap='viridis',
                    s=(np.sqrt(grp['M_earth'])) * 20,
                    marker=marker,
                    edgecolor=grp['ecolor'].iloc[0],
                    linewidth=0.8,
                    alpha=0.9
                )

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

        # 1) reserve left 80% for subplots
        fig.subplots_adjust(right=0.80)

        # 2) explicit colorbar axes in the remaining 20%
        cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(cf, cax=cbar_ax)
        cbar.set_label('log$_{10}$ x$_O$', fontsize=11)

        # shared legend for Teq‐shape and regime‐edgecolor
        legend_handles = [
            Line2D([0],[0], marker='o', color='w', label='Low T$_{eq}$ (300/400 K)',
                   markerfacecolor='gray', markeredgecolor='k', markersize=8),
            Line2D([0],[0], marker='s', color='w', label='High T$_{eq}$ (1000/2000 K)',
                   markerfacecolor='gray', markeredgecolor='k', markersize=8),
            Line2D([0],[0], marker='o', color='w', label='EL regime',
                   markerfacecolor='white', markeredgecolor='k', markersize=8),
            Line2D([0],[0], marker='o', color='w', label='RL regime',
                   markerfacecolor='white', markeredgecolor='red', markersize=8)
        ]
        fig.legend(handles=legend_handles,
                   loc='upper right', frameon=False, fontsize=9)

        # 3) tighten subplots within left 80%
        fig.tight_layout(rect=[0, 0.05, 0.80, 1.0])
        plt.show()

    #
    ### ----------- Oxygen escape in mass-radius, or similar space -----------
    #

    @staticmethod
    def mass_radius_xO(df_combined):
        """Mass-Radius oxygen escape: contours of log10(x_O) + scatter."""
        params = ModelParams()
        rearth, mearth = params.rearth, params.mearth

        # normalize & classify
        df = df_combined.dropna(subset=['FEUV','r_planet','m_planet','x_O','regime','Teq','planet_type']).copy()
        df['M_earth'] = df['m_planet']/mearth
        df['R_earth'] = df['r_planet']/rearth
        df['marker']  = df['Teq'].apply(lambda T: 'o' if T <= 400 else 's')
        df['ecolor']  = df['regime'].map({'EL':'k','RL':'red'})
        feuv = df['FEUV']
        df['size']   = ((feuv - feuv.min())/(feuv.max()-feuv.min())*50) + 50

        # global color‐scale
        vmin, vmax = df['x_O'].min(), df['x_O'].max()

        # two panels
        panels = [
            ("Super-Earths (3% WMF)",               df['planet_type']=='super-Earth'),
            ("Sub-Neptunes (3% AMF, 10-90% water)", df['planet_type'].str.contains('sub-Neptune'))
        ]
        fig, axes = plt.subplots(1, 2, figsize=(12,5), sharey=True)

        for ax, (title, mask) in zip(axes, panels):
            sub = df[mask]
            if sub.empty:
                ax.set_title(title)
                ax.axis('off')
                continue

            # panel‐specific grid
            xi = np.logspace(np.log10(sub['M_earth'].min()),
                             np.log10(sub['M_earth'].max()), 200)
            yi = np.linspace(sub['R_earth'].min(),
                             sub['R_earth'].max(), 200)
            X, Y = np.meshgrid(xi, yi)

            # contour for this subset
            Z = griddata(
                (sub['M_earth'], sub['R_earth']),
                sub['x_O'],
                (X, Y), method='linear'
            )

            cf = ax.contourf(
                X, Y, np.log10(Z),
                levels=20, cmap='viridis', alpha=0.7,
                vmin=np.log10(vmin), vmax=np.log10(vmax)
            )

            # scatter overlay
            for (regime, marker), grp in sub.groupby(['regime','marker']):
                ax.scatter(
                    grp['M_earth'], grp['R_earth'],
                    c=grp['x_O'],
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                    cmap='viridis',
                    s=grp['size'],
                    marker=marker,
                    edgecolor=grp['ecolor'].iloc[0],
                    linewidth=0.8, alpha=0.9
                )

            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Planet Mass (M$_\oplus$)", fontsize=11)

        axes[0].set_ylabel("Planet Radius (R$_\oplus$)", fontsize=11)

        # carve out space & draw colorbar
        fig.subplots_adjust(right=0.80)
        cax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(cf, cax=cax)
        cbar.set_label('log$_{10}$ x$_O$', fontsize=11)

        # legend
        handles = [
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='gray', markeredgecolor='k',
                   markersize=8, label='T$_{eq}$ ≤400 K'),
            Line2D([0],[0], marker='s', color='w',
                   markerfacecolor='gray', markeredgecolor='k',
                   markersize=8, label='T$_{eq}$ >400 K'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='k',
                   markersize=8, label='EL regime'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='red',
                   markersize=8, label='RL regime'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='none',
                   markersize=6, label='marker size ∝ FEUV'),
        ]
        fig.legend(handles=handles,
                   loc='upper right', frameon=False, fontsize=9)

        plt.tight_layout(rect=[0,0.05,0.80,1])
        plt.show()

    @staticmethod
    def mass_REUV_xO(df_combined):
        """Two-panel map of log10(x_O) vs M_p and R_EUV (cm)."""
        params = ModelParams()
        re, me = params.rearth, params.mearth

        df = (df_combined.dropna(subset=['FEUV','REUV','m_planet','x_O','regime','Teq','planet_type']).copy())
        df['M_earth'] = df['m_planet'] / me
        df['REUV_cm'] = df['REUV']
        df['marker']  = df['Teq'].apply(lambda T: 'o' if T <= 400 else 's')
        df['ecolor']  = df['regime'].map({'EL':'k','RL':'red'})
        feuv = df['FEUV']
        df['size']    = ((feuv - feuv.min())/(feuv.max()-feuv.min())*50) + 50

        # global log-range for x_O
        log_vmin, log_vmax = np.log10(df['x_O'].min()), np.log10(df['x_O'].max())

        panels = [
            ("Super-Earths (3% WMF)",               df['planet_type']=='super-Earth'),
            ("Sub-Neptunes (3% AMF, 10-90% H₂O)",   df['planet_type'].str.contains('sub-Neptune'))
        ]
        fig, axes = plt.subplots(1, 2, figsize=(12,5), sharey=True)

        for ax, (title, mask) in zip(axes, panels):
            sub = df[mask]
            if sub.empty:
                ax.set_title(title)
                ax.axis('off')
                continue

            # grid in (M_earth, REUV_cm)
            xi = np.logspace(np.log10(sub['M_earth'].min()),
                             np.log10(sub['M_earth'].max()), 200)
            yi = np.linspace(sub['REUV_cm'].min(),
                             sub['REUV_cm'].max(), 200)
            X, Y = np.meshgrid(xi, yi)

            Z = griddata((sub['M_earth'], sub['REUV_cm']),
                         sub['x_O'], (X, Y), method='linear')
            Z = np.ma.masked_invalid(Z)

            # contour in log10(x_O)
            cf = ax.contourf(
                X, Y, np.log10(Z),
                levels=20,
                vmin=log_vmin, vmax=log_vmax,
                cmap='viridis', alpha=0.7
            )

            # scatter with true x_O (for color consistency)
            for (regime, marker), grp in sub.groupby(['regime','marker']):
                ax.scatter(
                    grp['M_earth'], grp['REUV_cm'],
                    c=np.log10(grp['x_O']),
                    vmin=log_vmin, vmax=log_vmax,
                    cmap='viridis',
                    s=grp['size'],
                    marker=marker,
                    edgecolor=grp['ecolor'].iloc[0],
                    linewidth=0.8, alpha=0.9
                )

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Planet Mass (M$_\oplus$)", fontsize=11)

        axes[0].set_ylabel("R$_{EUV}$ (cm)", fontsize=11)

        # colorbar (log10 x_O)
        fig.subplots_adjust(right=0.80)
        cax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(cf, cax=cax)
        cbar.set_label('log$_{10}$ x$_O$', fontsize=11)

        # legend
        handles = [
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='gray', markeredgecolor='k', markersize=8,
                   label='T$_{eq}$ ≤400 K'),
            Line2D([0],[0], marker='s', color='w',
                   markerfacecolor='gray', markeredgecolor='k', markersize=8,
                   label='T$_{eq}$ >400 K'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='k', markersize=8,
                   label='EL regime'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='red', markersize=8,
                   label='RL regime'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='none', markersize=6,
                   label='marker size ∝ F$_{XUV}$'),
        ]
        fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=9)

        plt.tight_layout(rect=[0, 0.05, 0.80, 1])
        plt.show()

    @staticmethod
    def mass_REUV_xO_normalized(df_combined):
        """Two-panel map of log10(x_O) vs M_p and R_EUV/R_p (log axes)."""
        params = ModelParams()
        re, me = params.rearth, params.mearth

        df = (df_combined.dropna(subset=['FEUV','REUV','r_planet','m_planet','x_O','regime','Teq','planet_type']).copy())
        df['M_earth']   = df['m_planet'] / me
        df['REUV_norm'] = df['REUV']      / df['r_planet']
        df['marker']    = df['Teq'].apply(lambda T: 'o' if T <= 400 else 's')
        df['ecolor']    = df['regime'].map({'EL':'k','RL':'red'})
        feuv = df['FEUV']
        df['size']      = ((feuv - feuv.min())/(feuv.max()-feuv.min()) * 50) + 50

        log_vmin, log_vmax = np.log10(df['x_O'].min()), np.log10(df['x_O'].max())

        panels = [
            ("Super-Earths (3% WMF)",               df['planet_type']=='super-Earth'),
            ("Sub-Neptunes (3% AMF, 10-90% H₂O)",   df['planet_type'].str.contains('sub-Neptune'))
        ]
        fig, axes = plt.subplots(1, 2, figsize=(12,5), sharey=True)

        for ax, (title, mask) in zip(axes, panels):
            sub = df[mask]
            if sub.empty:
                ax.set_title(title)
                ax.axis('off')
                continue

            xi = np.logspace(np.log10(sub['M_earth'].min()),
                             np.log10(sub['M_earth'].max()), 200)
            yi = np.logspace(np.log10(sub['REUV_norm'].min()),
                             np.log10(sub['REUV_norm'].max()), 200)
            X, Y = np.meshgrid(xi, yi)

            Z = griddata((sub['M_earth'], sub['REUV_norm']),
                         sub['x_O'], (X, Y), method='linear')
            Z = np.ma.masked_invalid(Z)

            cf = ax.contourf(
                X, Y, np.log10(Z),
                levels=20,
                vmin=log_vmin, vmax=log_vmax,
                cmap='viridis', alpha=0.7
            )

            for (regime, marker), grp in sub.groupby(['regime','marker']):
                ax.scatter(
                    grp['M_earth'], grp['REUV_norm'],
                    c=np.log10(grp['x_O']),
                    vmin=log_vmin, vmax=log_vmax,
                    cmap='viridis',
                    s=grp['size'],
                    marker=marker,
                    edgecolor=grp['ecolor'].iloc[0],
                    linewidth=0.8, alpha=0.9
                )

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Planet Mass (M$_\oplus$)", fontsize=11)

        axes[0].set_ylabel("R$_{EUV}$/R$_p$", fontsize=11)

        fig.subplots_adjust(right=0.80)
        cax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(cf, cax=cax)
        cbar.set_label('log$_{10}$ x$_O$', fontsize=11)

        handles = [
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='gray', markeredgecolor='k', markersize=8,
                   label='T$_{eq}$ ≤400 K'),
            Line2D([0],[0], marker='s', color='w',
                   markerfacecolor='gray', markeredgecolor='k', markersize=8,
                   label='T$_{eq}$ >400 K'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='k', markersize=8,
                   label='EL regime'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='red', markersize=8,
                   label='RL regime'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='none', markersize=6,
                   label='marker size ∝ F$_{XUV}$'),
        ]
        fig.legend(handles=handles,
                   loc='upper right', frameon=False, fontsize=9)

        plt.tight_layout(rect=[0, 0.05, 0.80, 1])
        plt.show()