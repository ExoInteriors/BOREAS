import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm

from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

from Parameters import ModelParams

class ResultsHandler:
        
    @staticmethod
    def regime_scatter(df_combined):
        """
        Scatter plot showing which regime each planet is in.
        X-axis: FEUV, Y-axis: planet mass (in Earth masses).
        Assumes the combined dataframe contains columns:
            - 'FEUV': EUV flux (erg cm^-2 s^-1)
            - 'm_planet': planet mass in the same units as ModelParams().mearth
            - 'regime': a string with 'RL' for Roche lobe overflow and other strings for energy-limited (or else).
        """
        params = ModelParams()
        mearth = params.mearth

        df = df_combined[df_combined['regime'].notna()].copy()

        df['mass_earth'] = df['m_planet'] / mearth

        # Create a new column for plotting colors based on the regime.
        # "RL" planets will be red, others (assumed "EL") blue.
        df['color'] = df['regime'].apply(lambda x: 'blue' if x == 'EL' else 'red')

        plt.figure(figsize=(3, 3))
        plt.scatter(df['FEUV'], df['mass_earth'], c=df['color'], alpha=0.8)
        plt.xlabel("EUV Flux (erg cm$^{-2}$ s$^{-1}$)")
        plt.ylabel("Planet Mass (Earth Masses)")
        plt.yscale('linear')
        plt.xscale('log')

        legend_elems = [
            Line2D([0], [0], marker='o', color='w', label='EL', markerfacecolor='blue', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='RL', markerfacecolor='red', markersize=8)
        ]
        plt.legend(handles=legend_elems, loc='best')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def Mdot_FEUV_all_planets(df):

        # --- SUB-SAMPLING FUNCTION ---
        def subsample_by_flux_and_mass(group, desired_samples=4):
            group_sorted = group.sort_values('m_planet')
            n = len(group_sorted)
            if n <= desired_samples:
                return group_sorted
            else:
                # Evenly pick indices from the sorted group.
                indices = np.linspace(0, n - 1, desired_samples, dtype=int)
                return group_sorted.iloc[indices]
        
        df_subsample = df.groupby(['FEUV', 'planet_type'], group_keys=False).apply(
            lambda grp: subsample_by_flux_and_mass(grp, desired_samples=3)
        )

        # --- SET UP VISUAL MAPPINGS ---
        planet_types = df['planet_type'].unique()
        markers = ['o', '^', 's'] # as many shapes as planet types
        marker_dict = {ptype: markers[i % len(markers)] for i, ptype in enumerate(planet_types)}

        unique_Teq = sorted(df['Teq'].unique())
        colors = ['gold', 'darkorange', 'chocolate', 'maroon'] # as many colors as Teqs
        teq_color_dict = {teq: colors[i % len(colors)] for i, teq in enumerate(unique_Teq)}

        regime_edge_dict = {'EL': 'black', 'RL': 'red'} # edge colors for regimes

        # --- PLOTTING ---
        plt.figure(figsize=(7, 5))

        for _, row in df_subsample.iterrows():
            x = row['FEUV']
            y = row['Mdot']
            marker = marker_dict[row['planet_type']]                # marker shape based on planet_type
            fcolor = teq_color_dict[row['Teq']]                     # marker face color based on Teq value
            ecolor = regime_edge_dict.get(row['regime'], 'gray')    # marker edge color based on regime; default to gray if regime not found
            size = row['m_planet'] / df['m_planet'].max() * 300     # scale size using m_planet
            
            plt.scatter(x, y, s=size, marker=marker, color=fcolor, edgecolor=ecolor, linewidths=1.5, alpha=0.8, label='_nolegend_')

        plt.xlabel("FEUV (EUV Flux)")
        plt.ylabel("Mdot (Mass Loss Rate, g/s)")
        plt.xscale('log')
        plt.yscale('log')

        # --- COMBINED LEGEND ---
        combined_handles = []

        for ptype in marker_dict:
            combined_handles.append(
                mlines.Line2D([], [], color='black', marker=marker_dict[ptype],
                            linestyle='None', markersize=10, label=f"Planet: {ptype}")
            )

        for teq in teq_color_dict:
            combined_handles.append(
                mpatches.Patch(color=teq_color_dict[teq], label=f"Teq: {teq}")
            )

        for regime in regime_edge_dict:
            combined_handles.append(
                mlines.Line2D([], [], color=regime_edge_dict[regime], marker='o',
                            linestyle='None', markersize=10, label=f"Regime: {regime}")
            )

        plt.legend(handles=combined_handles, loc='lower right',  fontsize='8')

        plt.show()

    @staticmethod
    def Mdot_FEUV_subplots_by_planet(df):

        # --- FUNCTION FOR SUBSAMPLING ---
        # For each group (i.e. fixed FEUV value) within a planet type, sort the rows by m_planet and sample a fixed number of points.
        def subsample_by_flux(group, desired_samples=10):
            group_sorted = group.sort_values('m_planet')
            n = len(group_sorted)
            if n <= desired_samples:
                return group_sorted
            else:
                indices = np.linspace(0, n - 1, desired_samples, dtype=int)
                return group_sorted.iloc[indices]

        # --- SET UP VISUAL MAPPINGS ---

        planet_types = sorted(df['planet_type'].unique())
        marker_dict = {ptype: marker for ptype, marker in zip(planet_types, ['o', '^', 's'])}
        unique_Teq = sorted(df['Teq'].unique())
        colors = ['gold', 'darkorange', 'chocolate', 'maroon']
        teq_color_dict = {teq: colors[i % len(colors)] for i, teq in enumerate(unique_Teq)}

        regime_edge_dict = {'EL': 'black', 'RL': 'red'}

        # --- SET UP THE SUBPLOTS ---
        n_types = len(planet_types)
        fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 4), sharex=True, sharey=True)

        if n_types == 1:
            axes = [axes]

        for ax, ptype in zip(axes, planet_types):
            df_ptype = df[df['planet_type'] == ptype]
            df_subsample = df_ptype.groupby('FEUV', group_keys=False).apply(
                lambda grp: subsample_by_flux(grp, desired_samples=4)
            )

            for _, row in df_subsample.iterrows():
                x = row['FEUV']
                y = row['Mdot']
                marker = marker_dict[ptype]
                facecolor = teq_color_dict[row['Teq']]
                edgecolor = regime_edge_dict.get(row['regime'], 'gray')
                size = row['m_planet'] / df['m_planet'].max() * 300

                ax.scatter(x, y, s=size, marker=marker, color=facecolor,
                        edgecolor=edgecolor, linewidths=1.5, alpha=0.8, label='_nolegend_')

            ax.set_title(f"{ptype}")
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel("FEUV (EUV Flux)")
            ax.set_ylabel("Mdot (Mass Loss Rate, g/s)")

        # --- COMBINED LEGEND ---
        combined_handles = []

        for teq in teq_color_dict:
            combined_handles.append(
                mpatches.Patch(color=teq_color_dict[teq], label=f"Teq: {teq}")
            )

        for regime in regime_edge_dict:
            combined_handles.append(
                mlines.Line2D([], [], color=regime_edge_dict[regime], marker='o',
                            linestyle='None', markersize=10, label=f"Regime: {regime}")
            )

        fig.legend(handles=combined_handles, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0))
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()

    @staticmethod
    def Mdot_FEUV_OxygenLoss(df):
        df_filtered = df[df['planet_type'].isin(["super-Earth", "sub-Neptune (w/ H2O)"])].copy()
        df_filtered['size_ratio'] = (df_filtered['REUV'] / df_filtered['r_planet']) * 100 # scaled ratio REUV / r_planet
        marker_dict = {"super-Earth": "o", "sub-Neptune (w/ H2O)": "s"}
        
        plt.figure(figsize=(7, 5))
        cmap = plt.cm.plasma
        
        for ptype, marker in marker_dict.items():
            subset = df_filtered[df_filtered['planet_type'] == ptype]
            plt.scatter(subset['FEUV'], subset['Mdot'],
                        c=subset['x_O'],        # oxygen-loss indicator (x_O)
                        s=subset['size_ratio'], # marker size reflecting REUV/r_planet
                        cmap=cmap,
                        marker=marker,
                        edgecolor='k',
                        alpha=0.8,
                        label=ptype)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("F$_{EUV}$")
        plt.ylabel("Mass Loss Rate (g/s)")

        cbar = plt.colorbar()
        cbar.set_label("Oxygen Fractionation Factor (x_O)")

        handles = []
        for ptype, marker in marker_dict.items():
            handles.append(mlines.Line2D([], [], marker=marker, color='k', linestyle='None',
                                        markersize=8, label=ptype))
        plt.legend(handles=handles, title="Planet Type", loc='lower right')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def MassRadius_OxygenEscape(df):
        params = ModelParams()
        mearth = params.mearth
        rearth = params.rearth

        df_filtered = df[df['planet_type'].isin(["super-Earth", "sub-Neptune (w/ H2O)"])].copy()
    
        # Create an additional variable that scales REUV relative to r_planet.
        # The ratio REUV/r_planet gives an idea of how extended the EUV–irradiated region is relative to the planet.
        df_filtered['size_ratio'] = (df_filtered['REUV'] / df_filtered['r_planet']) * 100 # scale points

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
        cmap = plt.cm.plasma

        sc1 = axes[0].scatter(df_filtered['m_planet'] / mearth, df_filtered['r_planet'] / rearth,
                          c=df_filtered['x_O'], s=df_filtered['size_ratio'],
                          cmap=cmap, alpha=0.8, edgecolor='k')

        axes[0].set_xlabel("Earth masses")
        axes[0].set_ylabel("Earth radii")
        cbar1 = plt.colorbar(sc1, ax=axes[0])
        cbar1.set_label("Oxygen Fractionation Factor")

        sc2 = axes[1].scatter(df_filtered['m_planet'] / mearth, df_filtered['r_planet'] / rearth,
                          c=df_filtered['phi_O'], s=df_filtered['size_ratio'],
                          cmap=cmap, alpha=0.8, edgecolor='k')

        axes[1].set_xlabel("Earth masses")
        cbar2 = plt.colorbar(sc2, ax=axes[1])
        cbar2.set_label("Oxygen Escape Flux (g cm$^-2$ s$^-1$)")

        marker_dict = {"super-Earth": "o", "sub-Neptune (w/ H2O)": "s"}
        handles = []

        for ptype, marker in marker_dict.items():
            handles.append(mlines.Line2D([], [], marker=marker, color='k', linestyle='None',
                                        markersize=8, label=ptype))
        # Add legend to the first subplot (or you can add it to the entire figure)
        axes[0].legend(handles=handles, loc='lower right')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def REUV_rplanet_vs_OxygenIndicators(df):
        df_filtered = df[df['planet_type'].isin(["super-Earth", "sub-Neptune (w/ H2O)"])].copy()
        df_filtered['REUV_rplanet'] = df_filtered['REUV'] / df_filtered['r_planet']

        marker_dict = {"super-Earth": "o", "sub-Neptune (w/ H2O)": "s"}
        color_dict = {"super-Earth": "tab:blue", "sub-Neptune (w/ H2O)": "tab:orange"}

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        
        # 1. REUV/r_planet vs. x_O
        for ptype, marker in marker_dict.items():
            subset = df_filtered[df_filtered['planet_type'] == ptype]
            axes[0].scatter(subset['REUV_rplanet'], subset['x_O'],
                            marker=marker,
                            color=color_dict[ptype],
                            edgecolor='k',
                            alpha=0.8,
                            label=ptype)
        axes[0].set_xlabel("REUV / r_planet")
        axes[0].set_ylabel("Oxygen Fractionation Factor")
        axes[0].set_xscale('linear')
        axes[0].set_yscale('log')

        # REUV/r_planet vs. phi_O
        for ptype, marker in marker_dict.items():
            subset = df_filtered[df_filtered['planet_type'] == ptype]
            axes[1].scatter(subset['REUV_rplanet'], subset['phi_O'],
                            marker=marker,
                            color=color_dict[ptype],
                            edgecolor='k',
                            alpha=0.8,
                            label=ptype)
        axes[1].set_xlabel("REUV / r_planet")
        axes[1].set_ylabel("Oxygen Escape Flux")
        axes[1].set_xscale('linear')
        axes[1].set_yscale('log')

        handles = []
        for ptype, marker in marker_dict.items():
            handles.append(mlines.Line2D([], [], marker=marker, color='k', linestyle='None', markersize=8, label=ptype))
        axes[0].legend(handles=handles, loc='lower right')

        fig.tight_layout()
        plt.show()

    @staticmethod
    def REUV_rplanet_vs_OxygenIndicators2(df):
        df_filtered = df[df['planet_type'].isin(["super-Earth", "sub-Neptune (w/ H2O)"])].copy()
        df_filtered['REUV_rplanet'] = df_filtered['REUV'] / df_filtered['r_planet']
        df_filtered['marker_size'] = (df_filtered['m_planet'] / df_filtered['m_planet'].max()) * 100
 
        marker_dict = {"super-Earth": "o", "sub-Neptune (w/ H2O)": "s"}

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        cmap = plt.cm.plasma
        
        # 1. Oxygen Fractionation Factor (x_O)
        for ptype, marker in marker_dict.items():
            subset = df_filtered[df_filtered['planet_type'] == ptype]
            sc = axes[0].scatter(subset['REUV_rplanet'], subset['x_O'],
                                s=subset['marker_size'], marker=marker,
                                c=subset['FEUV'], cmap=cmap,
                                edgecolor='k', alpha=0.8,
                                label=ptype)
        axes[0].set_xlabel("REUV / r_planet")
        axes[0].set_ylabel("Oxygen Fractionation Factor (x_O)")
        axes[0].set_xscale('linear')
        axes[0].set_yscale('log')
        cbar0 = plt.colorbar(sc, ax=axes[0])
        cbar0.set_label("FEUV")

        # 2. Oxygen Escape Flux (φ_O)
        for ptype, marker in marker_dict.items():
            subset = df_filtered[df_filtered['planet_type'] == ptype]
            sc2 = axes[1].scatter(subset['REUV_rplanet'], subset['phi_O'],
                                s=subset['marker_size'], marker=marker,
                                c=subset['FEUV'], cmap=cmap,
                                edgecolor='k', alpha=0.8,
                                label=ptype)
        axes[1].set_xlabel("REUV / r_planet")
        axes[1].set_ylabel("Oxygen Escape Flux (φ_O)")
        axes[1].set_xscale('linear')
        axes[1].set_yscale('log')
        cbar1 = plt.colorbar(sc2, ax=axes[1])
        cbar1.set_label("FEUV")

        handles = []
        for ptype, marker in marker_dict.items():
            handles.append(mlines.Line2D([], [], marker=marker, color='k', linestyle='None',
                                        markersize=8, label=ptype))
        axes[0].legend(handles=handles, loc='lower right')

        fig.tight_layout()
        plt.show()

    @staticmethod
    def oxygen_indicators(df):

        df_filtered = df[df['planet_type'].isin(["super-Earth", "sub-Neptune (w/ H2O)"])].copy()
        marker_dict = {"super-Earth": "o", "sub-Neptune (w/ H2O)": "s"}
        color_dict = {"super-Earth": "tab:blue", "sub-Neptune (w/ H2O)": "tab:orange"}
        
        plt.figure(figsize=(7, 5))
        
        for ptype, marker in marker_dict.items():
            subset = df_filtered[df_filtered['planet_type'] == ptype]
            plt.scatter(subset['x_O'], subset['phi_O'], label=ptype, marker=marker, color=color_dict[ptype], alpha=0.8, edgecolor='k')
            
            # compute a regression line in log-log space to see a power-law correlation
            valid = (subset['x_O'] > 0) & (subset['phi_O'] > 0)
            if valid.sum() > 1:
                x_vals = np.log10(subset.loc[valid, 'x_O'])
                y_vals = np.log10(subset.loc[valid, 'phi_O'])
                coeffs = np.polyfit(x_vals, y_vals, 1)
                slope, intercept = coeffs
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100) # line spanning the range of valid x values in log space
                y_line = slope * x_line + intercept
                plt.plot(10**x_line, 10**y_line, color=color_dict[ptype], # convert back from log space
                        linestyle='--', label=f"{ptype} fit: slope={slope:.2f}")
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Oxygen Fractionation Factor")
        plt.ylabel("Oxygen Escape Flux (g cm$^{-2}$ s$^{-1}$)")
        
        plt.legend(loc='best')
        plt.show()
