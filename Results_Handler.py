import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from matplotlib.lines import Line2D

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
        plt.figure(figsize=(8, 6))

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

        plt.legend(handles=combined_handles, loc='lower right', title="Legend")

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
        fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 5), sharex=True, sharey=True)

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

            ax.set_title(f"Planet Type: {ptype}")
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

        fig.legend(handles=combined_handles, loc='lower center', ncol=4, title="Legend", bbox_to_anchor=(0.5, -0.05))
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()
