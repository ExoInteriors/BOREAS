import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.tri as mtri
import matplotlib as mpl

from matplotlib.lines import Line2D
from matplotlib import colors
from matplotlib.colors import LogNorm
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
        params = ModelParams()
        mearth = params.mearth
    
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

        def subsample_by_flux_and_mass(group, desired=6):
            g = group.sort_values('m_planet')
            n = len(g)
            if n <= desired:
                return g

            idx_min = 0
            idx_max = n - 1

            n_middle = desired - 2
            if n_middle > 0:
                middle_idxs = np.linspace(1, n - 2, n_middle, dtype=int)
                picks = [idx_min] + middle_idxs.tolist() + [idx_max]
            else:
                picks = [idx_min, idx_max]

            return g.iloc[picks]

        # build 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        for ax, ptype in zip(axes, selected):
            sub = df[df['planet_type'] == ptype]
            sub_s = sub.groupby('FEUV', group_keys=False).apply(subsample_by_flux_and_mass)

            local_min = sub['m_planet'].min()
            local_max = sub['m_planet'].max()

            for _, row in sub_s.iterrows():
                ax.scatter(
                    row['FEUV'], row['Mdot'],
                    s = row['m_planet']/local_max * 200,
                    marker = marker,
                    facecolor = teq_color[row['Teq']],
                    edgecolor = edge_map.get(row['regime'], 'gray'),
                    linewidths = 1,
                    alpha = 0.8
                )

            for Teq_value in sorted(sub['Teq'].unique()):
                # a) get sorted FEUV values for this Teq
                feuv_list = sorted(sub.loc[sub['Teq'] == Teq_value, 'FEUV'].unique())
                # make sure there are at least 3 entries
                if len(feuv_list) < 3:
                    continue
                third_feuv = feuv_list[2]

                # b) restrict to rows with this Teq and mid‐FEUV
                mask_third = (sub['Teq'] == Teq_value) & (sub['FEUV'] == third_feuv)
                sub_third = sub.loc[mask_third]

                # c) find the min-mass row and max-mass row in that slice
                row_min = sub_third.loc[sub_third['m_planet'].idxmin()]
                row_max = sub_third.loc[sub_third['m_planet'].idxmax()]

                # convert to Earth masses once
                mass_min_earth = row_min['m_planet'] / params.mearth
                mass_max_earth = row_max['m_planet'] / params.mearth

                # d) Annotate the two points
                ax.text(
                    row_min['FEUV'] * 0.20,
                    row_min['Mdot'] * 10.00,
                    f"{mass_min_earth:.1f} M$_\oplus$",
                    fontsize=15,
                    ha='left',
                    va='bottom',
                    alpha=0.8
                )
                ax.text(
                    row_max['FEUV'] * 1.02,
                    row_max['Mdot'] * 0.05,
                    f"{mass_max_earth:.1f} M$_\oplus$",
                    fontsize=15,
                    ha='left',
                    va='bottom',
                    alpha=0.8
                )

            # place planet type label inside
            ax.text(0.02, 0.98, ptype, transform=ax.transAxes, va='top', ha='left', fontsize=15)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.tick_params(axis='both', which='major', labelsize=25)

            # Add local legend with min/max mass values
            # mass_handles = [
            #     mlines.Line2D([], [], linestyle='None', marker='o', color='black', label=f"Min mass: {local_min/params.mearth:.2f} M$_\oplus$"),
            #     mlines.Line2D([], [], linestyle='None', marker='o', color='black', label=f"Max mass: {local_max/params.mearth:.2f} M$_\oplus$", markersize=12)
            # ]
            # ax.legend(handles=mass_handles, loc='lower right', fontsize=12, frameon=False)

        # common axes labels with spacing
        fig.text(0.5, 0.01, "F$_{XUV}$ (erg cm$^2$ s$^{-1}$)", ha='center', fontsize=25)
        fig.text(0.001, 0.5, "Mass loss rate (g s$^{-1}$)", va='center', rotation='vertical', fontsize=25)

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

        panels = [
            ("Super-Earths (3% WMF)", df['planet_type'].str.contains('super-Earths')),
            ("Sub-Neptunes (3% AMF, 10-90% WMF)", df['planet_type'].str.contains('sub-Neptunes'))
        ]

        fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)

        for ax, (title, mask) in zip(axes, panels):
            sub = df[mask]
            if sub.empty:
                ax.set_title(title, fontsize=15)
                ax.axis('off')
                continue

            Z_raw = griddata((sub['FEUV'], sub['R_earth']), sub['x_O'], (X, Y), method='linear')
            Z_raw = np.ma.masked_invalid(Z_raw)
            Z = np.log10(Z_raw)

            cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, extend='min', alpha=0.8)

            for (_, marker), grp in sub.groupby(['regime', 'marker']):
                vals = np.log10(grp['x_O'].values)
                vals = np.clip(vals, clamp, None)
                ax.scatter(grp['FEUV'], grp['R_earth'], c=vals, cmap=cmap, norm=norm,
                        marker=marker, edgecolor=grp['ecolor'].iloc[0],
                        linewidth=0.8, alpha=0.5)

            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.set_xscale('log')
            ax.set_title(title, fontsize=15)

        fig.text(0.5, 0.04, "F$_{XUV}$ (erg/cm$^2$/s)", ha='center', fontsize=20)
        fig.text(0.02, 0.5, "Planet radius (R$_\oplus$)", va='center', rotation='vertical', fontsize=20)

        fig.subplots_adjust(left=0.15, right=0.85, hspace=0.3)
        cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(cf, cax=cax, extend='min')
        cbar.set_label('log$_{10}$ x$_O$', fontsize=15)
        ticks = np.linspace(clamp, 0, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.0f}" for t in ticks])
        cbar.ax.tick_params(labelsize=15)

        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label='Low T$_{eq}$'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label='High T$_{eq}$'),
        ]

        fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=15)
        fig.tight_layout(rect=[0, 0.05, 0.88, 1.0])
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
        cmap  = plt.get_cmap('coolwarm')
        cmap.set_under(cmap(0))
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
        ]

        fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)

        for ax, (title, mask, do_scatter) in zip(axes, panels):
            sub = df[mask]
            if sub.empty:
                ax.set_title(title, fontsize=15)
                ax.axis('off')
                continue

            Z_raw = griddata((sub['FEUV'], sub['M_earth']), sub['x_O'], (X, Y), method='linear')
            Z_raw = np.ma.masked_invalid(Z_raw)
            Z = np.log10(Z_raw)

            cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, extend='min', alpha=0.8)

            for (_, marker), grp in sub.groupby(['regime','marker']):
                vals = np.log10(grp['x_O'].values)
                vals = np.clip(vals, clamp, None)
                ax.scatter(grp['FEUV'], grp['M_earth'], c=vals, cmap=cmap, norm=norm, marker=marker, edgecolor=grp['ecolor'].iloc[0], linewidth=0.8, alpha=0.5)
            
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.set_xscale('log')
            ax.set_title(title, fontsize=15)

        fig.text(0.5, 0.04, "F$_{XUV}$ (erg/cm$^2$/s)", ha='center', fontsize=20)
        fig.text(0.02, 0.5, "Planet mass (M$_\oplus$)", va='center', rotation='vertical', fontsize=20)

        fig.subplots_adjust(left=0.12, right=0.88, hspace=0.3)
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(cf, cax=cax, extend='min')
        cbar.set_label('log$_{10}$ x$_O$', fontsize=15)
        ticks = np.linspace(clamp, 0, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.0f}" for t in ticks])
        cbar.ax.tick_params(labelsize=20)

        handles = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label='Low T$_{eq}$'),
            Line2D([0],[0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label='High T$_{eq}$'),
        ]

        fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=15)
        fig.tight_layout(rect=[0, 0.05, 0.88, 1.0])
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

        plt.figure(figsize=(8, 7))
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
        plt.xlabel("$x_O$", fontsize=20)
        plt.ylabel("$\phi_O$ (g cm$^{-2}$ s$^{-1}$)", fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
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
            "sub-Neptunes (90% H2O)": "mediumblue",
        }
        
        plt.figure(figsize=(8, 10))

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
            plt.axhline(bound, color=color_dict[ptype], linestyle='--', label=f"Mixing-limit: {int(f_mass*100)}% H$_2$O, {int(100-f_mass*100)}% H/He")

            # # after you compute `bound` for each ptype:
            # signif_half = 0.1 * bound
            # plt.axhline(signif_half, color=color_dict[ptype], linestyle=':', label=f"50% of bulk of {int(f_mass*100)}% WMF")

        # Axis scales and labels
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Hydrogen escape rate $\\dot N_H$ (atoms $s^{-1}$)", fontsize=20)
        plt.ylabel("Atomic escape flux ratio $\\dot N_O/\\dot N_H$", fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        # plt.legend(fontsize='small', loc='center left', bbox_to_anchor=(1.02, 0.5))
        # plt.subplots_adjust(right=0.78)
        plt.legend(fontsize='small', loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=3)
        plt.subplots_adjust(bottom=0.25, right=0.95)
        # plt.tight_layout()
        plt.show()

    #
    ### ----------- Water loss -----------
    #
        
    @staticmethod
    def water_loss(df):
        params = ModelParams()
        d = df.copy()
        d = df[df['planet_type'].str.contains('super-Earths')].copy()

        sec_per_year = 365.25 * 24 * 3600
        sec_per_Myr  = 1e6 * sec_per_year
        ocean_mass   = 1.4e24 # grams

        d['phiO_gs']        = d['phi_O'] * 4*np.pi*d['REUV']**2
        d['phiH2O_gs']      = d['phiO_gs'] * (18/16)
        d['EO_water_1Myr']  = d['phiH2O_gs'] * sec_per_Myr / ocean_mass
        d['EO_water_200Myr']= d['EO_water_1Myr'] * 200

        # check if depleted
        initial_water_mass = 0.03 * d['m_planet'] # in grams
        initial_water_EO = initial_water_mass / ocean_mass
        d['water_depleted'] = d['EO_water_200Myr'] >= initial_water_EO

        teq = d['Teq']
        cmap = colors.LinearSegmentedColormap.from_list('teq_cmap', ['maroon', 'chocolate'])
        norm = colors.Normalize(vmin=teq.min(), vmax=teq.max())
        teq_colors = cmap(norm(teq))

        masses = d['m_planet'] / params.mearth
        sizes  = np.sqrt(masses) * 25**1.2

        fig, ax = plt.subplots(figsize=(8,12))

        # shade region >1 EO/200 Myr
        # ymax = d['EO_water_200Myr'].max() * 1e10
        # ax.axhspan(1, ymax, color='#B3E5FC', alpha=0.3)

        # edgecols = ['red' if r=='RL' else 'black' for r in d['regime']]
        # # sc = ax.scatter(d['FEUV'], d['EO_water_200Myr'], s=sizes, c=teq_colors, edgecolors=edgecols, alpha=0.8, marker='o')
        # for i, row in d.iterrows():
        #     ax.scatter(row['FEUV'], row['EO_water_200Myr'], s=sizes[i], c=[teq_colors[i]], edgecolors=edgecols[i],
        #         marker='*' if row['water_depleted'] else 'o', alpha=0.8)

        # legend
        uniq_teq = sorted(d['Teq'].unique())

        # make a handle for each
        teq_handles = []
        for t in uniq_teq:
            col = cmap(norm(t))
            teq_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=col, markersize=15, label=f'{t:.0f} K'))

        depletion_handles = [Line2D([0], [0], marker='*', color='w', markerfacecolor='none', markeredgecolor='black', markersize=15, label='Water-depleted')]
        
        min_mass = masses.min()
        max_mass = masses.max()
        min_size = np.sqrt(min_mass) * 25**1.2
        max_size = np.sqrt(max_mass) * 25**1.2
        size_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=np.sqrt(min_size), label=f'{min_mass:.1f} M$\oplus$'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=np.sqrt(max_size), label=f'{max_mass:.1f} M$\oplus$')
        ]
        all_handles = teq_handles + depletion_handles + size_handles
        ax.legend(handles=all_handles, loc='upper left', fontsize=15, frameon=False, title='Legend', title_fontsize=15)
        
        # log–log formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('F$_{XUV}$ (erg cm$^{-2}$ s$^{-1}$)', fontsize=20)
        ax.set_ylabel('Water loss per 200 Myr (Earth Oceans)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)

        def to_per_myr(y):    return y/200
        def to_per_200myr(y): return y*200

        ax2 = ax.secondary_yaxis('right', functions=(to_per_myr, to_per_200myr))
        ax2.set_yscale('log')
        ax2.set_ylabel('Water loss per 1 Myr (Earth oceans)', fontsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=20)


        plt.tight_layout()
        plt.show()

    @staticmethod
    def metallicity_change(df, delta=0.02):
        params = ModelParams()
        mearth = params.mearth

        mask = (df['planet_type'].str.contains('sub-Neptunes') & df['planet_type'].str.contains(r'(10% H2O|20% H2O)'))
        d = df[mask].copy()

        sec_per_year     = 365.25 * 24 * 3600
        sec200           = 200e6 * sec_per_year # seconds in 200 Myr
        area             = 4 * np.pi * d['REUV']**2

        d['M_atm']       = 0.03 * d['m_planet']
        d['Z_init']      = d['planet_type'].str.extract(r'(10|20)% H2O')[0].astype(float)/100

        # breakdown
        d['M_H2O_init']  = d['Z_init'] * d['M_atm']
        d['M_O_init']    = d['M_H2O_init'] * (16/18)
        d['M_H_init']    = d['M_atm'] - d['M_O_init']

        # remnant after escape of 200 Myr
        d['M_O_rem']     = (d['M_O_init'] - d['phi_O']*area*sec200).clip(0) # make sure it doesn't go below 0
        d['M_H_rem']     = (d['M_H_init'] - d['phi_H']*area*sec200).clip(0) # make sure it doesn't go below 0

        # max possible H2O by each species
        d['M_H2O_from_O'] = d['M_O_rem'] * (18/16)
        d['M_H2O_from_H'] = d['M_H_rem'] * (18/2)

        # actual water formed depending on the limiting case
        d['M_H2O_final'] = np.minimum(d['M_H2O_from_O'], d['M_H2O_from_H'])

        # atoms consumed
        d['used_O']      = d['M_H2O_final'] * (16/18)
        d['used_H']      = d['M_H2O_final'] * (2/18)

        # leftovers
        d['leftover_O']  = d['M_O_rem'] - d['used_O']
        d['leftover_H']  = d['M_H_rem'] - d['used_H']

        # final atmosphere mass
        d['M_atm_final'] = d['M_H2O_final'] + d['leftover_O'] + d['leftover_H']

        # classify regimes
        d['water_regime'] = np.select(
            [
                # no H at all -> pure O
                d['leftover_H'] <= 0,                   # O is limiting -> enriched in H2O, ran out of H entirely?
                d['M_H2O_from_O'] <= d['M_H2O_from_H'], # otherwise H is limiting but some H remains, O limited recomb
            ],
            [
                'Oxygen-dominated (pure O)',     # Z_final -> 0
                'Water-enriched (excess H)',      # Z_final -> M_H2O_final/M_atm_final
            ],
            default = 'H_limited'       # Z_final -> 1 (cap), where d['M_H2O_from_O'] >= d['M_H2O_from_H']
        )

        # assign Z_final by our definition
        d['Z_final'] = np.select(
            [
                d['water_regime']=='Water-enriched (excess H)',
                d['water_regime']=='Oxygen-dominated (pure O)',
                d['water_regime']=='Water-enriched (excess O)'
            ],
            [
                d['M_H2O_final'] / d['M_atm_final'],  # H2O + H or H2
                0.0,                                  # no H2O possible, pure oxygen world with no H2O or H
                d['M_H2O_final'] / d['M_atm_final']   # H2O + O
            ]
        )

        d['has_extra_O'] = (d['water_regime']=='H_limited')  # leftover_O > 0 by construction
        d['H_depleted']  = (d['water_regime']=='H_depleted')

        # ΔZ
        d['dZ'] = d['Z_final'] - d['Z_init']
        keep = (abs(d['dZ']) > delta) | (d['water_regime']=='H_depleted')
        d = d.loc[keep]

        d['mass_em'] = np.round(d['m_planet']/mearth, 1)
        mass_vals    = np.sort(d['mass_em'].unique())
        n_mass       = len(mass_vals)

        cmap   = plt.cm.get_cmap('inferno', n_mass)

        regime_markers = {
            'Water-enriched (excess H)' : '^',    
            'Water-enriched (excess O)' : 's',    
            'Oxygen-dominated (pure O)': 'd'     
        }
        
        # for legend
        legend_elements = []
        for idx, mv in enumerate(mass_vals):
            color = cmap(idx)
            legend_elements.append(
                mlines.Line2D([], [], color=color, marker='^', linestyle='None', markersize=10, markeredgecolor='k', label=f"{mv:.1f} M$_\oplus$"))
        # for bins
        d['mass_idx'] = np.digitize(d['mass_em'], mass_vals, right=False) - 1
        # for contours
        levels = np.arange(-1, n_mass, 1)
        norm = mpl.colors.BoundaryNorm(levels, n_mass)
        
        fig, axes = plt.subplots(nrows=2, figsize=(6, 10), sharex=True, constrained_layout=True)
        fractions = ['10% H2O', '20% H2O']

        
        for i, (ax, frac) in enumerate(zip(axes, fractions)):
            sub = d[d['planet_type'].str.contains(frac)]
            if len(sub) < 1:
                continue
            
            # 1. Mask for triangulation/shading only
            Zmin = float(frac.split('%')[0]) / 100
            mask = sub['Z_final'] >= Zmin
            sub_shade = sub[mask]

            # 2. Triangulate and shade using only sub_shade
            tri = mtri.Triangulation(sub_shade['Z_final'], sub_shade['FEUV'])
            # use mass index for each point
            m_idx = sub_shade['mass_idx'].values
            # shaded regions
            ax.tricontourf(tri, m_idx, levels=levels, cmap=cmap, norm=norm, alpha=0.3)

            # 3. Scatter + arrows for all of sub, not masked
            for _, row in sub.iterrows():
                i = row['mass_idx']
                col = cmap(i)
                regime_colors = {'EL': 'k', 'RL': 'r'}
                edge_col = regime_colors.get(row['regime'], 'g')  # default to green if unknown

                # initial: open circle
                ax.scatter(row['Z_init'], row['FEUV'], marker='o', s=60, facecolors='none', edgecolors='k', linewidth=1, alpha=0.8)
                # final: filled triangle
                m = regime_markers[row['water_regime']]
                ax.scatter(row['Z_final'], row['FEUV'], marker=m, s=80, facecolors=col, edgecolors=edge_col, linewidth=1, alpha=0.6)
                # arrow from init→final
                ax.annotate('', xy=(row['Z_final'], row['FEUV']), xytext=(row['Z_init'], row['FEUV']),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.7, alpha=0.6), annotation_clip=False)

            # axis formatting
            ax.set_yscale('log')
            ax.set_ylim(d['FEUV'].min()*0.8, d['FEUV'].max()*1.2)
            ax.set_xlim(-0.05, 0.9)
            ax.tick_params(labelsize=20)
            ax.set_title(f"Initial {frac} sub-Neptunes", fontsize=15)

            present_regimes = d['water_regime'].unique()
            legend_regimes = [
                Line2D(
                    [0], [0],
                    marker=shape, color='k', linestyle='None',
                    markerfacecolor='w', markersize=10,
                    label=name.replace('_', '-')
                )
                for name, shape in regime_markers.items()
                if name in present_regimes
            ]

            # Reverse planet mass legend so largest is on top
            mass_legend_elements = [
                Line2D(
                    [0], [0],
                    color=cmap(i), marker='^', linestyle='None',
                    markeredgecolor='k', markersize=10,
                    label=f"{mv:.1f} M$_\oplus$"
                )
                for i, mv in enumerate(mass_vals)
            ][::-1]

        # Regime legend (upper right)
        fig.legend(
            handles=legend_regimes,
            title='Regime',
            loc='upper left',
            bbox_to_anchor=(1.02, 0.85),
            borderaxespad=0,
            fontsize=13,
            title_fontsize=14,
            frameon=True
        )

        # Mass legend (lower right)
        fig.legend(
            handles=mass_legend_elements,
            title='Planet mass',
            loc='upper left',
            bbox_to_anchor=(1.02, 0.55),
            borderaxespad=0,
            fontsize=13,
            title_fontsize=14,
            frameon=True
        )
        # shared labels
        axes[-1].set_xlabel('Metallicity Z (M$_{H_2O}$/M$_{atm}$)', fontsize=20)
        # fig.text(0.1, 0.75, 'F$_{XUV}$ (erg cm$^{-2}$ s$^{-1}$)', va='center', rotation='vertical', fontsize=20)

        # horizontal colorbar for mass
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        plt.show()

        # # ONLY LEGENDS
        # legend_regimes = [
        #     mlines.Line2D([0], [0], marker=shape, color='k', linestyle='None',
        #                 markerfacecolor='w', markersize=10, label=name)
        #     for name, shape in regime_markers.items()
        # ]
        # mass_legend_elements = [
        #     mlines.Line2D([0], [0], color=cmap(i), marker='^', linestyle='None',
        #                 markeredgecolor='k', markersize=10, label=f"{mv:.1f} M$_\oplus$")
        #     for i, mv in enumerate(mass_vals)
        # ][::-1]

        # initial_marker = Line2D(
        #     [0], [0],
        #     marker='o',
        #     color='k',                # edge color
        #     markerfacecolor='none',   # open circle
        #     markersize=8,
        #     linestyle='None',
        #     label='Initial metallicity'
        # )


        # # --- Create empty figure ---
        # fig, ax = plt.subplots(figsize=(6, 2))
        # ax.axis('off')  # Hide axes completely

        # # Plot legends — vertical, side-by-side
        # fig.legend(
        #     handles= [initial_marker]+ legend_regimes,
        #     loc='center left',
        #     bbox_to_anchor=(0.1, 0.5),
        #     frameon=True,
        #     fontsize=12,
        #     title_fontsize=13
        # )

        # fig.legend(
        #     handles=mass_legend_elements,
        #     title="Planet mass",
        #     loc='center right',
        #     bbox_to_anchor=(0.9, 0.5),
        #     frameon=True,
        #     fontsize=12,
        #     title_fontsize=13
        # )

        # plt.show()