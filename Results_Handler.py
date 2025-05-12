import numpy as np
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
            ax.text(0.02, 0.98, ptype, transform=ax.transAxes, va='top', ha='left', fontsize=10)
            ax.set_xscale('log')
            ax.set_yscale('log')

        # common axes labels with spacing
        fig.text(0.5, 0.02, "F$_{XUV}$ (erg/cm$^2$/s)", ha='center')
        fig.text(0.02, 0.5, "Mass loss rate (g/s)", va='center', rotation='vertical')

        # legend handles (no marker shapes for planet types)
        handles = []
        # Teq color patches
        for t in unique_Teq:
            handles.append(mpatches.Patch(color=teq_color[t], label=f"{t} K"))
        # regime markers
        if show_regime:
            for r, ec in edge_map.items():
                handles.append(mlines.Line2D([], [], marker='o', linestyle='None', markerfacecolor='none',
                                             markeredgecolor=ec, markersize=8, label=f"{r} Regime"))

        # adjust margins to fit labels/legend
        plt.subplots_adjust(left=0.1, bottom=0.1, top=0.95, right=0.95, hspace=0.1, wspace=0.1)
        
        fig.legend(
            handles=handles,
            loc='lower center',
            ncol=len(handles),
            fontsize='8',
            frameon=False,
            bbox_to_anchor=(0.5, -0.07)
        )
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
        df = (df_combined
            .dropna(subset=['FEUV','r_planet','x_O','regime','planet_type','Teq'])
            .copy())
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
            cf = ax.contourf(
                X, Y, Z,
                levels=levels,
                cmap=cmap,
                norm=norm,
                extend='min', # “under” triangle for anything < clamp
                alpha=0.7
            )

            # scatter: log10 then clip at clamp
            for (_, marker), grp in sub.groupby(['regime','marker']):
                vals = np.log10(grp['x_O'].values)
                vals = np.clip(vals, clamp, None)
                ax.scatter(
                    grp['FEUV'], grp['R_earth'],
                    c=vals,
                    cmap=cmap,
                    norm=norm,
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
            Line2D([0],[0], marker='o', color='w', label='Low T$_{eq}$ (≤400 K)',
                markerfacecolor='gray', markeredgecolor='k', markersize=8),
            Line2D([0],[0], marker='s', color='w', label='High T$_{eq}$ (>400 K)',
                markerfacecolor='gray', markeredgecolor='k', markersize=8),
            Line2D([0],[0], marker='o', color='w', label='EL regime',
                markerfacecolor='white', markeredgecolor='k', markersize=8),
            Line2D([0],[0], marker='o', color='w', label='RL regime',
                markerfacecolor='white', markeredgecolor='red', markersize=8)
        ]
        fig.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=9)

        # tighten subplots within left 80%
        fig.tight_layout(rect=[0, 0.05, 0.80, 1.0])
        plt.show()

    @staticmethod
    def mass_FEUV_oxygen_contour_by_planet_type(df_combined):
        '''Mass vs FEUV plots with x_O clamped below −20 in colorbar.'''
        params = ModelParams()
        mearth = params.mearth

        # clean & normalize
        df = (df_combined
              .dropna(subset=['FEUV','r_planet','m_planet','x_O','regime','planet_type','Teq'])
              .copy())
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
            cf = ax.contourf(X, Y, Z,
                             levels=levels,
                             cmap=cmap,
                             norm=norm,
                             extend='min',
                             alpha=0.7)

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
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='gray', markeredgecolor='k', markersize=8,
                   label='Low T$_{eq}$ (≤500 K)'),
            Line2D([0],[0], marker='s', color='w',
                   markerfacecolor='gray', markeredgecolor='k', markersize=8,
                   label='High T$_{eq}$ (>500 K)'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='k', markersize=8,
                   label='EL regime'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='red', markersize=8,
                   label='RL regime'),
        ]
        fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=9)
        fig.tight_layout(rect=[0,0.05,0.80,1.0])
        plt.show()


    @staticmethod
    def radius_FEUV_oxygen_contour(df_combined):
        '''Radius vs FEUV 2x2 grid with x_O clamped below -10.'''
        params = ModelParams()
        rearth = params.rearth

        df = (df_combined
              .dropna(subset=['FEUV','r_planet','x_O','regime','planet_type','Teq'])
              .copy())
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
            ("Super-Earths",   df['planet_type'].str.contains('super-Earths'), True),
            ("Sub-Neptunes",   df['planet_type'].str.contains('sub-Neptunes'), True),
            ("Super-Earths",   df['planet_type'].str.contains('super-Earths'), False),
            ("Sub-Neptunes",   df['planet_type'].str.contains('sub-Neptunes'), False),
        ]

        fig, axes = plt.subplots(2,2, figsize=(12,8), sharex=True, sharey=True)
        axes = axes.flatten()

        for ax, (title, mask, do_scatter) in zip(axes, panels):
            sub = df[mask]
            if sub.empty:
                ax.set_title(title); ax.axis('off'); continue

            Z_raw = griddata((sub['FEUV'], sub['R_earth']), sub['x_O'], (X, Y), method='linear')
            Z_raw = np.ma.masked_invalid(Z_raw)
            Z = np.log10(Z_raw)

            cf = ax.contourf(X, Y, Z,
                             levels=levels,
                             cmap=cmap,
                             norm=norm,
                             extend='min',
                             alpha=0.8)

            if do_scatter:
                for (_,marker), grp in sub.groupby(['regime','marker']):
                    vals = np.log10(grp['x_O'].values)
                    vals = np.clip(vals, clamp, None)
                    ax.scatter(
                        grp['FEUV'], grp['R_earth'],
                        c=vals, cmap=cmap, norm=norm,
                        s=(np.sqrt(grp['M_earth']))*20 if 'M_earth' in grp else 20,
                        marker=marker, edgecolor=grp['ecolor'].iloc[0],
                        linewidth=0.8, alpha=0.5
                    )

            ax.set_xscale('log')
            ax.set_title(title, fontsize=12)

        # labels & colorbar
        axes[2].set_xlabel("F$_{XUV}$", fontsize=11)
        axes[3].set_xlabel("F$_{XUV}$", fontsize=11)
        axes[0].set_ylabel("R$_\oplus$", fontsize=11)
        axes[2].set_ylabel("R$_\oplus$", fontsize=11)

        fig.subplots_adjust(right=0.80)
        cax = fig.add_axes([0.82,0.15,0.02,0.7])
        cbar = fig.colorbar(cf, cax=cax, extend='min')
        cbar.set_label('log$_{10}$ x$_O$', fontsize=11)
        ticks = np.linspace(clamp, 0, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.0f}" for t in ticks])

        # legend
        handles = [
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='gray', markeredgecolor='k', markersize=8,
                   label='Low T$_{eq}$'),
            Line2D([0],[0], marker='s', color='w',
                   markerfacecolor='gray', markeredgecolor='k', markersize=8,
                   label='High T$_{eq}$'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='k', markersize=8,
                   label='EL regime'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='red', markersize=8,
                   label='RL regime'),
        ]
        fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=10)
        fig.tight_layout(rect=[0,0.05,0.80,1.0])
        plt.show()


    @staticmethod
    def mass_FEUV_oxygen_contour(df_combined):
        '''Mass vs FEUV 2x2 grid with x_O clamped below -10.'''
        params = ModelParams()
        mearth = params.mearth

        df = (df_combined
              .dropna(subset=['FEUV','m_planet','x_O','regime','planet_type','Teq'])
              .copy())
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
            ("Super-Earths", df['planet_type'].str.contains('super-Earths')),
            ("Sub-Neptunes", df['planet_type'].str.contains('sub-Neptunes'))
        ]
        panels = [
            (masks[0][0], masks[0][1], True),
            (masks[1][0], masks[1][1], True),
            (masks[0][0], masks[0][1], False),
            (masks[1][0], masks[1][1], False),
        ]

        fig, axes = plt.subplots(2,2, figsize=(12,8), sharex=True, sharey=True)
        axes = axes.flatten()

        for ax, (title, mask, do_scatter) in zip(axes, panels):
            sub = df[mask]
            if sub.empty:
                ax.set_title(title); ax.axis('off'); continue

            Z_raw = griddata((sub['FEUV'], sub['M_earth']), sub['x_O'], (X, Y), method='linear')
            Z_raw = np.ma.masked_invalid(Z_raw)
            Z = np.log10(Z_raw)

            cf = ax.contourf(X, Y, Z,
                             levels=levels,
                             cmap=cmap,
                             norm=norm,
                             extend='min',
                             alpha=0.8)

            if do_scatter:
                for (_,marker), grp in sub.groupby(['regime','marker']):
                    vals = np.log10(grp['x_O'].values)
                    vals = np.clip(vals, clamp, None)
                    ax.scatter(
                        grp['FEUV'], grp['M_earth'],
                        c=vals, cmap=cmap, norm=norm,
                        marker=marker, edgecolor=grp['ecolor'].iloc[0],
                        linewidth=0.8, alpha=0.5
                    )

            ax.set_xscale('log')
            ax.set_title(title, fontsize=12)

        fig.subplots_adjust(right=0.80)
        cax = fig.add_axes([0.82,0.15,0.02,0.7])
        cbar = fig.colorbar(cf, cax=cax, extend='min')
        cbar.set_label('log$_{10}$ x$_O$', fontsize=11)
        ticks = np.linspace(clamp, 0, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.0f}" for t in ticks])

        handles = [
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='gray', markeredgecolor='k', markersize=8,
                   label='Low T$_{eq}$'),
            Line2D([0],[0], marker='s', color='w',
                   markerfacecolor='gray', markeredgecolor='k', markersize=8,
                   label='High T$_{eq}$'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='k', markersize=8,
                   label='EL regime'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='red', markersize=8,
                   label='RL regime'),
        ]
        fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=9)
        fig.tight_layout(rect=[0,0.05,0.80,1.0])
        plt.show()

    @staticmethod
    def radius_mass_FEUV_oxygen_contour(df_combined):
        '''R+M vs FEUV quartet with x_O clamped below -10.'''
        params = ModelParams()
        rearth, mearth = params.rearth, params.mearth

        df = (df_combined
              .dropna(subset=['FEUV','r_planet','m_planet','x_O','planet_type','Teq','regime'])
              .copy())
        df['R_earth'] = df['r_planet'] / rearth
        df['M_earth'] = df['m_planet'] / mearth

        clamp = -10
        cmap  = plt.get_cmap('coolwarm'); cmap.set_under(cmap(0))
        norm  = colors.Normalize(vmin=clamp, vmax=0)
        levels = np.linspace(clamp, 0, 13)

        xi      = np.logspace(np.log10(df['FEUV'].min()), np.log10(df['FEUV'].max()), 200)
        yi_r    = np.linspace(df['R_earth'].min(), df['R_earth'].max(), 200)
        yi_m    = np.linspace(df['M_earth'].min(), df['M_earth'].max(), 200)
        Xr, Yr  = np.meshgrid(xi, yi_r)
        Xm, Ym  = np.meshgrid(xi, yi_m)

        panels = [
            ("Super-Earths",   df['planet_type'].str.contains('super-Earths'),  True),
            ("Sub-Neptunes",   df['planet_type'].str.contains('sub-Neptunes'),  True),
            ("Super-Earths",   df['planet_type'].str.contains('super-Earths'),  False),
            ("Sub-Neptunes",   df['planet_type'].str.contains('sub-Neptunes'),  False),
        ]

        fig, axes = plt.subplots(2,2, figsize=(12,8), sharex=True)
        axes = axes.flatten()

        for idx, (ax, (title, mask, is_radius)) in enumerate(zip(axes, panels)):
            sub = df[mask]
            if sub.empty:
                ax.axis('off')
                continue

            # pick grid and var name
            if is_radius:
                X, Y = Xr, Yr
                var = 'R_earth'
            else:
                X, Y = Xm, Ym
                var = 'M_earth'

            Z_raw = griddata((sub['FEUV'], sub[var]), sub['x_O'], (X, Y), method='linear')
            Z_raw = np.ma.masked_invalid(Z_raw)
            Z = np.log10(Z_raw)

            cf = ax.contourf(
                X, Y, Z,
                levels=levels,
                cmap=cmap,
                norm=norm,
                extend='min',
                alpha=0.7
            )

            ax.set_xscale('log')
            ax.set_title(title, fontsize=12)

        axes[2].set_xlabel("F$_{XUV}$", fontsize=11)
        axes[3].set_xlabel("F$_{XUV}$", fontsize=11)
        axes[0].set_ylabel("R$_\oplus$", fontsize=11)
        axes[2].set_ylabel("M$_\oplus$", fontsize=11)

        # shared colorbar
        fig.subplots_adjust(right=0.80)
        cax = fig.add_axes([0.82,0.15,0.02,0.7])
        cbar = fig.colorbar(cf, cax=cax, extend='min')
        cbar.set_label('log$_{10}$ x$_O$', fontsize=11)
        ticks = np.linspace(clamp, 0, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.0f}" for t in ticks])

        plt.tight_layout(rect=[0,0.05,0.80,1.0])
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
            plt.scatter(subset['x_O'], subset['phi_O'],
                label=ptype + slope_text,
                # marker=marker_dict[ptype],
                marker='o',
                color=color_dict[ptype],
                alpha=0.8,
                edgecolor='k'
            )

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Oxygen Fractionation Factor")
        plt.ylabel("Oxygen Escape Flux (g cm$^{-2}$ s$^{-1}$)")
        plt.legend(loc='best')
        plt.tight_layout()
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

            ax.set_xlim(1e-3, 1.5e0)
            ax.set_ylim(1e-15, 1e-7)

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