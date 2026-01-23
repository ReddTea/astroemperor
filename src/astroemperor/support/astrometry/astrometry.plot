# PLOT
pl.rcParams["font.family"] = "monospace"
cus_cmap = plc.LinearSegmentedColormap.from_list(f'cus_cmap', ['C1', 'C0'], N=N_GOST)


fig = pl.figure(figsize=figsize_)

gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1, 1],
            height_ratios=[1, 1], figure=fig)

# Subplots
ax_a = fig.add_subplot(gs[:, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 1])

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.1, hspace=0.30)

# plot A
if True:
    ax_a.plot(fdra, fdde, 'k-', linewidth=line_width, zorder=1)  # best fit
    ax_b.plot(fdra, fdde, 'k-', linewidth=line_width, zorder=1)  # best fit

    ax_a.plot(0,0, marker='+', color='k', mew=2, ms=12, zorder=3,
              label='Barycentre', ls='')  # barycenter
    # nodes
    ax_a.plot(nodes[0, 2:], nodes[1, 2:], color='gray',ls=':',linewidth = 1)
    ax_a.plot([0, nodes[0, 1]], [0, nodes[1, 1]], color='gray', ls='-')

    # this is the gaia segment

    colors_gost = cus_cmap(Normalize(vmin=min(time_iad_gost),
                                     vmax=max(time_iad_gost))(time_iad_gost))


    ax_a.scatter(dra_g, dec_g, marker='o',
                 s=30,
                 facecolors='none',
                 edgecolors=colors_gost,
                 label='GOST model',
                 lw=1.5
                 )


    dra_hb_mod, dde_hb_mod = ca_hipp[0], ca_hipp[1]
    nhipp_bin = len(dra_hb_mod)
    dplx_hb, dpmra_hb_mod, dpmde_hb_mod = np.zeros(nhipp_bin), np.zeros(nhipp_bin), np.zeros(nhipp_bin)

    dra_hb_mod = dra_hb_mod + dpmra_hb_mod * df_hipp_b['EPOCH']
    dde_hb_mod = dde_hb_mod + dpmde_hb_mod * df_hipp_b['EPOCH']

    dra_hb_mod = dra_hb_mod + dpmra_hb_mod * df_hipp_b['EPOCH']
    dde_hb_mod = dde_hb_mod + dpmde_hb_mod * df_hipp_b['EPOCH']

    dastro = AstroDiff(AM_catalogs_[0, 1:], bary[0, :])
    dra_hb = -dastro[0]
    dde_hb = -dastro[1]
    #dplx_hb = -dastro[2]
    dpmra_hb = -dastro[3]
    dpmde_hb = -dastro[4]

if True:
    ax_a.scatter(dra_hb_mod,
                    dde_hb_mod,
                    c=df_hipp_b['BJD'].values,
                    cmap='plasma',
                    marker='o',
                    s=scatter_size,
                    label='Hipp IAD',
                    zorder=2)   

if True:
    x_hipp = -dastro[0]
    y_hipp = -dastro[1]

    x_mu_hipp = -dastro[3]
    y_mu_hipp = -dastro[4]

    ax_a.plot(x_hipp, y_hipp, 'C2D',
                zorder=0, mec='k', mew=1.0)

    x_vec = x_hipp - x_mu_hipp, x_hipp + x_mu_hipp
    y_vec = y_hipp - y_mu_hipp, y_hipp + y_mu_hipp
    pos = np.array([x_hipp, y_hipp])           # [Δα*, Δδ] in mas
    mu  = np.array([x_mu_hipp, y_mu_hipp])     # [Δμ_α*, Δμ_δ] in mas/yr
    mycov = AM_gaia_cov_[-3]

    Sigma = mycov[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])]
    S_pp  = Sigma[:2, :2]
    S_pmu = Sigma[:2, 2:]
    S_mup = Sigma[2:, :2]
    S_mumu= Sigma[2:, 2:]

    smin, smax = -1.0, 1.0
    poly = band_polygon(pos, mu, S_pp, S_pmu, S_mup, S_mumu,
                        smin=smin, smax=smax, n=250, Nsigma=1.0)
    
    ax_a.add_patch(Polygon(poly, closed=True,
                            facecolor='C2', alpha=0.5,
                            edgecolor='none', zorder=0,
                            #label='Hipp Data',
                            ))

    shade_proxy0   = Patch(facecolor='C2', alpha=0.5, edgecolor='none')
    diamond_proxy0 = Line2D([], [], linestyle='None', marker='D',
                        markerfacecolor='C2', markeredgecolor='k',
                        markeredgewidth=1.0)
    HIPP_data_proxy = [(shade_proxy0, diamond_proxy0)]


    if 'GDR3' in AM_cats_:
        ax_a.plot(fdrag[-1], fddeg[-1],
                  marker='o', color='C0',
                  mec='k', mew=1.0,
                  markersize=marker_size, linewidth=line_width, zorder=3)

    if 'GDR2' in AM_cats_:
        ax_a.plot(fdrag[-2], fddeg[-2],
                  marker='o', color='C1',
                  mec='k', mew=1.0,
                  markersize=marker_size, linewidth=line_width, zorder=3)

    ax_a.set_xlabel(r'$\Delta \alpha_{*}$ [mas]', fontsize=label_fontsize, labelpad=8)
    ax_a.set_ylabel(r'$\Delta \delta$ [mas]', fontsize=label_fontsize, labelpad=8)


    ax_a.text(0.05, 0.95, '(a)', transform=ax_a.transAxes, fontsize=title_fontsize,
            fontweight='bold', va='top', ha='left')

    # Apply tick formatting to (a)
    ax_a.tick_params(axis='both', which='major',
                    length=major_tick_length, width=line_width,
                    labelsize=tick_labelsize, direction='in')
    ax_a.tick_params(axis='both', which='minor',
                    length=minor_tick_length, width=line_width,
                    direction='in')
    ax_a.minorticks_on()
    ax_a.xaxis.set_major_locator(MaxNLocator(5))
    #ax_a.xaxis.set_major_locator(FixedLocator(np.linspace(-1.2, 1.2, 4)))
    ax_a.invert_xaxis()

    #ax_a.set_xlim()  # TODO: smarter lims
    #ax_a.set_ylim()

# plot B
if True:
    # gost model
    ax_b.scatter(dra_g, dec_g, marker='o',
                 s=30,
                 facecolors='none',
                 edgecolors=colors_gost,
                 #label='GOST model',
                 lw=1.5
                 )
    
    if 'GDR3' in AM_cats_:
        fdrags_vec3 = fdrag[-1] - go[3, -1], fdrag[-1] + go[3, -1]
        fddegs_vec3 = fddeg[-1] - go[4, -1], fddeg[-1] + go[4, -1]

        ax_b.plot(fdrags_vec3, fddegs_vec3,
                '-', color='C0',
                alpha=0.75,
                zorder=1,
                linewidth=line_width)
    
        ax_b.plot(fdrag[-1], fddeg[-1],
                  'o', color='C0',
                  markersize=marker_size, linewidth=line_width+1,
                  zorder=2,
                  mec='k', mew=1.0,
                  label='GDR3 Model')

        dastro = AstroDiff(AM_catalogs_[-1, 1:], bary[-1, :])

        x_gdr3 = -dastro[0]
        y_gdr3 = -dastro[1]

        x_mu_gdr3 = -dastro[3]
        y_mu_gdr3 = -dastro[4]

                # Create an ellipse to represent proper motion uncertainty
        ax_b.plot(x_gdr3, y_gdr3, 'C0D')
        ax_a.plot(x_gdr3, y_gdr3, 'C0D')

        x_vec = x_gdr3 - x_mu_gdr3, x_gdr3 + x_mu_gdr3
        y_vec = y_gdr3 - y_mu_gdr3, y_gdr3 + y_mu_gdr3
        pos = np.array([x_gdr3, y_gdr3])           # [Δα*, Δδ] in mas
        mu  = np.array([x_mu_gdr3, y_mu_gdr3])     # [Δμ_α*, Δμ_δ] in mas/yr
        mycov = AM_gaia_cov_[-1]

        Sigma = mycov[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])]
        S_pp  = Sigma[:2, :2]
        S_pmu = Sigma[:2, 2:]
        S_mup = Sigma[2:, :2]
        S_mumu= Sigma[2:, 2:]

        smin, smax = -1.0, 1.0
        poly = band_polygon(pos, mu, S_pp, S_pmu, S_mup, S_mumu,
                            smin=smin, smax=smax, n=250, Nsigma=1.0)

        ax_b.add_patch(Polygon(poly, closed=True,
                               facecolor='C0', alpha=0.5,
                               edgecolor='none', zorder=0,
                               #label='GDR3 Data',
                               label='_nolegend_',
                               ))
        
        ax_a.add_patch(Polygon(poly, closed=True,
                               facecolor='C0', alpha=0.5,
                               edgecolor='none', zorder=0,
                               ))

        shade_proxy   = Patch(facecolor='C0', alpha=0.5, edgecolor='none')
        diamond_proxy = Line2D([], [], linestyle='None', marker='D',
                            markerfacecolor='C0', markeredgecolor='k',
                            markeredgewidth=1.0)
        GDR3_data_proxy = [(shade_proxy, diamond_proxy)]

    if 'GDR2' in AM_cats_:
        fdrags_vec2 = fdrag[-2] - go[3, -2], fdrag[-2] + go[3, -2]
        fddegs_vec2 = fddeg[-2] - go[4, -2], fddeg[-2] + go[4, -2]
        
        ax_b.plot(fdrags_vec2, fddegs_vec2,
                '-', color='C1',
                alpha=0.75,
                zorder=1,
                linewidth=line_width)

        ax_b.plot(fdrag[-2], fddeg[-2],
                'o', color='C1',
                markersize=marker_size, linewidth=line_width+1,
                zorder=2,
                mec='k', mew=1.0,
                label='GDR2 Model')
        

        dastro = AstroDiff(AM_catalogs_[-2, 1:], bary[-2, :])

        x_gdr2 = -dastro[0]
        y_gdr2 = -dastro[1]

        x_mu_gdr2 = -dastro[3]
        y_mu_gdr2 = -dastro[4]

                # Create an ellipse to represent proper motion uncertainty
        ax_b.plot(x_gdr2, y_gdr2)
        ax_a.plot(x_gdr2, y_gdr2)

        x_vec = x_gdr2 - x_mu_gdr2, x_gdr2 + x_mu_gdr2
        y_vec = y_gdr2 - y_mu_gdr2, y_gdr2 + y_mu_gdr2
        pos = np.array([x_gdr2, y_gdr2])           # [Δα*, Δδ] in mas
        mu  = np.array([x_mu_gdr2, y_mu_gdr2])     # [Δμ_α*, Δμ_δ] in mas/yr
        mycov = AM_gaia_cov_[-2]

        Sigma = mycov[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])]
        S_pp  = Sigma[:2, :2]
        S_pmu = Sigma[:2, 2:]
        S_mup = Sigma[2:, :2]
        S_mumu= Sigma[2:, 2:]

        smin, smax = -1.0, 1.0
        poly = band_polygon(pos, mu, S_pp, S_pmu, S_mup, S_mumu,
                            smin=smin, smax=smax, n=250, Nsigma=1.0)

        ax_b.add_patch(Polygon(poly, closed=True,
                               facecolor='C1', alpha=0.5,
                               edgecolor='none', zorder=0,
                               #label='GDR2 Data',
                               ))
        
        ax_a.add_patch(Polygon(poly, closed=True,
                               facecolor='C1', alpha=0.5,
                               edgecolor='none', zorder=0,
                               #label='GDR2 Catalogue',
                               ))
        
        shade_proxy2   = Patch(facecolor='C1', alpha=0.5, edgecolor='none')
        diamond_proxy2 = Line2D([], [], linestyle='None', marker='D',
                            markerfacecolor='C1', markeredgecolor='k',
                            markeredgewidth=1.0)
        GDR2_data_proxy = [(shade_proxy2, diamond_proxy2)]

# Legend A
if True:
    # Add a legend with these handles
    handles, labels = ax_a.get_legend_handles_labels()
    handles += HIPP_data_proxy
    labels += ['Hipp Data']


    # Define the new desired order (by index)
    new_order = [0, 2, 1, 3]  # Reorder as: third, first, second

    # Re-order handles and labels
    handles = [handles[i] for i in new_order]
    labels = [labels[i] for i in new_order]

    # Update the legend with the new order
    fig.subplots_adjust(top=0.9)  # Add space above the plots
    ax_a.legend(
        handles=handles,
        labels=labels, 
        loc='upper center',
        bbox_to_anchor=(0.5, 1.12),  # (0.5, 1.15)
        ncol=2,
        fontsize=legend_fs,
        frameon=False,
        labelspacing=0.25,
        columnspacing=1.5,
        scatterpoints=3,)

# Legend B
if True:
    handles, labels = ax_b.get_legend_handles_labels()
    handles += GDR3_data_proxy
    handles += GDR2_data_proxy
    labels += ['GDR3 Data']
    labels += ['GDR2 Data']

    # Define the new desired order (by index)
    new_order = [1, 0, 3, 2]  # Reorder as: third, first, second

    # Re-order handles and labels
    handles = [handles[i] for i in new_order]
    labels = [labels[i] for i in new_order]

    fig.subplots_adjust(top=0.9)  # Add space above the plots
    ax_b.legend(
        handles=handles,
        labels=labels, 
        loc='upper center',
        bbox_to_anchor=(0.5, 1.28),  # (0.5, 1.32)
        ncol=2,
        fontsize=legend_fs,
        frameon=False,
        labelspacing=0.25,
        columnspacing=1.5)


# Plot box

if True:
    contained_values_ra = np.concatenate([dra_g,
                                             fdrag[1:],
                                             np.array(fdrags_vec3),
                                             np.array(fdrags_vec2),
                                             np.array([x_gdr2+x_mu_gdr2,
                                             x_gdr2-x_mu_gdr2,
                                             x_gdr3+x_mu_gdr3,
                                             x_gdr3-x_mu_gdr3]),
                                             ])
    contained_values_de = np.concatenate([dec_g,
                                            fddeg[1:],
                                            np.array(fddegs_vec3),
                                            np.array(fddegs_vec2),
                                            np.array([y_gdr2+y_mu_gdr2,
                                            y_gdr2-y_mu_gdr2,
                                            y_gdr3+y_mu_gdr3,
                                            y_gdr3-y_mu_gdr3]),
                                            ])

    min_x_box, max_x_box = min(np.floor(contained_values_ra/ceil_step)*ceil_step), max(np.ceil(contained_values_ra/ceil_step)*ceil_step)
    min_y_box, max_y_box = min(np.floor(contained_values_de/ceil_step)*ceil_step), max(np.ceil(contained_values_de/ceil_step)*ceil_step)
    
    ax_a.plot([min_x_box, max_x_box, max_x_box, min_x_box, min_x_box],
              [min_y_box, min_y_box, max_y_box, max_y_box, min_y_box],
              'k--', alpha=0.5, lw=1)

    ax_b.set_xlabel(r'$\Delta \alpha_{*}$ [mas]', fontsize=label_fontsize, labelpad=8)
    ax_b.set_ylabel(r'$\Delta \delta$ [mas]', fontsize=label_fontsize, labelpad=8)
    ax_b.set_xlim(min_x_box, max_x_box)
    ax_b.set_ylim(min_y_box, max_y_box)
    ax_b.text(0.05, 0.95, '(b) Gaia',
              transform=ax_b.transAxes, fontsize=title_fontsize,
            fontweight='bold', va='top', ha='left')

    # Apply tick formatting to (b)
    ax_b.tick_params(axis='both', which='major', length=major_tick_length, width=line_width, labelsize=tick_labelsize, direction='in')
    ax_b.tick_params(axis='both', which='minor', length=minor_tick_length, width=line_width, direction='in')
    ax_b.minorticks_on()

    ax_b.xaxis.set_major_locator(MaxNLocator(3))
    ax_b.invert_xaxis()


# Plot C
if True:
    t_shadow = AstroTime(data_iad_hipp['BJD'], format='jd').to_value('decimalyear')
    y_shadow = data_iad_hipp['RES'].values
    ye_shadow = data_iad_hipp['SRES'].values

    ax_c.errorbar(t_shadow,
                  y_shadow, 
                  ye_shadow,
                  fmt='o',
                  color='gray',
                  alpha=0.3)
    
    t_yr = AstroTime(df_hipp_b['BJD'], format='jd').to_value('decimalyear')
    t_yr_tickers = np.unique([int(t) for t in t_yr])
    
    # Define colormap and normalizer
    cmap = plasma
    norm = Normalize(vmin=np.min(t_yr), vmax=np.max(t_yr))
    for i in range(len(t_yr)):
        ax_c.errorbar(t_yr[i],
                      df_hipp_b['RES'].values[i],
                      yerr=df_hipp_b['SRES'].values[i],
                      fmt='o',
                      color=cmap(norm(t_yr[i])))


# adjust and save plot
if True:
    ax_c.axhline(0, color='gray')

    ax_c.set_xlabel('Epoch [year]', fontsize=label_fontsize, labelpad=8)
    ax_c.set_ylabel('O - C [mas]', fontsize=label_fontsize, labelpad=8)
    #ax_c.set_xlim(1989, 1994)
    ax_c.set_ylim(-16, 16)
    ax_c.text(0.05, 0.95, '(c) Hipparcos',
              transform=ax_c.transAxes, fontsize=title_fontsize,
              fontweight='bold', va='top', ha='left')

    # Apply tick formatting to (c)
    ax_c.tick_params(axis='both', which='major', length=major_tick_length, width=line_width, labelsize=tick_labelsize, direction='in')
    ax_c.tick_params(axis='both', which='minor', length=minor_tick_length, width=line_width, direction='in')
    
    #ax_c.tick_params(axis='x', labelrotation=45)
    ax_c.xaxis.set_major_locator(FixedLocator(t_yr_tickers))
    ax_c.minorticks_on()

    for ax in [ax_b, ax_c]:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", right=True, left=False, pad=2)
        #ax.spines["left"].set_visible(False)
        #ax.spines["right"].set_visible(True)

    for ax in [ax_a, ax_b, ax_c]:
        for spine in ax.spines.values():
            spine.set_linewidth(fm_frame_lw)

    loc_x, loc_y = 0.08, 0.05 
    ax_a.plot(loc_x, loc_y, color='k',
              marker=r'$\circlearrowleft$',
              transform=ax_a.transAxes,
              markersize=24)
    #pl.show()
    pl.savefig(saveloc_)