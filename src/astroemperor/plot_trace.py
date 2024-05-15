def plot_trace2(sampler=None, eng_name='', my_model=None, options={}):
    if True:
        trace_modes = options['modes']
        saveplace = options['saveloc'] + '/plots/traces/'
        fmt = options['format']

        if trace_modes is None:
            trace_modes = [0]
        # 0:trace, 1:norm_post, 2:dens_interv, 3:corner
        
        trace_mode_dic = {0:'Trace Plot',
                        1:'Normalised Posterior',
                        2:'Density Interval',
                        3:'Corner Plot'}
        
        cor = ['C0', 'C1', 'C2', 'C4', 'C5', 'C7', 'C8', 'C9']
        colors_ = np.array([cor,cor,cor,cor,cor]).flatten()
        vn = []
        for v in my_model.get_attr_param('name'):
            vn.extend(v)
        vn = np.array(vn)[my_model.C_]

        dothis = []
        for i in range(4):
            dothis.append(i in trace_modes)

        pbar = tqdm(total=len(my_model.bloques_model))

    if True:
        if eng_name == 'reddemcee':
            import arviz as az
            arviz_data = az.from_emcee(sampler=sampler,
                                        var_names=vn)
            #arviz_data.sel(draw=slice(100, None))

            for b in my_model:
                if b.ndim_ == 0:
                    break
                
                vn_b = np.array(b.get_attr('name'))[b.C_]
                circ_var_names = vn_b[np.array(b.get_attr('is_circular'))[b.C_]]

                if False:
                    savefigname = saveplace + f'{trace_mode_dic[0]} {b.name_}.{fmt}'

                    az.plot_trace(arviz_data,
                                  figsize=(14, len(vn_b)*2.5),
                                  var_names=vn_b,
                                  circ_var_names=circ_var_names,
                                  plot_kwargs={'color':rc.fg},
                                  trace_kwargs={'color':rc.fg})

                    #pl.subplots_adjust(hspace=0.60)
                    pl.tight_layout()
                    pl.savefig(savefigname)
                    pl.close()

                if dothis[2]:

                    savefigname = saveplace + f'{trace_mode_dic[2]} {b.name_}.{fmt}'
                    axes = az.plot_density(
                            [arviz_data],
                            var_names=vn_b,
                            shade=0.2,
                            colors=colors_[b.bnumber_-1],
                            #hdi_markers='v'
                            )
                    fig = axes.flatten()[0].get_figure()
                    fig.suptitle("94% High Density Intervals")
                    pl.savefig(savefigname)
                    pl.close()

                if dothis[3]:
                    #pbar = tqdm(total=1)
                    cmap = mk_cmap([colors_[b.bnumber_-1]], ncolors=100)
                    savefigname = saveplace + f'{trace_mode_dic[3]} {b.name_}.{fmt}'
                    az.plot_pair(arviz_data,
                                 var_names=vn_b,
                                 figsize=(3*len(vn_b), 3*len(vn_b)),

                                kind=['kde',
                                    'scatter',
                                    ],

                                marginals=True,  # plot diagonals/histo
                                marginal_kwargs={'color':colors_[b.number_-1],
                                                 'quantiles':[.159, .5, .841],
                                                 #'plot_kwargs':{'lc':rc.fg,
                                                 #               'lw':2,
                                                 #               },
                                                 },

                                kde_kwargs={'contour_kwargs':{'lw':2,
                                                              #'cmap':cmap,
                                                              },
                                            'contourf_kwargs':{},
                                            #'color':colors_[b.number_-1],

                                            },

                                scatter_kwargs={'color':colors_[b.number_-1],
                                                's':10,
                                                'alpha':0.5,
                                                },  # points



                                #point_estimate="median",
                                #point_estimate_kwargs={'color':'red',  # line
                                #                    'alpha':0.75,
                                #                    },
                                #point_estimate_marker_kwargs={'color':'red',  # point
                                #                            's':200,
                                #                            'alpha':0.75},
                                )

                    #pl.subplots_adjust(hspace=0)
                    pl.tight_layout()
                    pl.savefig(savefigname)
                    pl.close()
                    #pbar.update(1)
                    #pbar.close()

                if dothis[1]:
                    for p in b[b.C_]:
                        fig, ax = pl.subplots(1, 1)
                        savefigname = saveplace + f'{trace_mode_dic[1]} {p.name}.{fmt}'
                        fig.suptitle(p.name)

                        az.plot_dist(arviz_data.posterior[p.name].values,
                                    color=rc.fg,
                                    rug=True,
                                        #figsize=(8, 6),
                                    )
                        #pl.ylabel('Probability Density')                            
                        pl.xlabel('Value')
                        pl.savefig(savefigname)
                        pl.close()

                pbar.update(1)
            


        pbar.close()
        pl.close('all')

plot_trace2(sim.sampler[0], sim.engine__.__name__, sim.model, sim.plot_all_list[-1])
