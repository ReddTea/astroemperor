# type: ignore
# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# version 0.3.0
# date 29 nov 2022
# sourcery skip

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT

import numpy as np
import pandas as pd

import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from importlib import reload

from copy import deepcopy
import os
import logging

from tqdm import tqdm

from reddcolors import Palette

from .model_repo import *
from .unmodel_repo import *
from .block import ReddModel
from .utils import fold_dataframe, nullify_output, flatten, get_support


rc = Palette()


def plot_GM_Estimator(estimator, saveloc='', fmt='png', sig_factor=4, fill_cor=0, plot_name='', plot_title=None, plot_ylabel=None):
    # sourcery skip: use-fstring-for-formatting

    cor = ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    colors = np.array([cor,cor,cor,cor,cor]).flatten()

    if plot_title is None:
        plot_title = 'Optimal estimate with Gaussian Mixtures\n for '

    if plot_ylabel is None:
        plot_ylabel = 'Probability Density'
    ## COL
    sig_factor = sig_factor
    saveplace = saveloc + '/posteriors/GMEstimates/'
    plot_fmt = fmt
    plot_nm = plot_name


    n_components = estimator.n_components

    mu = estimator.mixture_mean
    var = estimator.mixture_variance
    sig = estimator.mixture_sigma

    xx, yy = [], []

    xticks = [mu]

    fig, ax = pl.subplots(figsize=(8, 4))
    for i in range(sig_factor):
        xx.append(np.array([np.linspace(mu-sig*(i+1),
                                        mu+sig*(i+1),
                                        1000)]).T)

        yy.append(np.exp(estimator.score_samples(xx[i])))

        xx[i] = np.append(np.append(xx[i][0], xx[i]), xx[i][-1])
        yy[i] = np.append(np.append(0, yy[i]), 0)
        ax.fill(xx[i], yy[i], c=colors[fill_cor], alpha=1/sig_factor, zorder=2*(i+1)-1)

        ax.vlines([xx[i][0], xx[i][-1]], ymin=[min(yy[i]), min(yy[i])],
                                         ymax=[yy[i][1], yy[i][-2]],
                                         colors=[rc.fg, rc.fg],
                                         lw=[1.5, 1.5],
                                         ls=['--'],
                                         zorder=2*(i+1))

        xticks.extend((xx[i][0], xx[i][-1]))
    ax.vlines(mu, ymin=[min(yy[0])],
                  ymax=[np.exp(estimator.score_samples([[mu]]))],
                  colors=[rc.fg],
                  lw=[2],
                  ls=['-'],
                  zorder=2*sig_factor)


    ax.plot(xx[-1], yy[-1], c=rc.fg, alpha=1, lw=2, zorder=9)

    mu_display = np.round(mu, 3)
    sig_display = np.round(sig, 3)

    # Dummy plots for labels
    ax.plot(
        np.median(xx[0]),
        np.median(yy[0]),
        alpha=0.0,
        label=f'$\mu = {mu_display}$',
    )
    ax.plot(np.median(xx[0]), np.median(yy[0]), alpha=0., label=r'$\sigma = {}$'.format(sig_display))
    if n_components > 1:
        ax.plot(np.median(xx[0]), np.median(yy[-1]), alpha=0., label=r'$N = {}$'.format(n_components))

    # Set ticks, labels, title, etc
    xticks = np.sort(xticks)

    dif = xticks[-1] - xticks[0]
    nround = 2
    for i in range(4):
        nround = i + 2
        if np.round(dif, i) // 10**-i > 0:
            break

    xticks = np.round(xticks, nround)
    yticks = np.round(np.linspace(0, max(yy[0]), 5), 2)

    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xticks(xticks, minor=False)
    ax.set_yticks(yticks, minor=False)
    ax.legend(framealpha=0.)
    ax.set_title(plot_title+'{}'.format(plot_nm))
    ax.set_xlabel('{} [{}]'.format(plot_nm, estimator.unit))
    ax.set_ylabel(plot_ylabel)


    pl.savefig(saveplace+'{}.{}'.format(plot_nm, plot_fmt),
               bbox_inches='tight')

    pl.close('all')


def plot_traces(sampler, eng_name, my_model, saveloc='', trace_modes=None, fmt='png'):
    if trace_modes is None:
        trace_modes = [0]
    # 0:trace, 1:norm_post, 2:dens_interv, 3:corner
    trace_mode_dic = {0:'Trace Plot',
                      1:'Normalised Posterior',
                      2:'Density Interval',
                      3:'Corner Plot'}
    saveplace = saveloc + '/traces/'

    cor = ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    colors_ = np.array([cor,cor,cor,cor,cor]).flatten()
    vn = np.array(flatten(my_model.get_attr_param('name')))[my_model.C_]
    for trace_mode in trace_modes:
        try:
            if eng_name == 'reddemcee':
                import arviz as az
                arviz_data = az.from_emcee(sampler=sampler,
                                            var_names=vn)

                # trace
                if trace_mode == 0:
                    circ_mask = np.array(flatten(my_model.get_attr_param('is_circular')))[my_model.C_]
                    circ_var_names = vn[circ_mask]
                    for b in my_model:
                        vn_b = np.array(b.get_attr('name'))[b.C_]
                        az.plot_trace(arviz_data,
                                    figsize=(14, len(vn_b)*2.5),
                                    var_names=vn_b,
                                    circ_var_names=circ_var_names,
                                    plot_kwargs={'color':rc.fg},
                                    trace_kwargs={'color':rc.fg})

                        pl.subplots_adjust(hspace=0.60)
                        savefigname = saveplace+'{} {}.{}'.format(trace_mode_dic[trace_mode], b.name_, fmt)
                        pl.savefig(savefigname)

                # distr
                elif trace_mode == 1:
                    for b in my_model:
                        for p in b[b.C_]:
                            fig, ax = pl.subplots(1, 1)
                            fig.suptitle(p.name)

                            az.plot_dist(arviz_data.posterior[p.name].values,
                                        color=rc.fg,
                                        rug=True,
                                        #figsize=(8, 6),
                                        )
                            #pl.ylabel('Probability Density')
                            pl.xlabel('Value')

                            savefigname = saveplace+'{} {}.{}'.format(trace_mode_dic[trace_mode], p.name, fmt)
                            pl.savefig(savefigname)
                # density intervals
                elif trace_mode == 2:
                    for b in my_model:
                        axes = az.plot_density(
                            [arviz_data],
                            var_names=np.array(b.get_attr('name'))[b.C_],
                            shade=0.2,
                            colors=colors_[b.bnumber_-1],
                            #hdi_markers='v'
                            )

                        fig = axes.flatten()[0].get_figure()
                        fig.suptitle("94% High Density Intervals")

                        savefigname = saveplace+'{} {}.{}'.format(trace_mode_dic[trace_mode], b.name_, fmt)
                        pl.savefig(savefigname)
                # corner plot
                elif trace_mode == 3:
                    ax = az.plot_pair(arviz_data,
                            kind=["scatter", "kde"],
                            marginals=True,
                            marginal_kwargs={'color':rc.fg},
                            point_estimate="median",
                            scatter_kwargs={'color':rc.fg},
                            point_estimate_kwargs={'color':'red'},
                            point_estimate_marker_kwargs={'color':'red',
                                                        's':200,
                                                        'alpha':0.75},
                            )

                    savefigname = saveplace+'{}.{}'.format(trace_mode_dic[trace_mode], fmt)
                    pl.savefig(savefigname)

            elif eng_name == 'dynesty':
                from dynesty import plotting as dyplot
                res2 = sampler

                if trace_mode == 0:
                    # trace
                    for b in my_model:
                        vnb = np.array(b.get_attr('name'))[b.C_]
                        fig, axes = dyplot.traceplot(res2,
                                                    post_color=rc.fg,
                                                    trace_color=rc.fg,
                                                    labels=vnb,
                                                    dims=b.slice_true)
                        savefigname = saveplace+'{} {}.{}'.format(trace_mode_dic[trace_mode], b.name_, fmt)
                        pl.savefig(savefigname)
                # Normalised Posterior
                elif trace_mode == 1:   
                    arviz_data = az.from_emcee(sampler=sampler,
                                                var_names=vn)

                    for b in my_model:
                        for p in b[b.C_]:
                            fig, ax = pl.subplots(1, 1)
                            fig.suptitle(p.name)

                            az.plot_dist(arviz_data.posterior[p.name].values)

                            savefigname = saveplace+'{} {}.{}'.format(trace_mode_dic[trace_mode], p.name, fmt)
                            pl.savefig(savefigname)
                # density intervals
                elif trace_mode == 2:
                    arviz_data = az.from_emcee(sampler=sampler,
                                                var_names=vn)

                    for b in my_model:
                        axes = az.plot_density(
                            [arviz_data],
                            var_names=np.array(b.get_attr('name'))[b.C_],
                            shade=0.2,
                            #hdi_markers='v'
                            )

                        fig = axes.flatten()[0].get_figure()
                        fig.suptitle("94% High Density Intervals")

                        savefigname = saveplace+'{} {}.{}'.format(trace_mode_dic[trace_mode], b.name_, fmt)
                        pl.savefig(savefigname)
                # corner plot
                elif trace_mode == 3:
                    arviz_data = az.from_emcee(sampler=sampler,
                                                var_names=vn)

                    ax = az.plot_pair(arviz_data,
                            kind=["scatter", "kde"],
                            marginals=True,
                            marginal_kwargs={'color':rc.fg},
                            point_estimate="median",
                            scatter_kwargs={'color':rc.fg},
                            point_estimate_kwargs={'color':'red'},
                            point_estimate_marker_kwargs={'color':'red',
                                                        's':90},
                            )
                    savefigname = saveplace+'{}.{}'.format(trace_mode_dic[trace_mode], fmt)
                    pl.savefig(savefigname)

            else:
                print('Method is not yet implemented for {}'.format(eng_name))
                return None

            pl.close('all')
        except:
            print(f'Trace plot for {trace_mode_dic[trace_mode]} failed!')


def plot_KeplerianModel(my_data, my_model, res, saveloc='', options=None):
    if options is None:
        options = {}
    if True:
        saveplace = saveloc + '/models/'
        unsaveplace = saveloc + '/models/uncertainpy/'

        plot_fmt = options['format']
        switch_histogram = options['hist']
        switch_uncertain = options['uncertain']
        switch_errors = options['errors']
        logger_level = options['logger_level']
        gC = options['gC']


        posterior_method = 'GM'
        c = ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        colors = np.array([c,c,c,c,c]).flatten()

        temp_file_names = []
        temp_mod_names = []
        temp_dat_names = []

        if switch_uncertain:
            import uncertainpy as un
            import chaospy as cp
            if logger_level is not None:
                logging.getLogger("chaospy").setLevel(logger_level)
                logging.getLogger("numpoly").setLevel(logger_level)


    def create_mod(data_arg, blocks_arg, tail_x, mod_number=0):
        x = ReddModel(data_arg, blocks_arg)
        x.A_ = []

        temp_script = 'temp_mod_0{}.py'.format(mod_number)
        temp_file_names.append(temp_script)
        temp_mod_names.append('{}/temp/temp_model_{}{}.py'.format(saveloc, x.model_script_no, tail_x))
        temp_dat_names.append('{}/temp/temp_data{}.csv'.format(saveloc, tail_x))

        with open(temp_script, 'w') as f:
            f.write(open(get_support('init.scr')).read())
            # DEPENDENCIES
            f.write('''
import kepler
''')
            # CONSTANTS
            f.write('''
nan = np.nan
A_ = []
mod_fixed_ = []
gaussian_mixture_objects = dict()
''')

            f.write(open(x.write_model_(loc=saveloc, tail=tail_x)).read())

    if True:
        ## COL
        D = deepcopy(my_data)
        ajuste = res
        OGM = my_model

        # Block selection
        DB_A = [b for b in OGM if b.display_on_data_==True]
        NDB_A = [b for b in OGM if b.display_on_data_==False]

        pbar_tot = 1 + len(DB_A)

        pbar = tqdm(total=pbar_tot)
        # data for continuous
        x_c = np.linspace(D['BJD'].min(), D['BJD'].max(), 5000)
        DC = pd.DataFrame({'BJD':x_c, 'RV':np.zeros_like(x_c), 'eRV':np.zeros_like(x_c)})
        ### MOVE DATA AROUND
        # Get models which I dont want to display with a line
        # Substract them to the data
        create_mod(D, NDB_A, '_NDM_A', 0)


        import temp_mod_00
        NDM_A_mod = reload(temp_mod_00).my_model


        ndm, ndferr = NDM_A_mod(ajuste)

        # data gets cleaned from no display
        D['RV'] -= ndm.values
        D['eRV'] = np.sqrt(D['eRV'].values**2 + ndferr.values)

        # get the residuals

        create_mod(D, DB_A, '_DM_A', 1)

        import temp_mod_01
        DM_A_mod = reload(temp_mod_01).my_model

        D['residuals'] = D['RV'].values - DM_A_mod(ajuste)[0]

        # Here we are done. We have 3 different datasets
        # display, og-no_display, continuous

    # FULL MODEL
    # this requires a subgrid, for plotting the residuals

    if True:
        if True:
            fig = pl.figure(figsize=(10, 8))
            gs = gridspec.GridSpec(3, 4)
            if switch_histogram:
                ax = fig.add_subplot(gs[:2, :-1])
                axr = fig.add_subplot(gs[2, :-1], sharex=ax)
                axh = fig.add_subplot(gs[:2, 3], sharey=ax)
                axrh = fig.add_subplot(gs[2, 3], sharey=axr)

            else:
                ax = fig.add_subplot(gs[:2, :])
                axr = fig.add_subplot(gs[2, :], sharex=ax)

            pl.subplots_adjust(hspace=0)

            ax.axhline(0, color='gray', linewidth=2)
            axr.axhline(0, color='gray', linewidth=2)

            pl.subplots_adjust(wspace=0.15)

        # First we plot the data
        if True:
            for b_ins in NDB_A:
                if b_ins.type_ == 'Instrumental':
                    mask = D['Flag'] == b_ins.number_
                    if switch_errors:
                        ax.errorbar(D[mask]['BJD'], D[mask]['RV'], D[mask]['eRV'],
                                c=colors[b_ins.number_-1], marker='o', label=b_ins.instrument_label,
                                ls='')

                        axr.errorbar(D[mask]['BJD'], D[mask]['residuals'], D[mask]['eRV'],
                                c=colors[b_ins.number_-1], marker='o',
                                ls='')
                    else:
                        ax.plot(D[mask]['BJD'], D[mask]['RV'],
                                colors[b_ins.number_-1]+'o', label=b_ins.instrument_label)

                        axr.plot(D[mask]['BJD'], D[mask]['residuals'], colors[b_ins.number_-1]+'o')
        # We set unmodels for uncertainties
        if True:
            for b in DB_A:
                if b.parameterisation == 0:
                    kepmod = Keplerian_Model
                    unkepmod = unKeplerian_Model
                    un_model_name = "unKeplerian_Model"
                    chaos_names = ['Period', 'Amplitude', 'Phase', 'Eccentricity', 'Longitude_Periastron']
                if b.parameterisation == 1:
                    kepmod = Keplerian_Model_1
                    unkepmod = unKeplerian_Model_1
                    un_model_name = "unKeplerian_Model_1"
                    chaos_names = ['lPeriod', 'Amp_sin', 'Amp_cos', 'Ecc_sin', 'Ecc_cos']
                if b.parameterisation == 2:
                    kepmod = Keplerian_Model_2
                    unkepmod = unKeplerian_Model_2
                    un_model_name = "unKeplerian_Model_2"
                    chaos_names = ['Period', 'Amplitude', 'Time_Periastron', 'Eccentricity', 'Longitude_Periastron']
                if b.parameterisation == 3:
                    kepmod = Keplerian_Model_3
                    unkepmod = unKeplerian_Model_3
                    un_model_name = "unKeplerian_Model_3"
                    chaos_names = ['Period', 'Amplitude', 'Time_Periastron', 'Ecc_sin', 'Ecc_cos']

        # Now we plot our line
        DB_AC = deepcopy(DB_A)

        create_mod(DC, DB_AC, '_DM_AC', 2)

        import temp_mod_02
        DM_AC_mod = reload(temp_mod_02).my_model


        DC['RV'] = DM_AC_mod(ajuste)[0]

        ax.plot(DC['BJD'], DC['RV'], color=rc.fg, ls='--')

        if True and switch_histogram:
            nbins = 5
            while nbins < len(D):
                counts, bins = np.histogram(D['RV'], bins=nbins)
                if (counts==0).any():
                    break
                else:
                    nbins += 1

            nbins = 5
            while nbins < len(D):
                counts, bins = np.histogram(D['residuals'], bins=nbins)
                if (counts==0).any():
                    break
                else:
                    nbins += 1

            # PLOT HISTOGRAMS
            axh.hist(D['RV'], bins=nbins-1, orientation='horizontal', ec=rc.fg, lw=1)
            axrh.hist(D['residuals'], bins=nbins-1, orientation='horizontal', ec=rc.fg, lw=1)
            # HIDE TICKS
            axh.tick_params(axis="x", labelbottom=False)
            axh.tick_params(axis="y", labelleft=False)
            axrh.tick_params(axis="y", labelleft=False)

            axrh.set_xlabel('Counts')
        # Ticks and labels
        if True:
            ax.tick_params(axis="x", labelbottom=False)

            ax.set_title('Keplerian Model')
            ax.set_ylabel(r'RVs $\frac{m}{s}$')
            ax.legend(fontsize=10)#, framealpha=0)

            axr.set_xlabel('BJD (days)')
            axr.set_ylabel(r'Residuals $\frac{m}{s}$')

            pl.savefig(saveplace+'{}.{}'.format('keplerian_model', plot_fmt),
                       bbox_inches='tight')

            # here we plot the phasefolded versions

        # we also need a grid for the residuals
        pbar.update(1)


    # PHASEFOLD
    for mode in range(True+switch_uncertain):
        # add a tail to the name if uncertainties
        if mode == 0:
            name_tail = ''
        if mode == 1:
            name_tail = '_uncertainties'
            pbar_tot = len(DB_A)
            pbar = tqdm(total=pbar_tot)
        chaos_thetas = []
        nb_ = 0
        for b in DB_A:
            # make grid
            if True:
                fig = pl.figure(figsize=(10, 8))
                gs = gridspec.GridSpec(3, 4)

                if switch_histogram:
                    ax = fig.add_subplot(gs[:2, :-1])
                    axr = fig.add_subplot(gs[2, :-1], sharex=ax)
                    axh = fig.add_subplot(gs[:2, 3], sharey=ax)
                    axrh = fig.add_subplot(gs[2, 3], sharey=axr)
                else:
                    ax = fig.add_subplot(gs[:2, :])
                    axr = fig.add_subplot(gs[2, :], sharex=ax)

                pl.subplots_adjust(hspace=0)

                ax.axhline(0, color='gray', linewidth=2)
                axr.axhline(0, color='gray', linewidth=2)

            pl.subplots_adjust(wspace=0.15)

            # adjust params for different parameterisations
            # now just in period for param 1
            # Get PAE? ## RED
            per = np.exp(b[0].value) if b.parameterisation == 1 else b[0].value
            D_PF = deepcopy(D)

            TB = [deepcopy(b)]

            if True:
                create_mod(D_PF, TB, '_TB', 3)
                import temp_mod_03
                TM_mod = reload(temp_mod_03).my_model

            D_PF['RV_TB'] = TM_mod(ajuste)[0]
            D_PF['RV_D'] = D_PF['residuals'].values + D_PF['RV_TB'].values
            D_PF['eRV_D'] = TM_mod(ajuste)[1]

            D_PF = fold_dataframe(D_PF, per=per)

            ## plot data per instrument
            for b_ins in NDB_A:
                if b_ins.type_ == 'Instrumental':
                    mask = D_PF['Flag']==b_ins.number_
                    
                    if switch_errors:
                        ax.errorbar(D_PF[mask]['BJD'], D_PF[mask]['RV_D'], yerr=D_PF[mask]['eRV'],
                                    c=colors[b_ins.number_-1], marker='o', ls='', label=b_ins.instrument_label)

                        axr.errorbar(D_PF[mask]['BJD'], D_PF[mask]['residuals'], yerr=D_PF[mask]['eRV'],
                                c=colors[b_ins.number_-1], marker='o', ls='')
                    else:
                        ax.plot(D_PF[mask]['BJD'], D_PF[mask]['RV_D'],
                                colors[b_ins.number_-1]+'o', label=b_ins.instrument_label)

                        axr.plot(D_PF[mask]['BJD'], D_PF[mask]['residuals'],
                                colors[b_ins.number_-1]+'o')

            ### create model line for the phasefold
            xpf_c = np.linspace(D_PF['BJD'].min(), D_PF['BJD'].max(), 5000)
            D_PFC = pd.DataFrame({'BJD':xpf_c, 'RV':np.zeros_like(xpf_c), 'eRV':np.zeros_like(xpf_c)})

            TB_C = [deepcopy(b)]

            if True:
                create_mod(D_PFC, TB_C, '_TB_C', 4)
                import temp_mod_04
                TM_C_mod = reload(temp_mod_04).my_model

            D_PFC['RV'] = TM_C_mod(ajuste)[0]


            ## plot phasefold
            if True:
                # get uncertainties
                if mode==1:
                    un_model = un.Model(run=unkepmod,
                                        labels=['BJD (days)', r'RVs $\frac{m}{s}$'],
                                        interpolate=True,
                                        logger_level=u'error',
                                        #postprocess=func
                                        )

                    chaostheta = {}
                    for i in range(len(b)):
                        if posterior_method == 'GM':
                            if b[i].fixed is None:
                                chaostheta[chaos_names[i]] = cp.TruncNormal(lower=b[i].limits[0],
                                                                            upper=b[i].limits[1],
                                                                            mu=b[i].posterior.mixture_mean,
                                                                            sigma=b[i].posterior.mixture_sigma)
                            else:
                                chaostheta[chaos_names[i]] = b[i].value_mean

                    chaos_thetas.append(chaostheta)

                    parameters = un.Parameters(chaostheta)

                    UQ = un.UncertaintyQuantification(model=un_model,
                                                      parameters=parameters,
                                                      logger_level=u'critical',
                                                      logger_filename=unsaveplace+'uncertainpy.log')

                    with nullify_output(suppress_stdout=True, suppress_stderr=True):
                        undata = UQ.quantify(seed=10,
                                             method='pc',
                                             plot=None,
                                             #plot='all',
                                             pc_method='collocation',
                                             logger_level=u'critical',
                                             figure_folder=unsaveplace+'figures',
                                             data_folder=unsaveplace+'data',
                                             single=False)

                    keplerian_data = un.Data('{}data/{}.h5'.format(unsaveplace,
                                                                   un_model_name))

                    untime = undata[un_model_name].time
                    unmean = undata[un_model_name].mean
                    unvariance = undata[un_model_name].variance
                    unpercentile_5 = undata[un_model_name].percentile_5
                    unpercentile_95 = undata[un_model_name].percentile_95
                    unsensitivity = undata[un_model_name].sobol_first

                # Plot model line
                ax.plot(D_PFC['BJD'].values, D_PFC['RV'].values, color=rc.fg, ls='--')

                # plot uncertainties
                if mode == 1:
                    ax.fill_between(untime,
                                    unpercentile_5,
                                    unpercentile_95,
                                    color=rc.fg,
                                    alpha=0.5)


                    if True:
                        figs, axs = pl.subplots(figsize=(8, 6))
                        for i in range(unsensitivity.shape[0]):
                            axs.plot(untime, unsensitivity[i],
                                       linewidth=1.5,
                                       )
                        axs.set_title('First-order Sobol indices')
                        axs.set_xlabel('BJD (days)')
                        axs.set_ylabel('First-order Sobol indices')
                        axs.legend(parameters.get_from_uncertain(),
                                            loc='upper right',
                                            framealpha=0.5)

                        figs.savefig(saveplace+'sobol_{}.{}'.format(b.name_+name_tail, plot_fmt),
                                   bbox_inches='tight')
                        pl.close(figs)
                # Plot Histogram Model
                if switch_histogram:
                    nbins = 5
                    while nbins < len(D_PF):
                        counts, bins = np.histogram(D_PF['RV_D'], bins=nbins)
                        if (counts==0).any():
                            break
                        else:
                            nbins += 1

                    axh.hist(D_PF['RV_D'], bins=nbins-1, orientation='horizontal', ec=rc.fg, lw=1)

                # Plot histogram residuals
                if switch_histogram:
                    nbins = 5
                    while nbins < len(D_PF):
                            counts, bins = np.histogram(D_PF['residuals'], bins=nbins)
                            if (counts==0).any():
                                break
                            else:
                                nbins += 1
                    axrh.hist(D_PF['residuals'], bins=nbins-1, orientation='horizontal', ec=rc.fg, lw=1)

                # Ticks and labels

                if True:
                    ax.tick_params(axis="x", labelbottom=False)

                    if switch_histogram:
                        axh.tick_params(axis="x", labelbottom=False)
                        axh.tick_params(axis="y", labelleft=False)
                        axrh.tick_params(axis="y", labelleft=False)

                    ax.set_title('Keplerian Model')
                    ax.set_ylabel(r'RVs $\frac{m}{s}$')
                    ax.legend(fontsize=10)#, framealpha=0)

                    axr.set_xlabel('BJD (days)')
                    axr.set_ylabel(r'Residuals $\frac{m}{s}$')



            pl.savefig(saveplace+'{}.{}'.format(b.name_+name_tail, plot_fmt),
                       bbox_inches='tight')

            nb_ += 1

            pbar.update(1)


            # print('MARKER 5')
            if nb_ == len(DB_A):
                pbar.close()

    temp_file_folder = saveloc+'/models/temp/'


    with nullify_output():
        for file in list(set(temp_file_names)):
            try:
                os.system('mv {0} {1}{0}'.format(file, temp_file_folder))
            except Warning:
                print('Couldnt auto-delete temp files')

        for file in list(set(temp_mod_names)):
            try:
                os.system('mv {0} {1}{2}'.format(file, temp_file_folder, file.split('/')[-1]))
            except Warning:
                print('Couldnt auto-delete temp files')

        for file in list(set(temp_dat_names)):
            try:
                os.system('mv {0} {1}{2}'.format(file, temp_file_folder, file.split('/')[-1]))
            except Warning:
                print('Couldnt auto-delete temp files')

    return chaos_thetas









# blue verde rojo purp naran cyan brown blue green
