# type: ignore
# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# sourcery skip

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT

import contextlib
import sys, os
import logging
from copy import deepcopy
from importlib import reload, import_module

import gc
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as pl
import matplotlib.colors as plc
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from tqdm import tqdm

from .emp_model import ReddModel
from .globals import _PLATFORM_SYSTEM, _CORES
from .block_repo import *
from .qol_utils import *
from .math_utils import *

from reddcolors import Palette

import multiprocessing
if _PLATFORM_SYSTEM == 'Darwin':
    #multiprocessing.set_start_method('spawn')
    pass

if True:
    matplotlib.use('Agg')

rc = Palette()

cor_ = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
Color_Cycler = np.array(cor_*5)

def hex2rgb(hex):
    hex_cleaned = hex.lstrip('#')
    return tuple(int(hex_cleaned[i:i+2], 16) for i in (0, 2 ,4))


def mk_cmap(target_colors, ncolors=100, mode=0):
    # mode=0 adds rc.fg as last color
    bgfg = [rc.bg, rc.fg]
    for c in bgfg:
        if c[0] == '#':
            c = hex2rgb(c)
    res = [bgfg[0]]
    for tc in target_colors:
        res.append(tc)
    if mode==0:
        res.append(bgfg[1])
    return plc.LinearSegmentedColormap.from_list(f'mycmap_{tc}', res, N=ncolors)


def mk_bool_cmap(color):
    return plc.ListedColormap([rc.bg, color])


def mini_heatmap(x, y, s, bins=500):
    from scipy.ndimage import gaussian_filter
    xn = (x - min(x)) / (max(x)- min(x))
    yn = (y - min(y)) / (max(y)- min(y))

    heatmap, xedges, yedges = np.histogram2d(xn, yn, bins=(bins*4, bins))
    heatmap = gaussian_filter(heatmap, sigma=s)


    extent = [0.0, 4.0, 0.0, 1.0]
    return heatmap.T, extent


def plot_GM_Estimator(estimator, options=None):
    # sourcery skip: use-fstring-for-formatting
    if options is None:
        options = {}
    if True:
        saveloc = options['saveloc']
        saveplace = saveloc + '/plots/GMEstimates/'

        plot_fmt = options['format']  # 'png'
        plot_nm = options['plot_name']  # ''
        plot_title = options['plot_title']  # None
        plot_ylabel = options['plot_ylabel'] # None
        fill_cor = options['fill_cor']  # 0

        sig_factor = options['sig_factor']

        if plot_title is None:
            plot_title = 'Optimal estimate with Gaussian Mixtures\n for '

        if plot_ylabel is None:
            plot_ylabel = 'Probability Density'


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
        ax.fill(xx[i], yy[i], c=Color_Cycler[fill_cor], alpha=1/sig_factor, zorder=2*(i+1)-1)

        vlines_kwargs = {'lw':[1.5, 1.5], 'ls':['--']}
        ax.vlines([xx[i][0], xx[i][-1]], ymin=[min(yy[i]), min(yy[i])],
                                         ymax=[yy[i][1], yy[i][-2]],
                                         colors=[rc.fg, rc.fg],
                                         zorder=2*(i+1),
                                         **vlines_kwargs)

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
        label=rf'$\mu = {mu_display}$',
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
    ax.set_title(plot_title+'{}'.format(plot_nm[2:]))
    ax.set_xlabel('{} {}'.format(plot_nm[2:], estimator.unit))
    ax.set_ylabel(plot_ylabel)


    pl.savefig(saveplace+'{}.{}'.format(plot_nm, plot_fmt),
               bbox_inches='tight')

    pl.close('all')



def plot_trace(sampler=None, eng_name='', my_model=None, options={}):
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
    
    
    vn = np.array(flatten(my_model.get_attr_param('name')))[my_model.C_]

    dothis = []
    for i in range(4):
        dothis.append(i in trace_modes)

    try:
        if eng_name == 'reddemcee':
            import arviz as az
            arviz_data = az.from_emcee(sampler=sampler,
                                        var_names=vn)
            
            for trace_mode in trace_modes:

                # trace
                if trace_mode == 0:
                    circ_mask = np.array(flatten(my_model.get_attr_param('is_circular')))[my_model.C_]
                    circ_var_names = vn[circ_mask]
                    for b in my_model:
                        if b.ndim_ == 0:
                            break
                        vn_b = np.array(b.get_attr('name'))[b.C_]
                        az.plot_trace(arviz_data,
                                    figsize=(14, len(vn_b)*2.5),
                                    var_names=vn_b,
                                    circ_var_names=circ_var_names,
                                    plot_kwargs={'color':rc.fg},
                                    trace_kwargs={'color':rc.fg})

                        pl.subplots_adjust(hspace=0.60)
                        savefigname = saveplace + f'{trace_mode_dic[trace_mode]} {b.name_}.{fmt}'
                        pl.savefig(savefigname)
                        pl.close()

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

                            savefigname = saveplace + f'{trace_mode_dic[trace_mode]} {p.name}.{fmt}'
                            pl.savefig(savefigname)
                elif trace_mode == 2:
                    for b in my_model:
                        if b.ndim_ == 0:
                            break
                        axes = az.plot_density(
                            [arviz_data],
                            var_names=np.array(b.get_attr('name'))[b.C_],
                            shade=0.2,
                            colors=Color_Cycler[b.bnumber_-1],
                            #hdi_markers='v'
                            )

                        fig = axes.flatten()[0].get_figure()
                        fig.suptitle("94% High Density Intervals")

                        savefigname = saveplace + f'{trace_mode_dic[trace_mode]} {b.name_}.{fmt}'
                        pl.savefig(savefigname)
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

                    savefigname = saveplace + f'{trace_mode_dic[trace_mode]}.{fmt}'
                    pl.savefig(savefigname)

                
        elif eng_name == 'dynesty':
            from dynesty import plotting as dyplot
            res2 = sampler
            for trace_mode in trace_modes:
                if trace_mode == 0:
                    # trace
                    for b in my_model:
                        vnb = np.array(b.get_attr('name'))[b.C_]
                        fig, axes = dyplot.traceplot(res2,
                                                    post_color=rc.fg,
                                                    trace_color=rc.fg,
                                                    labels=vnb,
                                                    dims=b.slice_true)
                        savefigname = saveplace + f'{trace_mode_dic[trace_mode]} {b.name_}.{fmt}'
                        pl.savefig(savefigname)
                '''
                elif trace_mode == 1:
                    arviz_data = az.from_emcee(sampler=sampler,
                                                var_names=vn)

                    for b in my_model:
                        for p in b[b.C_]:
                            fig, ax = pl.subplots(1, 1)
                            fig.suptitle(p.name)

                            az.plot_dist(arviz_data.posterior[p.name].values)

                            savefigname = saveplace + f'{trace_mode_dic[trace_mode]} {p.name}.{fmt}'
                            pl.savefig(savefigname)
                    
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

                        savefigname = saveplace + f'{trace_mode_dic[trace_mode]} {b.name_}.{fmt}'
                        pl.savefig(savefigname)
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
                    savefigname = saveplace + f'{trace_mode_dic[trace_mode]}.{fmt}'
                    pl.savefig(savefigname)
                '''
        else:
            print(f'Method is not yet implemented for {eng_name}')
            return None

        pl.close('all')

    except Exception():
        print(f'Trace plot for {trace_mode_dic[trace_mode]} failed!')


def plot_trace2(sampler=None, eng_name='', my_model=None, options={}):
    if True:
        trace_modes = options['modes']
        saveplace = options['saveloc'] + '/plots/arviz/'
        fmt = options['format']
        burnin = options['burnin']
        thin = options['thin']


        if trace_modes is None:
            trace_modes = [0]
        # 0:trace, 1:norm_post, 2:dens_interv, 3:corner
        
        trace_mode_dic = {0:'Trace Plot',
                        1:'Normalised Posterior',
                        2:'Density Interval',
                        3:'Corner Plot'}
        


        vn = []
        for v in my_model.get_attr_param('name'):
            vn.extend(v)
        vn = np.array(vn)[my_model.C_]

        dothis = []
        for i in range(4):
            dothis.append(i in trace_modes)

        pbar = tqdm(total=len([1 for b in my_model]))

    if True:
        if eng_name == 'reddemcee':
            import arviz as az
            arviz_data = az.from_emcee(sampler=sampler,
                                        var_names=vn).sel(draw=slice(burnin, None, thin))
            #arviz_data.sel(draw=slice(100, None))

            for b in my_model:
                if b.ndim_ == 0:
                    pbar.update(1)
                    break
                
                vn_b = np.array(b.get_attr('name'))[b.C_]
                circ_var_names = vn_b[np.array(b.get_attr('is_circular'))[b.C_]]

                # TRACES
                if dothis[0]:
                    savefigname = saveplace + 'traces/' + f'{trace_mode_dic[0]} {b.name_}.{fmt}'

                    az.plot_trace(arviz_data,
                                  #compact=True,
                                  figsize=(14, len(vn_b)*2.5),
                                  var_names=vn_b,
                                  circ_var_names=circ_var_names,
                                  combined=False,
                                  compact=True,
                                  chain_prop={'color':[Color_Cycler[b.bnumber_-1] for _ in range(sampler.shape[0])],
                                              'alpha':[0.25 for _ in range(sampler.shape[0])],
                                              'lw':[2 for _ in range(sampler.shape[0])],
                                              },
                                  #plot_kwargs={'color':rc.fg},
                                  #trace_kwargs={'color':rc.fg},

                                  )

                    #pl.subplots_adjust(hspace=0.60)
                    pl.tight_layout()
                    pl.savefig(savefigname)
                    pl.close()

                # DENSITY INTERVALS
                if dothis[2]:
                    savefigname = saveplace + 'density_intervals/' + f'{trace_mode_dic[2]} {b.name_}.{fmt}'
                    axes = az.plot_density(
                            [arviz_data],
                            var_names=vn_b,
                            shade=0.2,
                            colors=Color_Cycler[b.bnumber_-1],
                            #hdi_markers='v'
                            )
                    fig = axes.flatten()[0].get_figure()
                    fig.suptitle("94% High Density Intervals")
                    pl.savefig(savefigname)
                    pl.close()

                # CORNERPLOT
                if dothis[3] and b.ndim_ > 2:
                    #pbar = tqdm(total=1)
                    #cmap_ = 
                    savefigname = saveplace + 'cornerplots/' + f'{trace_mode_dic[3]} {b.name_}.{fmt}'
                    az.plot_pair(arviz_data,
                                 var_names=vn_b,
                                 figsize=(3*len(vn_b), 3*len(vn_b)),
                                kind='kde',
                                
                                marginals=True,  # plot diagonals/histo
                                marginal_kwargs={'plot_kwargs':{'color':rc.fg,
                                                                'lw':2},
                                                 'fill_kwargs':{'color':Color_Cycler[b.number_-2],
                                                                'alpha':0.85,
                                                                },
                                                 },

                                kde_kwargs={'contourf_kwargs':{
                                                            'cmap':mk_cmap([Color_Cycler[b.bnumber_-1]], ncolors=100),
                                                            },
                                            'contour_kwargs':{
                                                            'colors':rc.fg,
                                                            'alpha':0.65,
                                                            },
                                            #'fill_last':True,
                                            #'hdi_probs':[0.5-0.341,
                                            #             0.5,
                                            #             0.5+0.341],

                                            },
                                point_estimate='mode',
                                point_estimate_kwargs={'lw':1.5,
                                                        'c':'r',
                                                        'alpha':0.75},
                                point_estimate_marker_kwargs={'marker':''},


                                )

                    pl.tight_layout()
                    pl.subplots_adjust(hspace=0)
                    pl.subplots_adjust(wspace=0)
                    pl.savefig(savefigname)
                    pl.close()
                    #pbar.update(1)
                    #pbar.close()

                # NORMALISED POSTERIORS
                if dothis[1]:
                    for p in b[b.C_]:
                        fig, ax = pl.subplots(1, 1)
                        savefigname = saveplace + 'normed_posteriors/'+ f'{trace_mode_dic[1]} {p.name}.{fmt}'
                        fig.suptitle(p.name)

                        az.plot_dist(arviz_data.posterior[p.name].values,
                                    color=rc.fg,
                                    rug=True,
                                    )
                        #pl.ylabel('Probability Density')                            
                        pl.xlabel('Value')
                        pl.savefig(savefigname)
                        pl.close()

                pbar.update(1)
                gc.collect()
                pl.close('all')

        elif eng_name == 'dynesty':
            from dynesty import plotting as dyplot
            res2 = sampler
            os.rename(f'{saveplace}density_intervals', f'{saveplace}runplot')
            os.rename(f'{saveplace}normed_posteriors', f'{saveplace}boundplot')

            if dothis[1]:
                fig, axes = dyplot.runplot(res2,
                                            color=rc.fg)  # summary (run) plot
                savefigname = f'{saveplace}runplot/runplot.{fmt}'
                pl.savefig(savefigname)
                pl.close()

            for b in my_model:
                if b.ndim_ == 0:
                    pbar.update(1)
                    break

                vnb = np.array(b.get_attr('name'))[b.C_]
                vnb_mini = np.array(b.get_attr('mininame'))[b.C_]
                vnb_mini = [f'${mn}$' for mn in vnb_mini]
                if dothis[0]:
                    try:
                        fig, axes = dyplot.traceplot(res2,
                                                    labels=vnb_mini,
                                                    dims=b.slice_true,
                                                    post_color=Color_Cycler[b.bnumber_-1],
                                                    show_titles=True
                                                    )
                        savefigname = f'{saveplace}traces/{trace_mode_dic[0]} {b.name_}.{fmt}'
                        pl.savefig(savefigname)
                    except:
                        print(f'Dynesty dyplot failed for block {b.name_}!')
                

                if dothis[2]:
                    # TODO: add bounding plots
                    print('Bounding plots not implemented for dynesty yet')

                if dothis[3] and b.ndim_ > 2:
                    fig, axes = dyplot.cornerplot(res2,
                                            color=Color_Cycler[b.bnumber_-1],
                                            dims=b.slice_true,
                                            labels=vnb_mini,
                                            show_titles=True,
                                            max_n_ticks=3,
                                            quantiles=[0.025, 0.5, 0.975])
                
                    savefigname = f'{saveplace}cornerplots/corner {b.name_}.{fmt}'
                    pl.savefig(savefigname)
                    pl.close()
                
                pbar.update(1)
                gc.collect()
                pl.close('all')

        else:
            print(f'Method is not yet implemented for {eng_name}')
            return None

        pbar.close()
        pl.close('all')
    #except Exception():
    else:
        print(f'Trace plot for {trace_mode_dic[0]} failed!')

#plot_trace2(sim.sampler.backend[0], sim.engine__.__name__, sim.model, sim.plot_all_list[-1])

#import arviz as az
#from tqdm import tqdm
#from reddcolors import Palette
#import matplotlib.colors as plc

#rc = Palette()

#plot_KeplerianModel(options=sim.plot_keplerian_model, **keplerian_kwargs)

def plot_KeplerianModel_old(my_data=None, my_model=None, res=[], common_t=0, options=None):
    if options is None:
        options = {}
    if True:
        saveloc = options['saveloc']
        saveplace = saveloc + '/plots/models/'
        unsaveplace = saveloc + '/plots/models/uncertainpy/'

        plot_fmt = options['format']
        switch_histogram = options['hist']
        switch_uncertain = options['uncertain']
        switch_errors = options['errors']
        switch_periodogram = options['periodogram']
        switch_celerite = options['celerite']
        logger_level = options['logger_level']
        gC = options['gC']
        use_c = options['use_c']

        # FULL_MODEL
        fm_figsize = (10, 8)

        if True:
            fm_axhline_kwargs = {'color':'gray', 'linewidth':3}
            fm_errorbar_kwargs = {'marker':'o', 'ls':'', 'alpha':0.8,
                                'lw':2,
                                'markersize':10,
                                'markeredgewidth':1,
                                'markeredgecolor':'k',
                                }

            # LINE
            fm_model_line = {'ls':'--', 'lw':3}  # ?
            # HIST
            fm_hist = {'lw':2}  # 1
            fm_hist_tick_fs = 0
            # LEGEND
            fm_legend_fs = 14 # 10

            # FM_FRAME
            fm_frame_lw = 3
            # FM TICKS
            fm_tick_xsize = 20
            fm_tick_ysize = 20
            # LABELS
            fm_label_fs = 22
            # title
            fm_title_fs = 24

            plot_fmt = 'pdf'
            # PHASE


        common_t = int(common_t)  # my_data['BJD'].min()
        posterior_method = 'GM'

        temp_file_names = []
        temp_mod_names = []
        temp_dat_names = []

        if switch_uncertain:
            import uncertainpy as un
            import chaospy as cp
            if logger_level is not None:
                logging.getLogger("chaospy").setLevel(logger_level)
                logging.getLogger("numpoly").setLevel(logger_level)


    def create_mod1(data_arg, blocks_arg, tail_x, mod_number=0):
        x = ReddModel(blocks_arg, data_RV=data_arg)
        x.switch_plot = True

        x.A_ = []
        x.nins__ = my_model.nins__
        #x.refresh__()  # needed to get nins

        temp_script = f'temp_mod_0{mod_number}.py'
        temp_file_names.append(temp_script)
        temp_mod_names.append(f'{saveloc}/temp/temp_model_{x.model_script_no}{tail_x}.py')
        temp_dat_names.append(f'{saveloc}/temp/temp_data{tail_x}.csv')

        with open(temp_script, 'w') as f:
            f.write(open(get_support('init_reddemcee.scr')).read())
            # DEPENDENCIES
            if use_c:
                f.write(f'''
from fast_kepler import calc_rv0
''')
            else:
                f.write('''
import kepler
''')
            if switch_celerite:
                f.write('''
import celerite2
import celerite2.terms as cterms
''')
            # CONSTANTS
            f.write(f'''
nan = np.nan
A_ = []
mod_fixed_ = []
gaussian_mixture_objects = dict()

cornums = {my_model.cornums}
''')

            f.write(open(x.write_model(loc=saveloc, tail=tail_x)).read())


    def create_mod(data_arg, blocks_arg, tail_x, mod_number=0):
        x = ReddModel(blocks_arg, data_RV=data_arg)
        x.switch_plot = True

        x.refresh__()  # needed to get nins

        temp_script = f'temp_mod_0{mod_number}.py'
        temp_file_names.append(temp_script)
        temp_mod_names.append(f'{saveloc}/temp/temp_model_{x.model_script_no}{tail_x}.py')
        temp_dat_names.append(f'{saveloc}/temp/temp_data{tail_x}.csv')

        with open(temp_script, 'w') as f:
            f.write(open(get_support('init_reddemcee.scr')).read())
            # DEPENDENCIES
            if use_c:
                f.write(f'''
from fast_kepler import calc_rv0
''')
            else:
                f.write('''
import kepler
''')
            if switch_celerite:
                f.write('''
import celerite2
import celerite2.terms as cterms
''')
            # CONSTANTS
            f.write(f'''
nan = np.nan
A_ = []
mod_fixed_ = []
gaussian_mixture_objects = dict()

cornums = {my_model.cornums}
''')

            f.write(open(x.write_model(loc=saveloc, tail=tail_x)).read())



    def dual_plot(data, DB, pbar, savename=''):
        data1 = deepcopy(data)
        if DB:
            create_mod(data1, DB, '_DB', 1)
            import temp_mod_01
            DM_aux = reload(temp_mod_01).my_model
            rv_aux, er_aux = DM_aux(ajuste)

        else:
            rv_aux, er_aux = np.zeros(len(data1['residuals'])), np.zeros(len(data1['residuals']))

        data1['RV'] = data1['residuals'] + rv_aux

        # make figure
        if True:
            fig = pl.figure(figsize=fm_figsize)
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
            ax.axhline(0, **fm_axhline_kwargs)
            axr.axhline(0, **fm_axhline_kwargs)

            pl.subplots_adjust(wspace=0.15)

        # plot data
        if True:
            for n_ins in range(OGM.nins__):
                mask = data1['Flag'] == (n_ins + 1)
                if switch_errors:
                    ax.errorbar(data1[mask]['BJD'],
                                data1[mask]['RV'],
                                data1[mask]['eRV'],
                                c=Color_Cycler[n_ins],
                                label=OGM.instrument_names_RV[n_ins],
                                **fm_errorbar_kwargs)

                    axr.errorbar(data1[mask]['BJD'],
                                 data1[mask]['residuals'],
                                 data1[mask]['eRV'],
                                c=Color_Cycler[n_ins],
                                **fm_errorbar_kwargs)
                else:
                    #print(f'{data1[mask]['RV']=}')
                    ax.plot(data1[mask]['BJD'],
                            data1[mask]['RV'],
                            f'{Color_Cycler[n_ins]}o',
                            label=OGM.instrument_names_RV[n_ins])

                    axr.plot(data1[mask]['BJD'],
                             data1[mask]['residuals'],
                             Color_Cycler[n_ins]+'o')

        if switch_periodogram:
                    plot_periodogram(data1, options)

        # make continuum
        if True:
            x_c = np.linspace(data1['BJD'].min(),
                              data1['BJD'].max(),
                              5000)

            DC = pd.DataFrame({'BJD':x_c,
                               'RV':np.zeros_like(x_c),
                               'eRV':np.ones_like(x_c)})

            if len(DB):
                DBc = deepcopy(DB)
                create_mod(DC, DBc, '_DBc', 2)
                import temp_mod_02
                DMc_aux = reload(temp_mod_02).my_model
                rv_aux = DMc_aux(ajuste)[0]
                DC['RV'] = rv_aux

        # plot line
        if True:
            ax.plot(DC['BJD'],
                    DC['RV'],
                    color=rc.fg,
                    ls=fm_model_line['ls'],
                    lw=fm_model_line['lw'],)

        if True and switch_histogram:
            nbins = 5
            while nbins < len(D):
                counts, bins = np.histogram(data1['RV'],
                                            bins=nbins)

                if (counts==0).any():
                    break
                else:
                    nbins += 1

            nbins = 5
            while nbins < len(D):
                counts, bins = np.histogram(data1['residuals'],
                                            bins=nbins)
                if (counts==0).any():
                    break
                else:
                    nbins += 1

            # PLOT HISTOGRAMS
            axh.hist(data1['RV'],
                     bins=nbins-1, orientation='horizontal', ec=rc.fg, lw=fm_hist['lw'])
            axrh.hist(data1['residuals'],
                      bins=nbins-1, orientation='horizontal', ec=rc.fg, lw=fm_hist['lw'])
            # HIDE TICKS
            axh.tick_params(axis="x", labelbottom=False, labelsize=fm_tick_xsize)
            axh.tick_params(axis="y", labelleft=False)

            axrh.tick_params(axis="y", labelleft=False)
            axrh.tick_params(axis="x", labelsize=fm_tick_xsize)
            axrh.set_xlabel('Counts', fontsize=fm_label_fs)

        # Ticks and labels
        if True:
            ax.tick_params(axis="x", labelbottom=False)
            axr.tick_params(axis="x", labelsize=fm_tick_xsize)

            ax.tick_params(axis="y", labelsize=fm_tick_ysize)
            axr.tick_params(axis="y", labelsize=fm_tick_ysize)

            #ax.set_title('Keplerian Model', fontsize=fm_title_fs)
            ax.set_ylabel(r'RVs ($\frac{m}{s}$)', fontsize=fm_label_fs)
            ax.legend(fontsize=fm_legend_fs)#, framealpha=0)

            if common_t:
                axr.set_xlabel(f'BJD (days) + {common_t}', fontsize=fm_label_fs)
            else:
                axr.set_xlabel(f'BJD (days)', fontsize=fm_label_fs)

            axr.set_ylabel(r'Residuals ($\frac{m}{s}$)', fontsize=fm_label_fs)


        # SPINES
        if True:
            for spine in ax.spines.values():
                spine.set_linewidth(fm_frame_lw)
            for spine in axr.spines.values():
                spine.set_linewidth(fm_frame_lw)

            for spine in axh.spines.values():
                spine.set_linewidth(fm_frame_lw)
            for spine in axrh.spines.values():
                spine.set_linewidth(fm_frame_lw)

        fig.savefig(saveplace+'{}.{}'.format(f'{savename}', plot_fmt),
                    bbox_inches='tight')


        pbar.update(1)
        pass


    if True:
        # find base for t
        #common_t = find_common_integer_sequence(my_data['BJD'])

        #if common_t:
        #    my_data['BJD'] -= common_t

        ## COL
        D = deepcopy(my_data)
        ajuste = res
        OGM = my_model

        # Block selection
        DB_all = list(OGM)
        DB_all_kep = [b for b in OGM if b.display_on_data_==True]  # Keplerians

        # pbar
        pbar_tot = 1 + len(DB_all_kep)
        pbar = tqdm(total=pbar_tot)

        ### MOVE DATA AROUND

        # this model contains everything
        if DB_all:
            create_mod(D, DB_all, '_DB_all', 0)
            import temp_mod_00
            DB_all_mod = reload(temp_mod_00).my_model


        # add additional error and residuals to base model
        ndm, ndferr = DB_all_mod(ajuste)
        #D['eRV_model'] = np.sqrt(ndferr - D['eRV'].values**2)
        D['eRV'] = np.sqrt(ndferr)
        D['residuals'] = D['RV'] - ndm


        ##############
        ##############
        ##############
        #  this model contains just the keplerians
        if DB_all_kep:
            create_mod(D, DB_all_kep, '_DB_all_kep', 1)
            import temp_mod_01
            DM_all_kep = reload(temp_mod_01).my_model
            rv_all_kep, error_all_kep = DM_all_kep(ajuste)

            # RVs without instrumentals!
            

    # FULL MODEL

    if True:
        # Plot top fig, residuals + kep_all
        '''
        # We set unmodels for uncertainties
        if len(DB_all_kep):
            for b in DB_all_kep:
                if b.parameterisation == 0:
                    #kepmod = Keplerian_Model
                    unkepmod = unKeplerian_Model
                    un_model_name = "unKeplerian_Model"
                    chaos_names = ['Period', 'Amplitude', 'Phase', 'Eccentricity', 'Longitude_Periastron']
                if b.parameterisation == 1:
                    #kepmod = Keplerian_Model_1
                    unkepmod = unKeplerian_Model_1
                    un_model_name = "unKeplerian_Model_1"
                    chaos_names = ['lPeriod', 'Amp_sin', 'Amp_cos', 'Ecc_sin', 'Ecc_cos']
                if b.parameterisation == 2:
                    #kepmod = Keplerian_Model_2
                    unkepmod = unKeplerian_Model_2
                    un_model_name = "unKeplerian_Model_2"
                    chaos_names = ['Period', 'Amplitude', 'Time_Periastron', 'Eccentricity', 'Longitude_Periastron']
                if b.parameterisation == 3:
                    #kepmod = Keplerian_Model_3
                    unkepmod = unKeplerian_Model_3
                    un_model_name = "unKeplerian_Model_3"
                    chaos_names = ['Period', 'Amplitude', 'Time_Periastron', 'Ecc_sin', 'Ecc_cos']

        '''
        pass


    fm_model_line['lw'] = 2
    dual_plot(D, DB_all_kep, pbar, savename='KeplerianModel')
    fm_model_line['lw'] = 3

    # PHASEFOLD
    for mode in range(True+switch_uncertain):
        # add a tail to the name if uncertainties
        if mode == 0:
            name_head = ''
            name_tail = ''
        if mode == 1:
            name_head = 'uncertainpy/'
            name_tail = '_uncertainties'
            pbar_tot = len(DB_all_kep)
            pbar = tqdm(total=pbar_tot)
        chaos_thetas = []
        nb_ = 0

        for b in DB_all_kep:
            per = b[0].value
            if b.parameterisation==2 or b.parameterisation==6:
                per = np.exp(b[0].value)
            D_PF = deepcopy(D)
            TB = [deepcopy(b)]                

            D_PF = fold_dataframe(D_PF, per=per)

            if True:
                dual_plot(D_PF, TB, pbar, savename=f'{name_head}{b.name_+name_tail}')
                # get uncertainties
                '''
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
                                chaos_holder = cp.Normal(mu=b[i].posterior.mixture_mean,
                                                        sigma=b[i].posterior.mixture_sigma)
                                
                                chaostheta[chaos_names[i]] = cp.Trunc(chaos_holder, lower=b[i].limits[0],
                                                                                    upper=b[i].limits[1])
                            else:
                                chaostheta[chaos_names[i]] = b[i].value_mean

                    #chaos_thetas.append(chaostheta)

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
                '''

                '''
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
                        axs.set_xlabel(f'BJD (days)',
                                       fontsize=fm_label_fs)
                        axs.set_ylabel('First-order Sobol indices',
                                       fontsize=fm_label_fs)
                        axs.legend(parameters.get_from_uncertain(),
                                            loc='upper right',
                                            framealpha=0.5)

                        figs.savefig(saveplace+f'{name_head}sobol_{b.name_+name_tail}.{plot_fmt}',
                                   bbox_inches='tight')
                        pl.close(figs)

                '''

            nb_ += 1

        pbar.close()
    temp_file_folder = saveloc+'/temp/models/'


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


def plot_KeplerianModel(model_=None, options={}):
    labels_ = model_.instrument_names_RV
    common_t = int(options['common_t'])

    if len(options):
        saveplace = options['saveloc']
        saveloc = saveplace + '/plots/models/'

        plot_fmt = options['format']

        switch_histogram = options['hist']
        switch_uncertain = options['uncertain']
        switch_periodogram = options['periodogram']
        switch_errors = options['errors']

        
        axhline_kwargs = {'color':'gray', 'linewidth':2}
        errorbar_kwargs_light = {'marker':'o', 'ls':'', 'alpha':0.4,
                                        'lw':1.2,
                                        'markersize':8,
                                        'markeredgewidth':1,
                                        'markeredgecolor':'k',
                                        }
        errorbar_kwargs_dark = {'marker':'o', 'ls':'', 'alpha':0.8,
                                        'lw':1.2,
                                        'markersize':8,
                                        'markeredgewidth':1,
                                        'markeredgecolor':'k',
                                        }
        continuum_kwargs = {'ls':'-', 'color':rc.fg, 'lw':1}

        fm_figsize = (10, 8)
        # HIST
        fm_hist = {'lw':2}  # 1
        fm_hist_tick_fs = 0
        # LEGEND
        fm_legend_fs = 14 # 10

        # FM_FRAME
        fm_frame_lw = 3
        # FM TICKS
        fm_tick_xsize = 20
        fm_tick_ysize = 20
        # LABELS
        fm_label_fs = 22
        # title
        fm_title_fs = 24
        
    def mk_model(data, blocks):
        temp_model = ReddModel(blocks,
                               data_RV=data)
        temp_model.refresh__()
        return temp_model
    

    def retrieve_model_func(model, tail=''):
        model.model_script_no = 0
        temp_script_name = f'temp_{tail}.py'
        temp_script_loc = f'{saveplace}/temp'

        dependencies = model.get_dependencies().tolist()
        constants = model.get_constants()

        with open(f'{temp_script_loc}/{temp_script_name}', 'w') as f:
            f.write(open(get_support('init_reddemcee.scr')).read())
            for d in dependencies:
                f.write(f'''
{d}''')
            f.write('''
''')
            for c in constants:
                f.write(f'''{c} = {constants[c]}
''')

            model_script_name = model.write_model(loc=saveplace, tail=tail)
            f.write(open(model_script_name).read())
        
        sys.path.insert(0, f'{saveplace}/temp')
        module = import_module(temp_script_name.split('.')[0])
        module = reload(module)
        temp_model_func = module.my_model
        sys.path.pop(0)
        return temp_model_func


    def mk_dual_plot(data, data_c, savename='', isfold=False):
        # make figure
        data1 = data
        if True:
            fig = pl.figure(figsize=fm_figsize)
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
            ax.axhline(0, **axhline_kwargs)
            axr.axhline(0, **axhline_kwargs)

            pl.subplots_adjust(wspace=0.15)

        # plot data
        if True:
            for n_ins in range(len(labels_)):
                mask = data1['Flag'] == (n_ins + 1)
                if switch_errors:
                    ax.errorbar(data1[mask]['BJD'],
                                data1[mask]['RV'],
                                data1[mask]['eRV'],
                                c=Color_Cycler[n_ins],
                                #label=labels_[n_ins],
                                **errorbar_kwargs_light)

                    axr.errorbar(data1[mask]['BJD'],
                                    data1[mask]['residuals'],
                                    data1[mask]['eRV'],
                                c=Color_Cycler[n_ins],
                                **errorbar_kwargs_light)

                    if switch_errors:
                        ax.errorbar(data1[mask]['BJD'],
                                    data1[mask]['RV'],
                                    data1[mask]['eRV_og'],
                                    c=Color_Cycler[n_ins],
                                    label=labels_[n_ins],
                                    **errorbar_kwargs_dark)
                        
                        axr.errorbar(data1[mask]['BJD'],
                                    data1[mask]['residuals'],
                                    data1[mask]['eRV_og'],
                                    c=Color_Cycler[n_ins],
                                    **errorbar_kwargs_dark)
                else:
                    #print(f'{data1[mask]['RV']=}')
                    ax.plot(data1[mask]['BJD'],
                            data1[mask]['RV'],
                            f'{Color_Cycler[n_ins]}o',
                            label=labels_[n_ins])

                    axr.plot(data1[mask]['BJD'],
                                data1[mask]['residuals'],
                                Color_Cycler[n_ins]+'o')   
        
        # continuum
        if True:
            ax.plot(data_c['BJD'],
                    data_c['RV'],
                    **continuum_kwargs)

        # histograms
        if True and switch_histogram:
            nbins = 5
            while nbins < len(data1):
                counts, bins = np.histogram(data1['RV'],
                                            bins=nbins)

                if (counts==0).any():
                    break
                else:
                    nbins += 1

            nbins = 5
            while nbins < len(data1):
                counts, bins = np.histogram(data1['residuals'],
                                            bins=nbins)
                if (counts==0).any():
                    break
                else:
                    nbins += 1

            # PLOT HISTOGRAMS
            axh.hist(data1['RV'],
                        bins=nbins-1, orientation='horizontal', ec=rc.fg, lw=fm_hist['lw'])
            axrh.hist(data1['residuals'],
                        bins=nbins-1, orientation='horizontal', ec=rc.fg, lw=fm_hist['lw'])
            # HIDE TICKS
            axh.tick_params(axis="x", labelbottom=False, labelsize=fm_tick_xsize)
            axh.tick_params(axis="y", labelleft=False)

            axrh.tick_params(axis="y", labelleft=False)
            axrh.tick_params(axis="x", labelsize=fm_tick_xsize)
            axrh.set_xlabel('Counts', fontsize=fm_label_fs)

        # Ticks and labels
        if True:
            ax.tick_params(axis="x", labelbottom=False)
            ax.tick_params(axis="y", labelsize=fm_tick_ysize)

            axr.tick_params(axis="x", labelsize=fm_tick_xsize)
            axr.tick_params(axis="y", labelsize=fm_tick_ysize)

            #axr.xaxis.set_major_locator(MaxNLocator(5))

            #ax.set_title('Keplerian Model', fontsize=fm_title_fs)
            ax.set_ylabel(r'RVs ($\frac{m}{s}$)', fontsize=fm_label_fs)
            ax.legend(fontsize=fm_legend_fs)#, framealpha=0)

            if common_t and not isfold:
                xlabel_sup = f'BJD (days) + {common_t}'
            else:
                xlabel_sup = 'BJD (days)'
                
            axr.set_xlabel(xlabel_sup, fontsize=fm_label_fs)
            axr.set_ylabel(r'Residuals ($\frac{m}{s}$)', fontsize=fm_label_fs)


        # SPINES
        if True:
            for spine in ax.spines.values():
                spine.set_linewidth(fm_frame_lw)
            for spine in axr.spines.values():
                spine.set_linewidth(fm_frame_lw)

            for spine in axh.spines.values():
                spine.set_linewidth(fm_frame_lw)
            for spine in axrh.spines.values():
                spine.set_linewidth(fm_frame_lw)

        fig.savefig(f'{saveloc}{savename}.{plot_fmt}',
                    bbox_inches='tight')


    def add_blocks_to_data(data0, blocks, tail=''):
        data = deepcopy(data0)
        model = mk_model(data, blocks)
        sol = model.get_attr_param('value', flat=True)

        rv, err = retrieve_model_func(model, tail=tail)(sol)
        data['RV'] += rv
        return data


    # GET ABSOLUTE RESIDUALS
    if True:
        model_og = deepcopy(model_)
        data_og = model_og.data
        sol_og = np.array(model_og.get_attr_param('value', flat=True))[model_og.C_]
        #sol_og = model_og.get_attr_param('value', flat=True)

        temp_model_func_ = retrieve_model_func(model_og, tail='full_model')

        rv0, err0 = temp_model_func_(sol_og)

        
        data_og['eRV_og'] = data_og['eRV'].copy()
        data_og['eRV'] = np.sqrt(err0)
        data_og['residuals'] = data_og['RV'] - rv0
        data_og['RV'] = data_og['residuals']


    # plot full model
    if True:
        blocks_keps = [b for b in model_og if b.display_on_data_==True]
        data_keps = add_blocks_to_data(data_og, blocks_keps, tail='AllKep')

        x_c = np.linspace(data_keps['BJD'].min(),
                          data_keps['BJD'].max(),
                          5000)
        data_c = pd.DataFrame({'BJD':x_c,
                               'RV':np.zeros_like(x_c),
                               'eRV':np.ones_like(x_c),
                               'eRV_og':np.ones_like(x_c),
                               'residuals':np.zeros_like(x_c),})
        
        data_keps_c = add_blocks_to_data(data_c, blocks_keps, tail='cont_AllKep')

        mk_dual_plot(data_keps, data_keps_c, savename='KeplerianModel')

    # plot phases
    if True:
        counter = 1
        for b in blocks_keps:
            per = b[0].value
            if b.parameterisation==2 or b.parameterisation==6:
                per = np.exp(b[0].value) 
            
            data_folded = fold_dataframe(deepcopy(data_og), per=per)
            data_folded = add_blocks_to_data(data_folded, [b], tail=f'FoldKep0{counter}')

            x_c = np.linspace(data_folded['BJD'].min(),
                              data_folded['BJD'].max(),
                              5000)
            data_c = pd.DataFrame({'BJD':x_c,
                                   'RV':np.zeros_like(x_c),
                                   'eRV':np.ones_like(x_c),
                                   'eRV_og':np.ones_like(x_c),
                                   'residuals':np.zeros_like(x_c),})
            data_keps_c = add_blocks_to_data(data_c, [b], tail=f'cont_FoldKep0{counter}')

            mk_dual_plot(data_folded, data_keps_c,
                         savename=f'KeplerianBlock {counter}',
                         isfold=True)
            counter += 1

    pass
    

def plot_periodogram(my_data, options, tail_name=''):
    from scipy.signal import lombscargle
    if options is None:
        options = {}

    saveplace = options['saveloc'] + '/plots/'
    plot_fmt = options['format']
    x = my_data['BJD'] - min(my_data['BJD']) + 0.1
    y = my_data['residuals']
    yerr = my_data['eRV']

    # other params
    Nfreq = 20000
    periods = np.linspace(x.min(), x.max()/2, Nfreq)
    ang_freqs = 2 * np.pi / periods

    maxpoints = 5
    ide = np.arange(maxpoints)+1

    TITLE = 'Residuals Periodogram'
    xaxis_label = 'Period (days)'
    yaxis_label = 'Power'
    title_fontsize = 'large'
    label_fontsize = 'medium'
    line_style = '-'
    line_color = rc.fg
    hsize, vsize = 10, 8
    dpi = 80
    xlog = True
    ylog = False

    scatter_marker_style = 'o'
    scatter_size = 40
    scatter_color = '#FF0000'
    scatter_alpha = 1

    method = 0
    t = np.ascontiguousarray(x.values)
    mag = np.ascontiguousarray(y.values)
    #dmag = np.ascontiguousarray(yerr.values)

    if method == 0:
        power = lombscargle(t, mag - np.mean(mag), ang_freqs)
        N = len(t)
        power *= 2 / (N * np.std(mag) ** 2)

    # Plot
    fig, ax = pl.subplots(figsize=(hsize, vsize), dpi=dpi)
    pl.title(TITLE, fontsize=title_fontsize)

    idx = getExtremePoints(power, typeOfExtreme='max', maxPoints=maxpoints)

    pl.plot(periods, power, ls=line_style, c=line_color)
    # fap line

    #ax.annotate('0.1% significance level', (3, 0.13))
    #pl.plot(periods, np.ones_like(periods)*0.12, ls='--', c=line_color, alpha=0.5)

    pl.scatter(periods[idx], power[idx],
                marker=scatter_marker_style,
                s=scatter_size, c=scatter_color,
                alpha=scatter_alpha)

    for i in idx:
        ax.annotate(f' Max = {np.round(periods[i], 2)}', (periods[i]+10, power[i]))

        ax.set_title(TITLE, fontsize=title_fontsize)
        ax.set_xlabel(xaxis_label, fontsize=label_fontsize)
        ax.set_ylabel(yaxis_label, fontsize=label_fontsize)

        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')

    #tabable = np.array([periods[idx][::-1], power[idx][::-1]])
    #taball = np.vstack([ide, tabable])
    #headers = ['Rank', 'Period', 'Power']

    #ax.table(cellText=taball.T, colLabels=headers, loc='top right')

    pl.savefig(f'{saveplace}periodogram{tail_name}.{plot_fmt}')
    pl.close(fig)


def make_block_plot(foo):
    if _PLATFORM_SYSTEM == 'Darwin':
        matplotlib.use('Agg')
    
    elif _PLATFORM_SYSTEM == 'Linux':
        matplotlib.use('Agg')

    plot_points, plot_args, index, pltd = foo

    if True:
        ptfmt = pltd['format']
        scatter_kwargs = pltd['scatter_kwargs']
        chain_kwargs = pltd['chain_kwargs']
        vlines_kwargs = pltd['vlines_kwargs']
        label_kwargs = pltd['label_kwargs']
        hexbin_kwargs = pltd['hexbin_kwargs']
        legend_kwargs = pltd['legend_kwargs']
        tick_params_kwargs = pltd['tick_params_kwargs']
        colorbar_kwargs = pltd['colorbar_kwargs']

        if pltd['paper_mode']:
            #pltd['fs_supt'] = 48#24
            #pltd['fs_supylabel'] = 44#22
            pltd['fs_xlabel'] = 28#14
            fm_frame_lw = 6#3
            pltd['figsize_xaxis'] = 20#10
        
        else:
            pltd['fs_supt'] = 24
            pltd['fs_supylabel'] = 22
            pltd['fs_xlabel'] = 14
            pltd['figsize_xaxis'] = 10

            fm_frame_lw = 3

            pl_scatter_alpha = 0.7
            pl_scatter_size = 10  #2
            fm_tick_xsize = 20  #20
            fm_tick_ysize = 20  #20
            plt_vlines_lw = 2#2
            pl_label_fs = 22#22



        ch1, lk0 = plot_points  # chains[t], likes[t]
        b, t = plot_args

        # plot_options as a dict: figsize_xaxis, fs_supt, fs_supylabel, fs_xlabel, pt_fmt
        XT_ = []
        XTL_ = []
        XL_ = []

        YT_ = []
        YTL_ = []
        YL_ = []
        for mode in pltd['modes']:
            # correct fig size for the iteration
            elongatex = 2.2 if mode != 0 else 0
            elongatey = 2 if b.ndim_ == 1 else 0.5
            #if mode == 1:
            #    elongatey += 5

            fig, axes = pl.subplots(b.ndim_,
                                    figsize=(pltd['figsize_xaxis'] + elongatex,
                                                        b.ndim_*6 + elongatey)
                                                        )
            # fig.suptitle(f'Posteriors {b.name_}', fontsize=pltd['fs_supt'])
            #fig.supylabel('Log Posterior', fontsize=pltd['fs_supylabel'])

            
            minl, maxl = min(lk0), max(lk0)

            for pi in range(b.ndim_):
                ch0 = ch1[:, b.cpointer[pi]]

                ax = axes if b.ndim_ == 1 else axes[pi]

                param = b[b.C_][pi]
                _param_value_max = param.value_max
                _param_value_mean = param.value_mean

                # plot on mode
                if mode == 0:
                    ax.scatter(ch0, lk0,
                               c=Color_Cycler[b.bnumber_-1],**scatter_kwargs)

                if mode == 1:
                    cmap = mk_cmap([Color_Cycler[b.bnumber_-1]], ncolors=100)

                    hb = ax.hexbin(ch0, lk0,
                                   cmap=cmap,
                                   **hexbin_kwargs)
                    
                    cb = fig.colorbar(hb, ax=ax, pad=0.02)
                    cb.set_label('log10(N)', **colorbar_kwargs)
                    
                if mode == 2:
                    # MAKE COLORMAP
                    cmap = mk_cmap([Color_Cycler[b.bnumber_-1]], ncolors=100)

                    # MAKE HEATMAP
                    gaussian_sigma = 8
                    img, extent = mini_heatmap(ch0, lk0, gaussian_sigma)

                    # ADJUST


                    # PLOT
                    imsh = ax.imshow(img, extent=extent,
                                     origin='lower', cmap=cmap,
                                     aspect='auto')
                    
                    cb = fig.colorbar(imsh, ax=ax, pad=0.02)
                    cb.set_label(f'Counts x {len(lk0)}',
                                 **colorbar_kwargs)

                    _param_value_max = (param.value_max - min(ch0)) / (max(ch0) - min(ch0)) * 4
                    _param_value_mean = (param.value_mean - min(ch0)) / (max(ch0) - min(ch0)) * 4
                    minl = 0
                    maxl = 1

                if mode == 3:
                    ax.plot(ch0.T,
                            color=pltd['colors'][b.bnumber_-1],
                            **chain_kwargs)
                    
                # ACCOMODATE TICKS
                if mode == 1 or mode == 2:
                    xticks_ = XT_[pi]
                    xticks_labels_ = XTL_[pi]
                    xticks_lims_ = XL_[pi]

                    yticks_ = YT_[pi]
                    yticks_labels_ = YTL_[pi]
                    yticks_lims_ = YL_[pi]

                    my_xticks = xticks_
                    my_yticks = yticks_
                    if mode == 2:
                        my_xticks = (xticks_ - xticks_lims_[0]) / (xticks_lims_[1] - xticks_lims_[0]) * (ax.get_xlim()[1] - ax.get_xlim()[0]) - ax.get_xlim()[0]
                        my_yticks = (yticks_ - yticks_lims_[0]) / (yticks_lims_[1] - yticks_lims_[0]) * (ax.get_ylim()[1] - ax.get_ylim()[0]) + ax.get_ylim()[0]

                    ax.set_xticks(my_xticks)
                    ax.set_xticklabels(xticks_labels_)

                    ax.set_yticks(my_yticks)
                    ax.set_yticklabels(yticks_labels_)
                    #ax.yaxis.set_tick_params(which="major")

                ax.tick_params(axis='x', labelrotation=45, **tick_params_kwargs)
                ax.tick_params(axis='y', **tick_params_kwargs)
                    

                # Draw max and mean
                if mode == 3:
                    ax.hlines(_param_value_max,
                                colors=rc.fg,
                                xmin=0,
                                xmax=len(ch0),
                                ls='-',
                                label=f'max = {np.round(param.value_max, 3)}',
                                **vlines_kwargs)
                    ax.hlines(_param_value_mean,
                                colors=rc.fg,
                                xmin=0,
                                xmax=len(ch0),
                                ls='-',
                                label=f'max = {np.round(param.value_mean, 3)}',
                                **vlines_kwargs)

                    
                    ax.set_ylabel(f'{param.name} {param.unit}',
                                  **label_kwargs)
                    
                    ax.set_xlabel(f'Steps',
                                  **label_kwargs)
                    
                    
                else:
                    ax.vlines(_param_value_max,
                                ymin=minl,
                                ymax=maxl,
                                colors=rc.fg,
                                ls='-',
                                label=f'max = {np.round(param.value_max, 3)}',
                                **vlines_kwargs)
                    ax.vlines(_param_value_mean,
                                ymin=minl,
                                ymax=maxl,
                                colors=rc.fg,
                                ls='--',
                                label=f'mean = {np.round(param.value_mean, 3)}',
                                **vlines_kwargs)
                
                    ax.set_xlabel(f'{param.name} {param.unit}',
                                  **label_kwargs)
                    ax.set_ylabel('Log P',
                                  **label_kwargs)

                ax.legend(**legend_kwargs)

                # SPINES
                if True:
                    for spine in ax.spines.values():
                        spine.set_linewidth(fm_frame_lw)

                if mode == 0:
                    XT_.append(deepcopy(ax.get_xticks()))
                    XTL_.append(deepcopy(ax.get_xticklabels()))
                    XL_.append(deepcopy(ax.get_xlim()))

                    YT_.append(deepcopy(ax.get_yticks()))
                    YTL_.append(deepcopy(ax.get_yticklabels()))
                    YL_.append(deepcopy(ax.get_ylim()))


            
            pl.tight_layout()
            
            
            
            mdname = pltd['modes_names'][mode]
            pl.savefig(pltd['saveloc']+f'/plots/posteriors/{mdname}/{t}_temp/{b.name_}.{ptfmt}',
                            bbox_inches='tight')

            pl.close()

    #del plot_points
    #del plot_args
    gc.collect()
    pass


def super_plots(chains=[], posts=[], options={}, my_model=None, ncores=None):
    modes_names = ['scatter', 'hexbin', 'gaussian', 'chains']

    saveplace = options['saveloc']
    temps = options['temps']

    options['colors'] = Color_Cycler
    options['modes_names'] = modes_names
    if ncores is None:
        ncores = _CORES

    # make folders
    for mode in options['modes']:
        for t in temps:
            try:
                os.makedirs(saveplace+f'/plots/posteriors/{modes_names[mode]}/{t}_temp')
            except:
                pass
        
    plot_pt_list = []
    plot_list = []
    for ti in temps:
        for b in my_model:
            if b.ndim_ > 0:
                plot_pt_list.append([chains[ti], posts[ti]])
                plot_list.append([b, ti])

    num_plots = len(plot_list)


    # TODO: enter this value through options
    ncores = 4
    tasks = []
    for i in range(num_plots):
        tasks.append([plot_pt_list[i],
                      plot_list[i],
                      i,
                      options])
        if (i+1) % ncores == 0:
            with Pool(ncores) as pool:
                for _ in tqdm(pool.imap_unordered(make_block_plot, tasks), total=len(tasks)):
                    pass

            pl.close('all')
            gc.collect()
            tasks = []
        
        if (i+1) == num_plots:
            with Pool(ncores) as pool:
                for _ in tqdm(pool.imap_unordered(make_block_plot, tasks), total=len(tasks)):
                    pass

            pl.close('all')
            gc.collect()
            tasks = []

        
    
    return


def plot_histograms(chains=[], posts=[], options={}, my_model=None, ncores=None):
    from scipy.stats import norm, skew, kurtosis, mode

    saveplace = options['saveloc']
    num_bins = 12
    dens_bool = True
    fs_supt = 48
    fs_supylabel = 44
    figsize_xaxis = 12

    pl_label_fs = 34

    fm_tick_xsize = 30
    fm_tick_ysize = 30
    ptfmt = options['format']
    temps = options['temps']

    label_kwargs = {'fontsize':pl_label_fs}
    # make folders
    for t in temps:
        try:
            os.makedirs(saveplace+f'/plots/histograms/{t}_temp')
        except:
            pass

    # plotting routine
    pbartot = np.sum([1 if b.ndim_ > 0 else 0 for b in my_model]) * len(temps)
    pbar = tqdm(total=pbartot)
    for ti in temps:
        for b in my_model:
            if True:
                if b.ndim_ > 0:
                    ch0 = chains[ti]
                    pt = posts[ti]

                    fig, axes = pl.subplots(b.ndim_,
                                        figsize=(figsize_xaxis,
                                                b.ndim_*6 + 1)
                                            )
                    fig.suptitle(f'Posteriors {b.name_}', fontsize=fs_supt)
                    fig.supylabel('Norm Density', fontsize=fs_supylabel)



                    for pi in range(b.ndim_):
                        param = b[b.C_][pi]
                        try:
                            ch = ch0[:, b.cpointer[pi]]
                            ax = axes if b.ndim_ == 1 else axes[pi]


                            mu, sigma = norm.fit(ch)
                            # first histogram of the data
                            n, bins = np.histogram(ch, bins=num_bins, density=dens_bool)

                            # Get the maximum and the data around it!!
                            maxi = ch[np.where(pt == np.amax(pt))][0]
                            dif = np.fabs(maxi - bins)
                            his_max = bins[np.where(dif == np.amin(dif))]

                            res = np.where(n == 0)[0]  # Find the zeros!!
                            if res.size:
                                if len(res) > 2:
                                    for j in range(len(res)):
                                        if res[j + 2] - res[j] == 2:
                                            sub = j
                                            break
                                else:
                                    sub = res[0]

                                # Get the data subset!!
                                if bins[sub] > his_max:
                                    pt = pt[np.where(ch <= bins[sub])]
                                    ch = ch[np.where(ch <= bins[sub])]
                                else:
                                    pt = pt[np.where(ch >= bins[sub])]
                                    ch = ch[np.where(ch >= bins[sub])]


                            # Get the maximum and the data around it!!
                            _param_value_max = param.value_max
                            _param_value_mean = param.value_mean

                            n, bins, patches = ax.hist(ch,
                                                    num_bins,
                                                    density=True,
                                                    facecolor=Color_Cycler[b.bnumber_-1],
                                                    edgecolor=rc.bg,
                                                    linewidth=3,
                                                    alpha=0.6,
                                                    )

                            mu, sigma = norm.fit(ch)
                            var = sigma**2.

                            # Some Stats!!
                            skew0 = np.round(skew(ch), 3)
                            kurt0 = np.round(kurtosis(ch), 3)
                            medi0 = np.round(np.median(ch), 3)
                            mode0 = np.round(mode(ch)[0], 3)

                            mu0 = np.round(mu, 3)
                            var0 = np.round(var, 3)


                            # Make a model x-axis!!
                            span = bins[len(bins) - 1] - bins[0]
                            bins_x = ((np.arange(num_bins * 100.) /
                                    (num_bins * 100.)) * span) + bins[0]

                            # Renormalised to the histogram maximum!!
                            y = np.exp(-np.power((bins_x - mu) / sigma, 2.) / 2.) * np.amax(n)

                            ax.plot(bins_x, y, rc.bg, linewidth=4)
                            ax.plot(bins_x, y, 'r', linewidth=3)

                            ax.set_xlabel(f'{param.name} {param.unit}',
                                        **label_kwargs)

                            ax.tick_params(axis='x', labelsize=fm_tick_xsize)
                            ax.tick_params(axis='y', labelsize=fm_tick_ysize)


                            # Get the axis positions!!
                            ymin, ymax = ax.get_ylim()
                            xmin, xmax = ax.get_xlim()

                            ax.text(xmax - (xmax - xmin) * 0.65, ymax - (ymax - ymin)
                                    * 0.1, r"$\mathcal{N}(\mu_1,\sigma^2,\mu_3,\mu_4)$", size=28)


                            left_fact = 0.9
                            right_fact = 0.4
                            textsize = 26
                            ax.text(xmax - (xmax - xmin) * left_fact, ymax - (ymax - ymin)
                                    * 0.180, r"$\mu_1 ={}$".format(mu0), size=textsize)
                            ax.text(xmax - (xmax - xmin) * left_fact, ymax - (ymax - ymin)
                                    * 0.265, r"$\sigma^2 ={}$".format(var0), size=textsize)
                            ax.text(xmax - (xmax - xmin) * left_fact, ymax - (ymax - ymin)
                                    * 0.350, r"$\mu_3 ={}$".format(skew0), size=textsize)


                            ax.text(xmax - (xmax - xmin) * right_fact, ymax - (ymax - ymin)
                                    * 0.180, r"$\mu_4 ={}$".format(kurt0), size=textsize)
                            ax.text(xmax - (xmax - xmin) * right_fact, ymax - (ymax - ymin)
                                    * 0.265, r"$Median ={}$".format(medi0), size=textsize)
                            ax.text(xmax - (xmax - xmin) * right_fact, ymax - (ymax - ymin)
                                    * 0.350, r"$Mode ={}$".format(mode0), size=textsize)

                        except Exception:
                            print(f'Failed to plot the {param.name} histogram')

                    pl.tight_layout()
                    pl.savefig(saveplace+f'/plots/histograms/{ti}_temp/{b.name_}.{ptfmt}',
                                bbox_inches='tight')

                    pl.close()
                    pbar.update(1)


    pbar.close()
    gc.collect()
    pass

#plot_histograms(sim.chain, sim.posts, sim.plot_histograms, sim.model)

def plot_betas(betas=[], logls=[], setup=[], Z=0, options={}):
    cmap = matplotlib.colormaps['plasma']
    colors = cmap(np.linspace(0, 0.85, setup[0]))
    # TODO check if this works

    if True:
        saveplace = options['saveloc']

        title_fs = options['title_fs']
        title_la = 'Temperature Ladder'

        xaxis_fs = options['xaxis_fs']
        xaxis_la = r'$\beta$'

        yaxis_fs = options['yaxis_fs']
        yaxis_la = r'$E[\log \mathcal{L}]_\beta$'

        fm_tick_xsize = 14
        fm_tick_ysize = 14
        ptfmt = options['format']

        pbar = tqdm(total=1)


    with contextlib.suppress(Exception):
        os.makedirs(saveplace + '/plots/betas')

    my_text = rf'Evidence: {np.round(Z[0], 3)} $\pm$ {np.round(Z[1], 3)}'

    if True:
        fig, ax = pl.subplots()
        for ti in range(setup[0]):
            bet = betas[ti]
            ax.plot(bet, np.ones_like(bet)*logls[ti], color=colors[ti], alpha=0.7)
            ax.plot(bet[-1], logls[ti], color=colors[ti], marker='o')

        ylims = ax.get_ylim()
        
        betas0 = [x[-1] for x in betas]
        ax.fill_between(betas0, logls,
                        y2=0,
                        color=rc.fg,
                        #color='w',
                        alpha=0.25)
        
        ax.set_ylim(ylims)

    if True:
        ax.scatter([], [], alpha=0, label=my_text)
        pl.legend(loc=4)
        ax.set_xlabel(xaxis_la, fontsize=xaxis_fs)
        ax.set_ylabel(yaxis_la, fontsize=yaxis_fs)
        
        ax.set_xlim([0, 1])
        pl.tight_layout()
        pl.savefig(f'{saveplace}/plots/betas/beta_ladder.{ptfmt}',
                                    bbox_inches='tight')
        pl.close()

    pbar.update(1)
    pbar.close()


def plot_rates(betas=[], tsw=[], smd=[], setup=[], options={}, run_config={}):
    if True:
        saveplace = options['saveloc']
        ptfmt = options['format']
        pbar = tqdm(total=setup[0]-1)
        cmap = matplotlib.colormaps['plasma']
        colors = cmap(np.linspace(0, 0.85, setup[0]))
        window = options['window']
        adapt_batches = run_config['adaptation_batches']
        adapt_nsweeps = run_config['adaptation_nsweeps']


    with contextlib.suppress(Exception):
        os.makedirs(saveplace + '/plots/betas')

    if True:
        fig, axes = pl.subplots(3, 1, figsize=(10, 7), sharex=True)

        #x0 = np.arange(setup[2]) * setup[3]

        for t in range(setup[0]-1):
            # PLOT BETAS
            y_bet = betas[t]
            axes[0].plot(1/y_bet, c=colors[t])

            # PLOT TS_ACCEPTANCE

            if False:#adapt_batches:
                x0 = np.arange(setup[2] + adapt_batches*adapt_nsweeps) * setup[3]
            else:
                x0 = np.arange(setup[2]) * setup[3]
            
            y_tsw = running(tsw[:, t], window)
            axes[1].plot(x0, y_tsw, alpha=0.75, color=colors[t])

            # PLOT SWAP MEAN DISTANCE
            y_smd = running(smd[:, t], window)
            axes[2].plot(x0, y_smd, alpha=0.75, color=colors[t])

            pbar.update(1)


        if True:
            axes[0].set_ylabel(r"$T$")
            axes[1].set_ylabel(r"$T_{swap}$")
            axes[2].set_ylabel(r"$SMD$")

            axes[2].set_xlabel("N Step")

            axes[0].set_xscale('log')
            axes[0].set_yscale('log')

        pl.tight_layout()
        pl.savefig(saveplace+f'/plots/betas/rates.{ptfmt}',
                                        bbox_inches='tight')
        pl.close()
        
        pbar.close()


def plot_beta_density(betas=[], options={}):
    saveplace = options['saveloc']
    ptfmt = options['format']
    pbar = tqdm(total=1)

    fig, axes = pl.subplots(figsize=(4, 3))
    y = -1/np.diff(np.log(betas))
    max_index = np.argmax(y)
    x_position = (betas[max_index] + betas[max_index + 1]) / 2

    pl.step(betas[:-1], y,
            color=rc.fg,
            where='post',
            )
    pl.axvline(x=x_position, color='r', linestyle='--',
               label=fr'$\beta={x_position:.3f}$')


    pl.gca().invert_xaxis()
    axes.set_xscale('log')
    axes.set_xlabel(r"$\beta$")
    axes.set_ylabel(r"$\eta(\beta)$")
    pl.legend()
    pl.savefig(saveplace+f'/plots/betas/density.{ptfmt}',
                                        bbox_inches='tight')
    pl.close()
    pbar.update(1)
    pbar.close()

#plot_betas(sim.sampler.betas_history, elogl, sim.evidence, options=opts, temps=setup[0])


def plot_astrometry1(options={}):
    pass


def plot_periodogram_new(my_data, options={}):
    from astropy.timeseries import LombScargle
    if True:
        Nfreq = 100000
        min_per = 0.5
        max_per = 10000

        Npoints = 10
        Npoints_disp = 5
    
    saveplace = options['saveloc']
    ptfmt = options['format']

    if False:
        Nfreq = options['Nfreq']
        min_per = options['min_per']
        max_per = options['max_per']

        Npoints = options['Npoints']
        Npoints_disp = options['Npoints_disp']

    work_df = my_data.copy()
    work_df['Window'] = 1

    YAXIS_ = list(work_df.columns)
    YAXIS_.remove('BJD')
    YAXIS_.remove('Flag')
    labels_ = YAXIS_.copy()

    nplots = len(YAXIS_)
    fig, axes = pl.subplots(nplots, figsize=(10, 2*nplots), sharex=True)
    t = work_df['BJD'].values

    for i in range(nplots):
        if True:
            yaxis_ = YAXIS_[i]
            mag = work_df[yaxis_].values
            
            periods = np.linspace(min_per, max_per, Nfreq)
            ang_freqs = 1 / periods

            ls = LombScargle(t, mag, fit_mean=True, center_data=True)  # astropy

            if yaxis_ == 'Window':
                ls = LombScargle(t, mag, fit_mean=False, center_data=False)  # scipy, astropy
            power = ls.power(ang_freqs,
                            method='auto',
                            #normalization='model',
                            )

            idx = getExtremePoints(power, 'max')[-Npoints:][::-1]
            # FAP LINES
            probabilities = [0.10, 0.01, 0.001]
            power_rows = 0
            if yaxis_ != 'Window':
                fap = ls.false_alarm_level(probabilities, method='baluev')

                relevantpl = idx[(fap[0] >= power[idx])]
                relevant10 = idx[(fap[1] > power[idx])&(power[idx] > fap[0])]
                relevant5 = idx[(fap[2] >= power[idx])&(power[idx] >= fap[1])]
                relevant1 = idx[power[idx] > fap[2]]

                celltext = []
                cellcolours = []

                counter = 0
                for relevant in [relevant1, relevant5, relevant10, relevantpl]:
                    for j in relevant:
                        if counter < Npoints_disp:
                            celltext.append([np.round(periods[j], 1), np.round(power[j], 2)])
                            power_rows += 1
                            counter += 1

            else:
                celltext = []
                counter = 0
                for j in idx:
                    if counter < Npoints_disp:
                        celltext.append([np.round(periods[j], 1), np.round(power[j], 2)])
                        counter += 1
                power_rows=5
            if celltext == []:
                celltext = [['', '']]


            collabels = ['Period', 'Power']
            if power_rows:
                axes[i].set
                the_table = axes[i].table(cellText=celltext,
                            #rowLabels=['label1'],
                            #rowColours=colors,
                            colLabels=collabels,
                            colWidths=[0.125, 0.125],
                            #cellColours=cellcolours,
                            #loc='right',
                            bbox = [1, 1-(1+power_rows)/7, 0.25, (1+power_rows)/7], 
                            fontsize=3)

        if True:
            axes[i].plot(periods, power,
                    ls='-', c='k')

            # ADD VSPANS
            #axes[i].axvspan(2403-218, 2403+266,
            #                color='C1')
            #axes[i].axvspan(27.4-3.2, 27.4+3.2,
            #                color='C4')

            #print(periods[idx][:10])

            if yaxis_ == 'Window':
                axes[i].scatter(periods[idx],
                        power[idx],
                        marker='o',
                        color='C2',
                        alpha=0.9,
                        s=35)           
            else:
                axes[i].scatter(periods[relevantpl],
                        power[relevantpl],
                        marker='o',
                        color='gray',
                        alpha=0.9,
                        s=35)

                axes[i].scatter(periods[relevant10],
                        power[relevant10],
                        marker='o',
                        color='C0',
                        alpha=0.9,
                        s=35)

                axes[i].hlines(fap,
                        xmin=[0],
                        xmax=[1],
                        transform=axes[i].get_yaxis_transform(),
                        color=['C0', 'C3', 'C2'],
                        ls=[':', ':', '--'])


                axes[i].scatter(periods[relevant5],
                        power[relevant5],
                        marker='o',
                        color='C3',
                        alpha=0.9,
                        s=35)

                axes[i].scatter(periods[relevant1],
                        power[relevant1],
                        marker='o',
                        color='C2',
                        alpha=1,
                        s=35)


            # AXES
            if True:
                axes[i].set_xscale('log')
                asd = True if i == (nplots-1) else False
                axes[i].tick_params(axis="x", which="both",
                            top=True, labeltop=False,
                            bottom=True, labelbottom=asd,
                            direction="in",
                            labelsize=20
                            #length=3,
                            #width=2,
                            )

                axes[i].tick_params(axis="y", which="both",
                            top=True, labeltop=False,
                            bottom=True, labelbottom=False,
                            direction="in",
                            labelsize=20
                            #length=3,
                            #width=2,
                            )

                axes[i].tick_params(axis="both", which="major",
                                    size=8,
                                    width=1.5)  # default:6; 0.8
                
                axes[i].tick_params(axis="both", which="minor",
                                    size=6,
                                    width=1.2)  # default:3; 0.4

                axes[i].annotate(f'{labels_[i]}',
                                xy=(.025, .8),
                                xycoords='axes fraction',
                                fontsize=20,
                                )

                axes[i].set_xlim([min_per, max_per])
                
                for spine in axes[i].spines.values():
                    spine.set_linewidth(3)
                    spine.set_color('k')

    axes[i].set_xlabel(r'Period (days)', fontsize=22)
    fig.supylabel('Power', fontsize=22)
    pl.subplots_adjust(hspace=0.0)
    #pl.savefig('last_periodogram.pdf', bbox_inches='tight', pad_inches=0.2)
    pl.savefig(saveplace+f'/plots/periodogram.{ptfmt}',
               bbox_inches='tight',
               pad_inches=0.2,
               )
#plot_periodogram_new(sim.my_data, options=sim.plot_posteriors)

def plot_correlogram(my_data, options={}):
    from scipy.stats import pearsonr, spearmanr, kendalltau, linregress
    if True:
        plot_scale = 2
    
    saveplace = options['saveloc']
    ptfmt = options['format']

    if False:
        Nfreq = options['Nfreq']
        min_per = options['min_per']
        max_per = options['max_per']

        Npoints = options['Npoints']
        Npoints_disp = options['Npoints_disp']

    work_df = my_data.copy()

    for ins in np.unique(my_data['Flag']):
        work_df = my_data[sim.my_data['Flag']==ins].copy()
        for col in work_df:
            if np.all(work_df[col]==0):
                work_df = work_df.drop(columns=col)
        work_df = work_df.drop(columns=['eRV', 'Flag'])

        columns = list(work_df.columns)    
        columns.remove('BJD')
        labels_ = columns.copy()
        rev = columns[::-1]  # ['D','C','B','A']

        ncols = len(columns)
        colors = work_df['BJD'].values

        fig, axes = pl.subplots(nrows=ncols, ncols=ncols, figsize=(plot_scale*ncols,
                                                                plot_scale*ncols))
        
        uni=True
        for i in range(ncols):
            for j in range(i + 1):
                if j<=i:
                    # i corresponds to the "row" (top to bottom)
                    # j corresponds to the "column" (left to right)
                    xcol = rev[j]
                    ycol = rev[i]
                    ax = axes[i, j]
                    diag = True if i==j else False

                    y_ = work_df[ycol]
                    x_ = work_df[xcol]
                    if diag:
                        xcol = 'BJD'
                        x_ = work_df[xcol]
                        #ax.axhline(y_.mean(),
                        #           color='k',
                        #           ls='--')
                        nbins = 5
                        brc, brc_tol = 0, 5
                        while nbins < len(y_):
                            counts, bins = np.histogram(y_, bins=nbins)
                            if (counts==0).any():
                                nbins += 1
                                brc += 1
                            else:
                                nbins += 1
                                brc = 0
                            if brc == brc_tol:
                                nbins -= brc_tol
                                break
                        ax.hist(y_,
                                bins=nbins-1,
                                #orientation='horizontal',
                                ec='k',
                                lw=1)
                        
                        #ax.annotate(rf'$\rho = {np.round(r, 2)}$',
                        #            xy=(1.025, .850),
                        #            xycoords='axes fraction',
                        #            )

                    else:
                        slope, intercept, r, p, stderr = linregress(x_, y_)
                        ax.axline((x_[0], intercept + slope * x_[0]),
                                slope=slope, color='k',
                                label=rf'$\rho$ = {np.round(r, 2)}',)

                        sc = ax.scatter(x_, work_df[ycol],
                                c=colors, marker='o', cmap='winter',
                                lw=1, ec='k',
                                s=20, alpha=1)
                        if uni:
                            uni = False

                    if not diag:
                        ax.legend(fontsize=12, markerscale=0,
                                framealpha=0,
                                handlelength=0)
                    #print(xcol, ycol, i, j)
                    #ax.xaxis.set_tick_params(which='minor', size=6, width=1.5, direction='in', top='on')


                if True:
                    if True:
                        ax.xaxis.set_tick_params(which='major', size=8, width=1.5, direction='in',
                                                        top=True, labeltop=False,
                                                        bottom=True, labelbottom=False,
                                                        right=True, labelright=False,
                                                        left=True, labelleft=False,)        

                        ax.xaxis.set_tick_params(which='minor', size=5, width=1.5, direction='in',
                                                        top=True, labeltop=False,
                                                        bottom=True, labelbottom=False,
                                                        right=True, labelright=False,
                                                        left=True, labelleft=False,)   
                        if not diag:
                            ax.yaxis.set_tick_params(which='major', size=8, width=1.5, direction='in',
                                                            top=True, labeltop=False,
                                                            bottom=True, labelbottom=False,
                                                            right=True, labelright=False,
                                                            left=True, labelleft=False,) 
                

                            ax.yaxis.set_tick_params(which='minor', size=5, width=1.5, direction='in',
                                                            top=True, labeltop=False,
                                                            bottom=True, labelbottom=False,
                                                            right=True, labelright=False,
                                                            left=True, labelleft=False,) 
                if j==0:
                    #print(ncols, xcol, ycol, i, j)
                    if not diag:
                        ax.yaxis.set_tick_params(which='major', size=8, width=1.5, direction='in',
                                                top=True, labeltop=False,
                                                bottom=True, labelbottom=True,
                                                right=True, labelright=False,
                                                left=True, labelleft=True,
                                                labelsize=14)
                    
                    ax.set_ylabel(labels_[::-1][i], fontsize=20)

                if i==(ncols-1):
                    ax.xaxis.set_tick_params(which='major', size=8, width=1.5, direction='in',
                                            top=True, labeltop=False,
                                            bottom=True, labelbottom=True,
                                            right=True, labelright=False,
                                            left=True, labelleft=False,
                                            labelsize=14,
                                            labelrotation=45)
                    #lab ='BJD' if j==(ncols-1) else labels_[::-1][j]
                    lab = labels_[::-1][j]
                    ax.set_xlabel(lab, fontsize=20)

                if diag:
                    ax.yaxis.set_tick_params(which='major', size=8, width=1.5, direction='in',
                                            top=True, labeltop=False,
                                            bottom=True, labelbottom=False,
                                            right=True, labelright=True,
                                            left=True, labelleft=False,
                                            labelsize=14)
                    
                    ax.yaxis.tick_right()

                for spine in ax.spines.values():
                    spine.set_linewidth(3)
                    spine.set_color('k')


        # Hide the empty subplots (those where j > i)
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                axes[i, j].set_visible(False)

        cb = pl.colorbar(sc, ax=axes, fraction=0.035, pad=1.5, anchor=(0.3, 0.9),
                        label='BJD')
        cb.ax.tick_params(labelsize=14)
        cb.outline.set_color('k')
        cb.outline.set_linewidth(2)

        pl.subplots_adjust(hspace=0.0)
        pl.subplots_adjust(wspace=0.0)
        fig.subplots_adjust(right=0.85)

        pl.savefig(saveplace+f'/plots/correlogram{ins}.{ptfmt}',
                bbox_inches='tight',
                pad_inches=0.2,
                )

#plot_correlogram(sim.my_data, options=sim.plot_posteriors)


def activate_paper_mode():
    # TITLE
    matplotlib.rcParams['font.size'] = 22

    # AXIS
    matplotlib.rcParams['axes.labelsize'] = 22

    # FRAME
    matplotlib.rcParams['axes.linewidth'] = 3

    # errorbar
    matplotlib.rcParams['lines.markersize'] = 7


#
    
'''
opts = {'plot':True,
        'format':'png',
        'logger_level':'ERROR',
        'saveloc':sim.saveplace,
        'title_fs':24,
        'xaxis_fs':18,
        'yaxis_fs':18,
        'paper_mode':True,
        }

reddemcee_dict = {'discard':sim.reddemcee_discard,
                    'thin':sim.reddemcee_thin,
                    'flat':True}
ch = sim.sampler.get_func('get_chain', kwargs=reddemcee_dict)
lk = sim.sampler.get_func('get_blobs', kwargs=reddemcee_dict)
pt = sim.sampler.get_func('get_log_prob', kwargs=reddemcee_dict)


if sim.cherry['cherry']:
    for t in range(setup[0]):
        if sim.cherry['median']:
            mask = pt[t] > np.median(pt[t])
        elif sim.cherry['diff']:
            mask = max(pt[t]) - pt[t] <= sim.cherry['diff']

        ch[t] = ch[t][mask]
        lk[t] = lk[t][mask]
        pt[t] = pt[t][mask]

'''