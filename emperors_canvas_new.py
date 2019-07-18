from __future__ import division, print_function

import copy
import os
from decimal import Decimal  # histograms

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy as sp
from scipy.stats import norm
from tqdm import tqdm

import corner
import emperors_library as emplib
import emperors_mirror as empmir


class CourtPainter:

    markers = ['o', 'v', '^', '>', '<', '8', 's', 'p', 'H', 'D', '*', 'd']
    error_kwargs = {'lw': 1.75, 'zorder': 0}
    chain_titles = sp.array(
        [
            'Period', 'Amplitude', 'Phase', 'Eccentricity', 'Longitude',
            'Acceleration', 'Jitter', 'Offset'
        ]
    )
    chain_units = [
        ' [Days]', r' $[\frac{m}{s}]$',  r' $[rad]$', '', r' $[rad]$',
        r' $[\frac{m}{s^2}]$'
    ]

    def __init__(self, setup, kplanets, working_dir, pdf, png):
        self.ntemps, self.nwalkers, self.nsteps = setup
        self.kplanets = kplanets
        self.working_dir = working_dir
        self.pdf = pdf
        self.png = png

        # Read chains, posteriors and data for plotting.
        self.chains = emplib.read_chains(working_dir + 'chains.pkl')
        self.cold = self.chains[0]
        self.posteriors = emplib.read_posteriors(
            working_dir + 'posteriors.pkl')
        self.all_rv = emplib.read_rv_data(working_dir + 'rv_data.pkl')
        self.time, self.rv, self.err, self.ins = self.all_rv

        self.nins = len(sp.unique(self.ins))
        self.ndim = 1 + 5 * kplanets + self.nins * 2

        self.__clean_rvs()
        # Setup plots.
        self.__read_config()
        self.time_cb = copy.deepcopy(self.time) - 2450000

        # Create directories.
        dirs = ['chains', 'posteriors', 'histograms']
        print('\nCREATING SHOWROOMS.')
        for d in dirs:
            path = self.working_dir + d
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the showroom %s failed" % path)
            else:
                print("Successfully created the showroom %s " % path)
        pass

    def __get_params(self, kplanet):
        """Retrieve model parameters."""
        period = sp.median(self.cold[:, 5 * kplanet])
        amplitude = sp.median(self.cold[:, 5 * kplanet + 1])
        phase = sp.median(self.cold[:, 5 * kplanet + 2])
        eccentricity = sp.median(self.cold[:, 5 * kplanet + 3])
        longitude = sp.median(self.cold[:, 5 * kplanet + 4])
        params = (period, amplitude, phase, eccentricity, longitude)
        return params

    def __get_CI_params(self, kplanet, alpha):
        """Retrieve model credibility interval for a given alpha."""
        _, period_lo, period_up = emplib.credibility_interval(
            self.cold[:, 5 * kplanet], alpha)
        _, amplitude_lo, amplitude_up = emplib.credibility_interval(
            self.cold[:, 5 * kplanet + 1], alpha)
        _, phase_lo, phase_up = emplib.credibility_interval(
            self.cold[:, 5 * kplanet + 2], alpha)
        _, eccentricity_lo, eccentricity_up = emplib.credibility_interval(
            self.cold[:, 5 * kplanet + 3], alpha)
        _, longitude_lo, longitude_up = emplib.credibility_interval(
            self.cold[:, 5 * kplanet + 4], alpha)
        params_lo = (period_lo, amplitude_lo, phase_lo,
                     eccentricity_lo, longitude_lo)
        params_up = (period_up, amplitude_up, phase_up,
                     eccentricity_up, longitude_up)
        return params_lo, params_up

    def __rv_residuals(self):
        """Calculate model residuals."""
        model = 0.
        for k in range(self.kplanets):
            params = self.__get_params(k)
            model += empmir.mini_RV_model(params, self.time)
        residuals = self.rv0 - model
        return residuals

    def __clean_rvs(self):
        """Clean radial-velocities by adding the offset and jitter."""
        instrumental = self.cold[:, -2 * self.nins:]
        rv0 = copy.deepcopy(self.rv)
        err0 = copy.deepcopy(self.err)
        acc = sp.median(self.cold[:, -2 * self.nins - 1])
        for i in range(self.nins):
            jitter = sp.median(instrumental[:, i])
            offset = sp.median(instrumental[:, i + 1])
            ins = self.ins == i
            # Assume linear acceleration for now.
            rv0[ins] -= offset + acc
            err0[ins] = sp.sqrt(err0[ins] ** 2 + jitter ** 2)
        self.rv0 = rv0
        self.err0 = err0
        pass

    def paint_fold(self):
        """Create phasefold plot."""
        print('\nPAINTING PHASE FOLDS.')
        # Get globbal max and min for plots
        minx, maxx = self.time.min(), self.time.max()
        cmin, cmax = self.time_cb.min(), self.time_cb.max()

        for k in tqdm(range(self.kplanets)):
            params = self.__get_params(k)

            fig = plt.figure(figsize=self.phase_figsize)
            gs = gridspec.GridSpec(3, 4)
            ax = fig.add_subplot(gs[:2, :])
            ax_r = fig.add_subplot(gs[-1, :])
            cbar_ax = fig.add_axes([.85, .125, .015, .755])
            fig.subplots_adjust(right=.84, hspace=0)

            for i in range(self.nins):  # plot per instrument.
                ins = self.ins == i
                t_p, rv_p, err_p = emplib.phasefold(
                    self.time[ins], self.rv0[ins], self.err0[ins], params[0]
                )
                _, res_p, _p = emplib.phasefold(
                    self.time[ins], self.__rv_residuals()[ins], self.err0[ins],
                    params[0]
                )
                # phasefold plot.
                ax.errorbar(
                    t_p, rv_p, yerr=err_p, linestyle='', marker=None,
                    alpha=.75, ecolor=self.error_color, **self.error_kwargs
                )
                im = ax.scatter(
                    t_p, rv_p, marker=self.markers[i], edgecolors='k',
                    s=self.phase_size, c=self.time_cb[ins],
                    cmap=self.phase_cmap
                )
                im.set_clim(cmin, cmax)
                ax_r.errorbar(
                    t_p, res_p, yerr=err_p, linestyle='', marker=None,
                    ecolor=self.error_color, **self.error_kwargs
                )
                im_r = ax_r.scatter(
                    t_p, res_p, marker=self.markers[i], edgecolors='k',
                    s=self.phase_size, c=self.time_cb[ins],
                    cmap=self.phase_cmap
                )
                im_r.set_clim(cmin, cmax)
            fig.colorbar(
                im, cax=cbar_ax).set_label(
                'JD - 2450000', rotation=270, labelpad=self.cbar_labelpad,
                fontsize=self.label_fontsize, fontname=self.fontname
            )

            time_m = sp.linspace(self.time.min() - 10,
                                 self.time.max() + 10, 10000)
            rv_m = empmir.mini_RV_model(params, time_m)
            time_m_p, rv_m_p, _ = emplib.phasefold(
                time_m, rv_m, sp.zeros(10000), params[0])

            # Plot best model.
            ax.plot(time_m_p, rv_m_p, '-k', linewidth=2)
            # Plot models CI.
            cred_intervals = [.99, .95, .68]  # 3, 2, and 1 sigma
            for s in cred_intervals:
                params_lo, params_up = self.__get_CI_params(k, s)
                # Calculate new models.
                rv_m_lo = empmir.mini_RV_model(params_lo, time_m)
                rv_m_up = empmir.mini_RV_model(params_up, time_m)
                _, rv_m_lo_p, _ = emplib.phasefold(
                    time_m, rv_m_lo, sp.zeros(10000), params_lo[0])
                _, rv_m_up_p, _ = emplib.phasefold(
                    time_m, rv_m_up, sp.zeros(10000), params_up[0])
                ax.fill_between(time_m_p, rv_m_lo_p, rv_m_up_p,
                                color=self.CI_color, alpha=.25)

            # A line to guide the eye.
            ax_r.axhline(0, color='k', linestyle='--', linewidth=2, zorder=0)

            # Labels and tick stuff.
            ax.set_ylabel(
                r'Radial Velocity (m s$^{-1}$)', fontsize=self.label_fontsize,
                fontname=self.fontname
            )
            ax_r.set_ylabel(
                'Residuals', fontsize=self.label_fontsize,
                fontname=self.fontname
            )
            ax_r.set_xlabel(
                'Phase', fontsize=self.label_fontsize, fontname=self.fontname
            )

            ax_r.get_yticklabels()[-1].set_visible(False)
            ax_r.minorticks_on()
            ax.set_xticks([])
            ax.tick_params(
                axis='both', which='major',
                labelsize=self.tick_labelsize
            )
            ax_r.tick_params(
                axis='both', which='major',
                labelsize=self.tick_labelsize
            )
            for tick in ax.get_yticklabels():
                tick.set_fontname(self.fontname)
            for tick in ax_r.get_yticklabels():
                tick.set_fontname(self.fontname)
            for tick in ax_r.get_xticklabels():
                tick.set_fontname(self.fontname)
            for tick in cbar_ax.get_yticklabels():
                tick.set_fontname(self.fontname)
            cbar_ax.tick_params(labelsize=self.tick_labelsize)

            ax.set_xlim(-.01, 1.01)
            ax_r.set_xlim(-.01, 1.01)
            if self.pdf:
                fig.savefig(self.working_dir + 'phase_fold_' +
                            str(k + 1) + '.pdf', bbox_inches='tight')
            if self.png:
                fig.savefig(self.working_dir + 'phase_fold_' +
                            str(k + 1) + '.png', bbox_inches='tight')

        pass

    def paint_timeseries(self):
        """Create timeseries plot."""
        print('\nPAINTING TIMESERIES.')
        # Get globbal max and min for plots
        minx, maxx = self.time.min(), self.time.max()
        cmin, cmax = self.time_cb.min(), self.time_cb.max()

        for k in tqdm(range(self.kplanets)):
            params = self.__get_params(k)

            fig = plt.figure(figsize=self.full_figsize)
            gs = gridspec.GridSpec(3, 4)
            ax = fig.add_subplot(gs[:2, :])
            ax_r = fig.add_subplot(gs[-1, :])
            cbar_ax = fig.add_axes([.85, .125, .015, .755])
            fig.subplots_adjust(right=.84, hspace=0)

            for i in range(self.nins):
                ins = self.ins == i

                ax.errorbar(
                    self.time[ins] - 2450000, self.rv0[ins],
                    yerr=self.err0[ins], linestyle='', marker=None,
                    ecolor=self.error_color, **self.error_kwargs
                )
                im = ax.scatter(
                    self.time[ins] - 2450000, self.rv0[ins],
                    marker=self.markers[i], edgecolors='k', s=self.full_size,
                    c=self.time_cb[ins], cmap=self.full_cmap
                )
                im.set_clim(cmin, cmax)

                # Get residuals.
                res = self.__rv_residuals()[ins]

                ax_r.errorbar(
                    self.time[ins] - 2450000, res, yerr=self.err0[ins],
                    linestyle='', marker=None, ecolor=self.error_color,
                    **self.error_kwargs
                )
                im_r = ax_r.scatter(
                    self.time[ins] - 2450000, res, marker=self.markers[i],
                    edgecolors='k', s=self.full_size, c=self.time_cb[ins],
                    cmap=self.full_cmap
                )

                im_r.set_clim(cmin, cmax)
            fig.colorbar(
                im, cax=cbar_ax).set_label(
                'JD - 2450000', rotation=270, labelpad=self.cbar_labelpad,
                fontsize=self.label_fontsize, fontname=self.fontname
            )
            time_m = sp.linspace(self.time.min() - 10,
                                 self.time.max() + 10, 10000)
            time_m -= 2450000
            rv_m = empmir.mini_RV_model(params, time_m)

            # Plot best model.
            ax.plot(time_m, rv_m, '-k', linewidth=2)

            # Plot models CI.
            cred_intervals = [.99, .95, .68]  # 3, 2, and 1 sigma
            for s in cred_intervals:
                params_lo, params_up = self.__get_CI_params(k, s)
                params_lo = (params[0], params_lo[1],
                             params_lo[2], params_lo[3], params_lo[4])
                params_up = (params[0], params_up[1],
                             params_up[2], params_up[3], params_up[4])
                # Calculate new models.
                rv_m_lo = empmir.mini_RV_model(params_lo, time_m)
                rv_m_up = empmir.mini_RV_model(params_up, time_m)

                ax.fill_between(
                    time_m, rv_m_lo, rv_m_up, color=self.CI_color, alpha=.25
                )

            # A line to guide the eye.
            ax_r.axhline(0, color='k', linestyle='--', linewidth=2, zorder=0)

            # Labels and tick stuff.
            ax.set_ylabel(
                r'Radial Velocity (m s$^{-1}$)', fontsize=self.label_fontsize,
                fontname=self.fontname
            )
            ax_r.set_ylabel(
                'Residuals', fontsize=self.label_fontsize,
                fontname=self.fontname
            )
            ax_r.set_xlabel(
                'Time (JD - 2450000)', fontsize=self.label_fontsize,
                fontname=self.fontname
            )

            ax_r.get_yticklabels()[-1].set_visible(False)
            ax_r.minorticks_on()
            ax.set_xticks([])
            ax.tick_params(
                axis='both', which='major',
                labelsize=self.tick_labelsize
            )
            ax_r.tick_params(
                axis='both', which='major',
                labelsize=self.tick_labelsize
            )
            for tick in ax.get_yticklabels():
                tick.set_fontname(self.fontname)
            for tick in ax_r.get_yticklabels():
                tick.set_fontname(self.fontname)
            for tick in ax_r.get_xticklabels():
                tick.set_fontname(self.fontname)
            for tick in cbar_ax.get_yticklabels():
                tick.set_fontname(self.fontname)
            cbar_ax.tick_params(labelsize=self.tick_labelsize)

            offset = (time_m.max() - time_m.min()) * .01
            ax.set_xlim(time_m.min() - offset, time_m.max() + offset)
            ax_r.set_xlim(time_m.min() - offset, time_m.max() + offset)
            if self.pdf:
                fig.savefig(self.working_dir + 'timeseries_' +
                            str(k + 1) + '.pdf', bbox_inches='tight')
            if self.png:
                fig.savefig(self.working_dir + 'timeseries_' +
                            str(k + 1) + '.png', bbox_inches='tight')
        pass

    def paint_chains(self):
        """Create traceplots or chain plots for each temperature."""
        print('\nPAINTING CHAINS.')
        for t in tqdm(range(self.ntemps), desc='Brush temperature'):
            chain = self.chains[t]

            leftovers = len(chain) % self.nwalkers
            if leftovers == 0:
                pass
            else:
                chain = chain[:-leftovers]
            quasisteps = len(chain) // self.nwalkers
            color = sp.arange(quasisteps)
            colors = sp.array(
                [color for i in range(self.nwalkers)]).reshape(-1)

            # Auxiliary variables to coordinate labels and filenames.
            tcount = 0
            pcount = 1
            acc = True
            ins = 0
            ins_count = 1

            for i in tqdm(range(self.ndim), desc='Brush type'):
                fig, ax = plt.subplots(figsize=self.chain_figsize)

                im = ax.scatter(
                    sp.arange(chain.shape[0]), chain[:, i],
                    c=colors, lw=0, cmap=self.chain_cmap, s=self.chain_size
                )

                ax.set_xlabel('N', fontsize=self.label_fontsize)
                ax.tick_params(
                    axis='both', which='major',
                    labelsize=self.tick_labelsize
                )

                cb = plt.colorbar(im, ax=ax)
                cb.set_label('Step Number', fontsize=self.label_fontsize,
                             rotation=270, labelpad=self.cbar_labelpad)
                cb.ax.tick_params(labelsize=self.tick_labelsize)

                # plot only accel and instrumental chains.
                if self.kplanets == 0:

                    if i == 0:
                        title = self.chain_titles[5]
                        ax.set_ylabel(
                            title + self.chain_units[-1],
                            fontsize=self.label_fontsize
                        )
                        counter = 0
                    else:
                        title = self.chain_titles[6 + counter % 2]
                        ax.set_ylabel(
                            title + self.chain_units[1],
                            fontsize=self.label_fontsize
                        )
                        counter += 1

                    if self.pdf:
                        plt.savefig(self.working_dir + 'chains/' + title
                                    + '_INS' + str(ins) + '_' + str(t)
                                    + 'T_K0.pdf')
                    if self.png:
                        plt.savefig(self.working_dir + 'chains/' + title
                                    + '_INS' + str(ins) + '_' + str(t)
                                    + 'T_K0.pdf')
                    ins_count += 1
                    ins += 1 if ins_count % 2 == 0 else 0
                else:

                    if pcount <= self.kplanets:
                        title = self.chain_titles[tcount % 5]
                        ax.set_ylabel(title + self.chain_units[tcount % 5],
                                      fontsize=self.label_fontsize)
                        tcount += 1
                    else:
                        if acc:
                            title = self.chain_titles[5]
                            ax.set_ylabel(
                                title + self.chain_units[-1],
                                fontsize=self.label_fontsize
                            )
                            acc = False
                            counter = 0
                        else:
                            title = self.chain_titles[6 + counter % 2]
                            ax.set_ylabel(
                                title + self.chain_units[1],
                                fontsize=self.label_fontsize
                            )
                            counter += 1

                if pcount <= self.kplanets:
                    if self.pdf:
                        plt.savefig(self.working_dir + 'chains/' + title +
                                    '_K' + str(pcount) + '_T' + str(t)
                                    + '.pdf')
                    if self.png:
                        plt.savefig(self.working_dir + 'chains/' + title +
                                    '_K' + str(pcount) + '_T' + str(t)
                                    + '.png')
                else:
                    if self.pdf:
                        plt.savefig(self.working_dir + 'chains/' + title
                                    + '_INS' + str(ins) + '_T' + str(t)
                                    + '.pdf')
                    if self.png:
                        plt.savefig(self.working_dir + 'chains/' + title
                                    + '_INS' + str(ins) + '_T' + str(t)
                                    + '.png')
                    ins_count += 1
                    ins += 1 if ins_count % 2 == 0 else 0
                pcount += 1 if tcount % 5 == 0 else 0
        pass

    def paint_posteriors(self):
        """Create posterior plots."""
        print('\nPAINTING POSTERIORS.')
        for t in tqdm(range(self.ntemps), desc='Brush temperature'):
            chain = self.chains[t]
            post = self.posteriors[t]

            leftovers = len(chain) % self.nwalkers
            if leftovers == 0:
                pass
            else:
                chain = chain[:-leftovers]
                post = post[:-(len(post) % self.nwalkers)]
            quasisteps = len(chain) // self.nwalkers
            color = sp.arange(quasisteps)
            colors = sp.array(
                [color for i in range(self.nwalkers)]).reshape(-1)

            # Auxiliary variables to coordinate labels and filenames.
            tcount = 0
            pcount = 1
            acc = True
            ins = 0
            ins_count = 1

            for i in tqdm(range(self.ndim), desc='Brush type'):
                fig, ax = plt.subplots(figsize=self.post_figsize)

                im = ax.scatter(
                    chain[:, i], post, s=self.post_size, c=colors, lw=0,
                    cmap=self.post_cmap, alpha=self.post_alpha
                )

                ax.axvline(
                    chain[sp.argmax(post), i], color=self.post_v_color,
                    linestyle=self.post_v_linestyle, alpha=self.post_v_alpha,
                    zorder=10
                )

                ax.tick_params(
                    axis='both', which='major',
                    labelsize=self.tick_labelsize
                )
                ax.tick_params(axis='x', rotation=45)
                ax.set_ylabel('Posterior', fontsize=self.label_fontsize)
                cb = plt.colorbar(im, ax=ax)
                cb.set_label('Step Number', fontsize=self.label_fontsize,
                             rotation=270, labelpad=self.cbar_labelpad)
                cb.ax.tick_params(labelsize=self.tick_labelsize)

                xaxis = ax.get_xaxis()
                xaxis.set_major_locator(
                    ticker.LinearLocator(numticks=self.post_ticknum)
                )
                yaxis = ax.get_yaxis()
                yaxis.set_major_locator(
                    ticker.LinearLocator(numticks=self.post_ticknum)
                )

                # plot only accel and instrumental chains.
                if self.kplanets == 0:

                    if i == 0:
                        title = self.chain_titles[5]
                        ax.set_xlabel(
                            title + self.chain_units[-1],
                            fontsize=self.label_fontsize
                        )
                        counter = 0
                    else:
                        title = self.chain_titles[6 + counter % 2]
                        ax.set_xlabel(
                            title + self.chain_units[1],
                            fontsize=self.label_fontsize
                        )
                        counter += 1

                    if self.pdf:
                        plt.savefig(self.working_dir + 'posteriors/' + title
                                    + '_INS' + str(ins) + '_' + str(t)
                                    + 'T_K0.pdf')
                    if self.png:
                        plt.savefig(self.working_dir + 'posteriors/' + title
                                    + '_INS' + str(ins) + '_' + str(t)
                                    + 'T_K0.pdf')
                    ins_count += 1
                    ins += 1 if ins_count % 2 == 0 else 0
                else:

                    if pcount <= self.kplanets:
                        title = self.chain_titles[tcount % 5]
                        ax.set_xlabel(title + self.chain_units[tcount % 5],
                                      fontsize=self.label_fontsize)
                        tcount += 1
                    else:
                        if acc:
                            title = self.chain_titles[5]
                            ax.set_xlabel(
                                title + self.chain_units[-1],
                                fontsize=self.label_fontsize
                            )
                            acc = False
                            counter = 0
                        else:
                            title = self.chain_titles[6 + counter % 2]
                            ax.set_xlabel(
                                title + self.chain_units[1],
                                fontsize=self.label_fontsize
                            )
                            counter += 1

                if pcount <= self.kplanets:
                    if self.pdf:
                        plt.savefig(self.working_dir + 'posteriors/' + title +
                                    '_K' + str(pcount) + '_T' + str(t)
                                    + '.pdf')
                    if self.png:
                        plt.savefig(self.working_dir + 'posteriors/' + title +
                                    '_K' + str(pcount) + '_T' + str(t)
                                    + '.png')
                else:
                    if self.pdf:
                        plt.savefig(self.working_dir + 'posteriors/' + title
                                    + '_INS' + str(ins) + '_T' + str(t)
                                    + '.pdf')
                    if self.png:
                        plt.savefig(self.working_dir + 'posteriors/' + title
                                    + '_INS' + str(ins) + '_T' + str(t)
                                    + '.png')
                    ins_count += 1
                    ins += 1 if ins_count % 2 == 0 else 0
                pcount += 1 if tcount % 5 == 0 else 0
        pass

    def paint_histograms(self):
        """Create histograms."""
        print('\nPAINTING HISTOGRAMS.')
        for t in tqdm(range(self.ntemps), desc='Brush temperature'):
            chain = self.chains[t]
            post = self.posteriors[t]

            # Auxiliary variables to coordinate labels and filenames.
            tcount = 0
            pcount = 1
            acc = True
            ins = 0
            ins_count = 1
            for i in tqdm(range(self.ndim), desc='Brush type'):
                fig, ax = plt.subplots(figsize=self.post_figsize)

                ax.set_ylabel('Frequency', fontsize=self.label_fontsize)

                dist = chain[:, i]

                peak = dist[sp.argmax(post)]
                n, bins = sp.histogram(dist, self.num_bins, density=1)
                dif = sp.fabs(peak - bins)
                his_peak = bins[sp.argmin(dif)]

                res = sp.where(n == 0)[0]

                if res.size:
                    if len(res) > 2:
                        for j in range(len(res)):
                            if res[j + 2] - res[j] == 2:
                                sub = j
                                break
                    else:
                        sub = res[0]

                    if bins[sub] > his_peak:
                        idx = sp.where(dist <= bins[sub])
                        post_sub = post[idx]
                        dist_sub = dist[idx]
                    else:
                        idx = sp.where(dist >= bins[sub])
                        post_sub = post[idx]
                        dist_sub = dist[idx]
                else:
                    dist_sub = dist
                    post_sub = post

                n, bins, patches = ax.hist(
                    dist_sub, self.num_bins, density=1,
                    facecolor=self.hist_facecolor, alpha=self.hist_alpha
                )

                mu, sigma = norm.fit(dist_sub)
                var = sigma ** 2

                # Statistics.
                skew = '{:.4e}'.format(Decimal(sp.stats.skew(dist_sub)))
                kurt = '{:.4e}'.format(Decimal(sp.stats.kurtosis(dist_sub)))
                gmod = '{:.4e}'.format(Decimal(bins[sp.argmax(n)]))
                med = '{:.4e}'.format(Decimal(sp.median(dist_sub)))

                span = bins[len(bins) - 1] - bins[0]
                bins_x = ((sp.arange(self.num_bins * 100) /
                           (self.num_bins * 100)) * span) + bins[0]

                # Make a renormalised gaussian plot.
                y = emplib.hist_gaussian(bins_x, mu, sigma) * n.max()

                ax.plot(bins_x, y, 'r-', linewidth=3)

                fig.subplots_adjust(left=.15)

                ax.set_ylim([0, n.max() * 1.7])

                ax.autoscale(enable=True, axis='x', tight=True)

                # Add stats to plot as text.

                ymin, ymax = ax.get_ylim()
                xmin, xmax = ax.get_xlim()

                mu_o = '{:.4e}'.format(Decimal(mu))
                sigma_o = '{:.4e}'.format(Decimal(sigma))
                var_o = '{:.4e}'.format(Decimal(var))

                ax.text(xmax - (xmax - xmin) * 0.65, ymax - (ymax - ymin)
                        * 0.1, r"$\mathcal{N}(\mu_1,\sigma^2,\mu_3,\mu_4)$",
                        size=25)
                ax.text(xmax - (xmax - xmin) * 0.8, ymax - (ymax - ymin)
                        * 0.180, r"$\mu_1 ={}$".format(mu_o), size=20)
                ax.text(xmax - (xmax - xmin) * 0.8, ymax - (ymax - ymin)
                        * 0.255, r"$\sigma^2 ={}$".format(var_o), size=20)
                ax.text(xmax - (xmax - xmin) * 0.8, ymax - (ymax - ymin)
                        * 0.330, r"$\mu_3 ={}$".format(skew), size=20)

                ax.text(xmax - (xmax - xmin) * 0.5, ymax - (ymax - ymin)
                        * 0.180, r"$\mu_4 ={}$".format(kurt), size=20)
                ax.text(xmax - (xmax - xmin) * 0.5, ymax - (ymax - ymin)
                        * 0.255, r"$Median ={}$".format(med), size=20)
                ax.text(xmax - (xmax - xmin) * 0.5, ymax - (ymax - ymin)
                        * 0.330, r"$Mode ={}$".format(gmod), size=20)

                if self.kplanets == 0:

                    if i == 0:
                        title = self.chain_titles[5]
                        ax.set_xlabel(
                            title + self.chain_units[-1],
                            fontsize=self.label_fontsize
                        )
                        counter = 0
                    else:
                        title = self.chain_titles[6 + counter % 2]
                        ax.set_xlabel(
                            title + self.chain_units[1],
                            fontsize=self.label_fontsize
                        )
                        counter += 1

                    if self.pdf:
                        plt.savefig(self.working_dir + 'histograms/' + title
                                    + '_INS' + str(ins) + '_' + str(t)
                                    + 'T_K0.pdf')
                    if self.png:
                        plt.savefig(self.working_dir + 'histograms/' + title
                                    + '_INS' + str(ins) + '_' + str(t)
                                    + 'T_K0.pdf')
                    ins_count += 1
                    ins += 1 if ins_count % 2 == 0 else 0

                else:

                    if pcount <= self.kplanets:
                        title = self.chain_titles[tcount % 5]
                        ax.set_xlabel(title + self.chain_units[tcount % 5],
                                      fontsize=self.label_fontsize)
                        tcount += 1
                    else:
                        if acc:
                            title = self.chain_titles[5]
                            ax.set_xlabel(
                                title + self.chain_units[-1],
                                fontsize=self.label_fontsize
                            )
                            acc = False
                            counter = 0
                        else:
                            title = self.chain_titles[6 + counter % 2]
                            ax.set_xlabel(
                                title + self.chain_units[1],
                                fontsize=self.label_fontsize
                            )
                            counter += 1

                if pcount <= self.kplanets:
                    if self.pdf:
                        plt.savefig(self.working_dir + 'histograms/' + title +
                                    '_K' + str(pcount) + '_T' + str(t)
                                    + '.pdf')
                    if self.png:
                        plt.savefig(self.working_dir + 'histograms/' + title +
                                    '_K' + str(pcount) + '_T' + str(t)
                                    + '.png')
                else:
                    if self.pdf:
                        plt.savefig(self.working_dir + 'histograms/' + title
                                    + '_INS' + str(ins) + '_T' + str(t)
                                    + '.pdf')
                    if self.png:
                        plt.savefig(self.working_dir + 'histograms/' + title
                                    + '_INS' + str(ins) + '_T' + str(t)
                                    + '.png')
                    ins_count += 1
                    ins += 1 if ins_count % 2 == 0 else 0
                pcount += 1 if tcount % 5 == 0 else 0
        pass

    def paint_corners(self):
        """Create corner plots. Cold chain only."""
        print('\nPAINTING CORNERS.')
        titles = ['P', 'K', r'$\phi$', 'e', r'$\omega$']
        if self.kplanets == 0:
            print('No cornerplots for K0.')
            return
        for k in tqdm(range(self.kplanets), desc='Brush number'):
            labels = [t + ' ' + str(k + 1) + '\n' + u
                      for t, u in zip(
                      self.chain_titles[:-1 - self.nins],
                      self.chain_units[:-1]
                      )]
            fig = corner.corner(
                self.cold[:, k * 5:(k + 1) * 5],
                plot_contours=True,
                fill_contours=False,
                plot_datapoints=True,
                no_fill_contours=True,
                max_n_ticks=3
            )
            params = self.__get_params(k)
            params_lo, params_up = self.__get_CI_params(k, .68)

            axes = sp.array(fig.axes).reshape((5, 5))

            for i in range(5):
                ax = axes[i, i]
                ax.axvline(params[i], color=self.corner_med_c,
                           linestyle=self.corner_med_style)
                ax.axvline(params_lo[i], color=self.corner_v_c,
                           linestyle=self.corner_v_style)
                ax.axvline(params_up[i], color=self.corner_v_c,
                           linestyle=self.corner_v_style)
                t = titles[i] + '={:.2f}'.format(params[i]) + \
                    r'$^{+' + '{:.2f}'.format(params_up[i] - params[i]) + \
                    r'}_{-' + '{:.2f}'.format(params[i] - params_lo[i]) + r'}$'

                ax.set_title(t, fontsize=self.corner_fontsize,
                             fontname=self.fontname)

            for yi in range(5):
                for xi in range(yi):
                    ax = axes[yi, xi]
                    if xi == 0:
                        for tick in ax.yaxis.get_major_ticks():
                            tick.label.set_fontsize(self.corner_tick_fontsize)
                            tick.label.set_fontname(self.fontname)
                            ax.set_ylabel(
                                labels[yi],
                                labelpad=self.corner_labelpad,
                                fontsize=self.corner_fontsize,
                                fontname=self.fontname
                            )
                    if yi == 4:
                        for tick in ax.xaxis.get_major_ticks():
                            tick.label.set_fontsize(self.corner_tick_fontsize)
                            tick.label.set_fontname(self.fontname)
                            ax.set_xlabel(
                                labels[xi],
                                labelpad=self.corner_labelpad,
                                fontsize=self.corner_fontsize,
                                fontname=self.fontname
                            )
                    ax.axvline(params[xi], color=self.corner_med_c,
                               linestyle=self.corner_med_style)
                    ax.axhline(params[yi], color=self.corner_med_c,
                               linestyle=self.corner_med_style)
                    ax.plot(params[xi], params[yi], self.corner_marker)
                axes[-1, -1].set_xlabel(
                    labels[-1],
                    labelpad=self.corner_labelpad,
                    fontsize=self.corner_fontsize,
                    fontname=self.fontname
                )
                for tick in axes[-1, -1].xaxis.get_major_ticks():
                    tick.label.set_fontsize(self.corner_tick_fontsize)
                    tick.label.set_fontname(self.fontname)
            fig.show()
        pass

    def __read_config(self):
        """Read configuration file for plotting."""
        # TODO: implement.
        self.phase_figsize = (20, 10)
        self.full_figsize = (20, 10)
        self.chain_figsize = (12, 7)
        self.post_figsize = (12, 7)
        self.hist_figsize = (12, 7)
        self.phase_cmap = 'cool_r'
        self.full_cmap = 'cool_r'
        self.phase_size = 100
        self.full_size = 100
        self.chain_size = 20
        self.chain_cmap = 'viridis'
        self.post_size = 20
        self.post_ticknum = 10
        self.post_alpha = .8
        self.post_cmap = 'viridis'
        self.post_v_color = 'red'
        self.post_v_linestyle = '--'
        self.post_v_linewidth = 2
        self.post_v_alpha = .7
        self.label_fontsize = 22
        self.num_bins = 12
        self.hist_facecolor = 'blue'
        self.hist_alpha = .5
        self.CI_color = 'mediumseagreen'
        self.error_color = 'k'
        self.fontname = 'serif'
        self.corner_med_c = 'firebrick'
        self.corner_v_c = 'lightcoral'
        self.corner_v_style = '-.'
        self.corner_med_style = '--'
        self.tick_labelsize = 20
        self.cbar_labelpad = 30
        self.corner_fontsize = 20
        self.corner_tick_fontsize = 15
        self.corner_labelpad = 15
        self.corner_marker = 'sr'
        pass
