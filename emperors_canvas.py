#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
import emperors_mirror as empmir
import matplotlib.pyplot as plt
from PyAstronomy.pyasl import foldAt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from tqdm import tqdm
from scipy.stats import norm
from decimal import Decimal  # histograms



def plot1(thetas, flattened, temp, kplanets, nins, totcornum, saveplace, setup,
          MOAV, PACC, HISTOGRAMS, CORNER, STARMASS, PNG, PDF, thin, draw_every_n,
          ticknum=10):
          #CORNER_MASK, CORNER_K, CORNER_I
    def gaussian(x, mu, sig):
        return sp.exp(-sp.power((x - mu)/sig, 2.)/2.)

    def plot(thetas, flattened, temp, kplanets, CORNER=False, ticknum=ticknum):
        ndim = 1 + 5 * kplanets + nins*2*(MOAV+1) + totcornum + PACC
        ntemps, nwalkers, nsteps = setup

        titles = sp.array(["Amplitude","Period","Longitude", "Phase","Eccentricity", 'Acceleration', 'Jitter', 'Offset', 'MACoefficient', 'MATimescale', 'Stellar Activity'])
        units = sp.array([" $[\\frac{m}{s}]$"," [Days]"," $[rad]$", " $[rads]$","", ' $[\\frac{m}{s^2}]$', ' $[\\frac{m}{s}]$', ' $[\\frac{m}{s}]$', ' $[\\frac{m}{s}]$', ' [Days]', ''])

        p_titles = sp.array(['p_Amplitude', 'p_phase', 'p_ecc'])

        leftovers = len(thetas)%nwalkers
        if leftovers == 0:
            pass
        else:
            thetas = thetas[:-leftovers]
            flattened = flattened[:-( len(flattened)%nwalkers )]
        quasisteps = len(thetas)//nwalkers

        color = sp.arange(quasisteps)
        colores = sp.array([color for i in range(nwalkers)]).reshape(-1)
        i = 0
        sorting = sp.arange(len(thetas))

        subtitles, namen = sp.array([]), sp.array([])

        for k in range(kplanets):
            subtitles = sp.append(subtitles, [titles[i] + ' '+str(k+1)+units[i] for i in range(5)])
            namen = sp.append(namen, [titles[i] + '_'+str(k) for i in range(5)])

        subtitles = sp.append(subtitles, titles[5]+units[5])  # for acc
        namen = sp.append(namen, titles[5])  # for acc
        if PACC:
            subtitles = sp.append(subtitles, 'Parab Accel $[\\frac{m}{s}]$')
            namen = sp.append(namen, 'Parab Accel')
        for i in range(nins):
            subtitles = sp.append(subtitles, [titles[ii] + ' '+str(i+1)+units[ii] for ii in sp.arange(2)+6])
            namen = sp.append(namen, [titles[ii] + '_'+str(i+1) for ii in sp.arange(2)+6])
            for j in range(MOAV):
                subtitles = sp.append(subtitles, [titles[ii] + ' '+str(i+1) + ' '+str(j+1)+units[ii] for ii in sp.arange(2)+8])
                namen = sp.append(namen, [titles[ii] + '_'+str(i+1) + '_'+str(j+1) for ii in sp.arange(2)+8])

        for h in range(totcornum):
            subtitles = sp.append(subtitles, titles[-1]+' '+str(h+1))
            namen = sp.append(namen, titles[-1]+'_'+str(h+1))

        print('\n PLOTTING CHAINS for temperature '+str(temp)+'\n')
        pbar_chain = tqdm(total=ndim)
        #############
        for i in range(ndim):  # chains
            fig, ax = plt.subplots(figsize=(12, 7))
            if subtitles[i][:3] == 'Per':
                pass

            ydif = (max(thetas[:,i]) - min(thetas[:,i])) / 10.
            ax.set(ylim=(min(thetas[:,i]) - ydif, max(thetas[:,i]) + ydif))

            im = ax.scatter(sorting, thetas[:,i], c=colores, lw=0., cmap='gist_rainbow', alpha=0.8)
            plt.xlabel("N", fontsize=24)
            plt.ylabel(subtitles[i], fontsize=24)

            cb = plt.colorbar(im, ax=ax)
            lab = 'Step Number'

            if thin * draw_every_n != 1:
                lab = 'Step Number * '+str(thin*draw_every_n)

            cb.set_label('Step Number')
            if PNG:
                fig.savefig(saveplace+"/chains"+temp+'_'+str(i+1)+'_'+namen[i]+".png")
            if PDF:
                fig.savefig(saveplace+"/chains"+temp+'_'+str(i+1)+'_'+namen[i]+".pdf")

            pbar_chain.update(1)
            plt.close('all')
        pbar_chain.close()

        print('\n PLOTTING POSTERIORS for temperature '+str(temp)+'\n')
        pbar_post = tqdm(total=ndim)
        for i in range(ndim):  # posteriors
            fig1, ax1 = plt.subplots(figsize=(12, 7))

            xdif1, ydif1 = (max(thetas[:,i]) - min(thetas[:,i])) / 10., (max(flattened) - min(flattened)) / 10.
            ax1.set(xlim=((min(thetas[:,i]) - xdif1), (max(thetas[:,i]) + xdif1)),
                    ylim=((min(flattened) - ydif1), (max(flattened) + ydif1)))

            im = ax1.scatter(thetas[:,i], flattened, s=10 , c=colores, lw=0., cmap='gist_rainbow', alpha=0.8)

            xaxis = ax1.get_xaxis()
            xaxis.set_major_locator(ticker.LinearLocator(numticks=ticknum))
            yaxis = ax1.get_yaxis()
            yaxis.set_major_locator(ticker.LinearLocator(numticks=ticknum))
            #yaxis.set_minor_locator(ticker.LinearLocator(numticks=5))
            '''
            if subtitles[i][:3] == 'Per':
                ax1.set_xscale('log')
                xaxis.set_major_locator(ticker.LogLocator(numticks=ticknum))
            '''
            ax1.axvline(thetas[sp.argmax(flattened), i], color='r', linestyle='--', linewidth=2, alpha=0.70)
            # ax1.invert_yaxis()

            plt.xlabel(subtitles[i], fontsize=24)
            plt.ylabel("Posterior", fontsize=24)

            cb = plt.colorbar(im, ax=ax1)
            lab = 'Step Number'
            if thin * draw_every_n != 1:
                lab = 'Step Number * '+str(thin*draw_every_n)
            cb.set_label(lab)

            if PNG:
                fig1.savefig(saveplace+"/posteriors"+temp+'_'+str(i+1)+'_'+namen[i]+".png")
            if PDF:
                fig1.savefig(saveplace+"/posteriors"+temp+'_'+str(i+1)+'_'+namen[i]+".pdf")
            plt.close('all')

            pbar_post.update(1)
        pbar_post.close()

        if HISTOGRAMS:
            if kplanets == 0:
                print 'Sorry! No histograms here yet! We are working on it ! '
                pass
            print('\n PLOTTING HISTOGRAMS for temperature '+str(temp)+'\n')
            lab=['Amplitude [m/s]','Period [d]',r'$\phi$ [rads]',r'$\omega$ [rads]','Eccentricity','a [AU]',r'Msin(i) [$M_{\oplus}$]']
            params=len(lab)
            pbar_hist = tqdm(total=params*kplanets)
            num_bins = 12
            for k in range(kplanets):
                per_s = thetas.T[5*k+1] * 24. * 3600.
                if STARMASS:
                    semi = ((per_s**2.0) / ( (4.0*sp.pi**2.0) / (6.67e-11 * STARMASS * 1.99e30) ))**(1./3) / 1.49598e11 #AU!!
                    Mass = thetas.T[5*k] / ( (28.4/sp.sqrt(1. - thetas.T[5*k+4]**2.)) * (STARMASS**(-0.5)) * (semi**(-0.5)) ) * 317.8 #Me!!
                else:
                    params = len(lab) - 2
                for ii in range(params):
                    if ii < 5:
                        Per = thetas.T[5*k + ii]
                    if ii == 5:
                        Per = semi
                    if ii == 6:
                        Per = Mass

                    mu,sigma = norm.fit(Per)  # Mean and sigma of distribution!!
                    # first histogram of the data
                    n, bins, patches = plt.hist(Per, num_bins, normed=1)
                    plt.close("all")  # We don't need the plot just data!!

                    #Get the maximum and the data around it!!
                    maxi = Per[sp.where(flattened == sp.amax(flattened))][0]
                    dif = sp.fabs(maxi - bins)
                    his_max = bins[sp.where(dif == sp.amin(dif))]

                    res=sp.where(n == 0)[0]  # Find the zeros!!
                    if res.size:
                        if len(res) > 2:
                            for j in range(len(res)):
                                if res[j+2] - res[j] == 2:
                                    sub=j
                                    break
                        else:
                            sub=res[0]

                        # Get the data subset!!
                        if bins[sub] > his_max:
                            post_sub=flattened[sp.where(Per <= bins[sub])]
                            Per_sub=Per[sp.where(Per <= bins[sub])]
                        else:
                            post_sub=flattened[sp.where(Per >= bins[sub])]
                            Per_sub=Per[sp.where(Per >= bins[sub])]

                    else:
                        Per_sub=Per
                        post_sub=flattened

                    plt.subplots(figsize=(12,7))  # Define the window size!!
                    # redo histogram of the subset of data
                    n, bins, patches = plt.hist(Per_sub, num_bins, normed=1, facecolor='blue', alpha=0.5)
                    mu, sigma = norm.fit(Per_sub)  # add a 'best fit' line
                    var = sigma**2.
                    #Some Stats!!
                    skew='%.4E' % Decimal(sp.stats.skew(Per_sub))
                    kurt='%.4E' % Decimal(sp.stats.kurtosis(Per_sub))
                    gmod='%.4E' % Decimal(bins[sp.where(n == sp.amax(n))][0])
                    med='%.4E' % Decimal(sp.median(Per_sub))
                    # print 'The skewness, kurtosis, mean, and median of the data are {} : {} : {} : {}'.format(skew,kurt,gmod,med)

                    #Make a model x-axis!!
                    span=bins[len(bins)-1] - bins[0]
                    bins_x=((sp.arange(num_bins*100.) / (num_bins*100.)) * span) + bins[0]

                    y = gaussian(bins_x, mu, sigma) * sp.amax(n) #Renormalised to the histogram maximum!!

                    axes = plt.gca()
                    #y = mlab.normpdf(bins, mu, sigma)
                    plt.plot(bins_x, y, 'r-',linewidth=3)

                    # Tweak spacing to prevent clipping of ylabel
                    plt.subplots_adjust(left=0.15)

                    #axes.set_xlim([])
                    axes.set_ylim([0.,sp.amax(n)+sp.amax(n)*0.7])

                    axes.set_xlabel(lab[ii],size=15)
                    axes.set_ylabel('Frequency',size=15)
                    axes.tick_params(labelsize=15)

                    plt.autoscale(enable=True, axis='x', tight=True)

                    #Get the axis positions!!
                    ymin, ymax = axes.get_ylim()
                    xmin, xmax = axes.get_xlim()

                    #Add a key!!
                    mu_o = '%.4E' % Decimal(mu)
                    sigma_o = '%.4E' % Decimal(sigma)
                    var_o = '%.4E' % Decimal(var)

                    axes.text(xmax - (xmax - xmin)*0.65, ymax - (ymax - ymin)*0.1, r"$\mathcal{N}(\mu_1,\sigma^2,\mu_3,\mu_4)$",size=25)
                    axes.text(xmax - (xmax - xmin)*0.8, ymax - (ymax - ymin)*0.180, r"$\mu_1 ={}$".format(mu_o),size=20)
                    axes.text(xmax - (xmax - xmin)*0.8, ymax - (ymax - ymin)*0.255, r"$\sigma^2 ={}$".format(var_o),size=20)
                    axes.text(xmax - (xmax - xmin)*0.8, ymax - (ymax - ymin)*0.330, r"$\mu_3 ={}$".format(skew),size=20)

                    axes.text(xmax - (xmax - xmin)*0.5, ymax - (ymax - ymin)*0.180, r"$\mu_4 ={}$".format(kurt),size=20)
                    axes.text(xmax - (xmax - xmin)*0.5, ymax - (ymax - ymin)*0.255, r"$Median ={}$".format(med),size=20)
                    axes.text(xmax - (xmax - xmin)*0.5, ymax - (ymax - ymin)*0.330, r"$Mode ={}$".format(gmod),size=20)

                    plt.savefig(saveplace+'/hist_test'+temp+'_'+str(k)+'_'+str(ii)+'.pdf') #,bbox_inches='tight')
                    plt.close('all')
                    pbar_hist.update(1)

            '''
                if i < 5*kplanets and i%7==5:
                    plt.savefig(saveplace+"/histogram"+temp+'_'+str(i+1)+'_'+'SMA'+".pdf")
                if i < 5*kplanets and i%7==6:
                    plt.savefig(saveplace+"/histogram"+temp+'_'+str(i+1)+'_'+'Mass'+".pdf")
                else:
                    plt.savefig(saveplace+"/histogram"+temp+'_'+str(i+1)+'_'+namen[i]+".pdf")
            '''

            pbar_hist.close()


        if CORNER:
            # ndim = 1 + 5 * kplanets + nins*2*(MOAV+1) + totcornum + PACC
            try:
                print('Plotting Corner Plot... May take a few seconds')
                fig = corner.corner(thetas, labels=subtitles)
                fig.savefig(saveplace+"/triangle.pdf")
            except:
                print('Corner Plot Failed!!')
                pass # corner
        try:
            plt.close('all')
        except:
            pass
        pass

    ntemps, nwalkers, nsteps = setup

    for i in range(ntemps):
        check_length = len(thetas[i])//nwalkers
        if check_length // draw_every_n < 100:
            draw_every_n = 1

        if i == 0:
            try:
                plot(thetas[0][::draw_every_n], flattened[0][::draw_every_n], '0', kplanets, CORNER=CORNER)
            except:
                print('Sample size insufficient to draw the posterior plots for the cold chain!')
                pass
        else:
            try:
                plot(thetas[i][::draw_every_n], flattened[i][::draw_every_n], str(i), kplanets)
            except:
                print('Sample size insufficient to draw the posterior plots for temp '+str(i)+' ! !')
                pass
    pass


def plot2(setup, all_data, fit, kplanets, nins, totcornum, starflag, staract,
          saveplace, ndat, MOAV, PACC, SHOW=False):

    def phasefold(TIME, RV, ERR, PER):
        phases = foldAt(TIME, PER, T0=0.0)
        sortIndi = sp.argsort(phases)  # sorts the points
        Phases = phases[sortIndi]  # gets the indices so we sort the RVs correspondingly(?)
        rv_phased = RV[sortIndi]
        time_phased = Phases * PER
        err_phased = ERR[sortIndi]
        return time_phased, rv_phased, err_phased

    def clear_noise(RV, theta_acc, theta_k, theta_i, theta_sa, staract, ndat):
        '''
        This should clean offset, add jitter to err
        clear acc, red noise and stellar activity
        '''
        time, rv, err, ins = all_data

        JITTER, OFFSET, MACOEF, MATS = sp.zeros(ndat), sp.zeros(ndat), sp.array([sp.zeros(ndat) for i in range(MOAV)]), sp.array([sp.zeros(ndat) for i in range(MOAV)])

        for i in range(ndat):
            jittpos = int(ins[i]*2*(MOAV+1))
            JITTER[i], OFFSET[i] = theta_i[jittpos], theta_i[jittpos + 1]
            for j in range(MOAV):
                MACOEF[j][i], MATS[j][i] = theta_i[jittpos + 2*(j+1)], theta_i[jittpos + 2*(j+1)+1]

        ERR = sp.sqrt(err ** 2 + JITTER ** 2)
        tmin, tmax = sp.amin(time), sp.amax(time)

        #RV0 = RV - OFFSET - ACC * (time - time[0])
        if totcornum:
            COR = sp.array([sp.array([sp.zeros(ndat) for k in range(len(starflag[i]))]) for i in range(len(starflag))])
            assert len(theta_sa) == totcornum, 'error in correlations'
            AR = 0.0  # just to remember to add this
            counter = -1

            for i in range(nins):
                for j in range(len(starflag[i])):
                    counter += 1
                    passer = -1
                    for k in range(ndat):
                        if starflag[i][j] == ins[k]:  #
                            passer += 1
                            COR[i][j][k] = theta_sa[counter] * staract[i][j][passer]

            FMC = 0
            for i in range(len(COR)):
                for j in range(len(COR[i])):
                    FMC += COR[i][j]
        else:
            FMC = 0


        if PACC:
            ACC = theta_acc[0] * (time - time[0]) + theta_acc[1] * (time - time[0]) ** 2
        else:
            ACC = theta_acc[0] * (time - time[0])

        MODEL = OFFSET + ACC + FMC

        for k in sp.arange(kplanets):
            MODEL += empmir.mini_RV_model(theta_k[5*k:5*(k+1)], time)
        residuals, MA = sp.zeros(ndat), sp.zeros(ndat)
        for i in range(ndat):
            residuals = RV - MODEL
            for c in range(MOAV):
                if i > c:
                    MA[i] = MACOEF[c][i] * sp.exp(-sp.fabs(time[i-1] - time[i]) / MATS[c][i]) * residuals[i-1]
                    MODEL[i] += MA[i]
                    residuals[i] -= MA[i]

        RV0 = RV - OFFSET - ACC - FMC - MA

        return RV0, ERR, residuals

    time, rv, err, ins = all_data
    ndim = 1 + 5 * kplanets + nins*2*(MOAV+1) + totcornum + PACC
    colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k', 'xkcd:indigo', 'xkcd:scarlet', 'xkcd:burnt orange', 'xkcd:apple green', 'xkcd:coral']
    #letter = ['a', 'b', 'c', 'd', 'e', 'f']  # 'a' is just a placeholder

    theta_k = fit[:kplanets * 5]
    theta_acc = fit[kplanets * 5:kplanets * 5 + PACC + 1]
    theta_i = fit[kplanets * 5 + PACC + 1:kplanets * 5 + nins*2*(MOAV+1) + PACC + 1]
    theta_sa = fit[kplanets * 5 + nins*2*(MOAV+1) + PACC + 1:]

    for k in range(kplanets):
        rv0, err0, residuals = clear_noise(rv, theta_acc, theta_k, theta_i, theta_sa, staract, ndat)
        rvk, errk = rv0, err0
        for kk in sp.arange(kplanets-1)+1:
            rvk -= empmir.mini_RV_model(theta_k[5*kk:5*(kk+1)], time)
        t_p, rv_p, err_p = phasefold(time, rvk, errk, theta_k[1])
        t_p, res_p, err_p = phasefold(time, residuals, errk, theta_k[1])

        time_m = sp.linspace(min(time), max(time), int(1e4))
        rv_m = empmir.mini_RV_model(theta_k[:5], time_m)

        for mode in range(2):
            fig = plt.figure(figsize=(20,10))
            gs = gridspec.GridSpec(3, 4)
            ax = fig.add_subplot(gs[:2, :])
            axr = fig.add_subplot(gs[-1, :])
            plt.subplots_adjust(hspace=0)

            for i in range(nins):  # printea datos separados por instrumento
                x, y, yerr = sp.array([]), sp.array([]), sp.array([])
                yr = sp.array([])
                for j in range(len(ins)):
                    if ins[j] == i:
                        x = sp.append(x, time[j])
                        y = sp.append(y, rvk[j])
                        yerr = sp.append(yerr, errk[j])

                        yr = sp.append(yr, residuals[j])
                if mode == 1:  # phasefolded
                    xp, yp, errp = phasefold(x, y, yerr, theta_k[1])  # phase fold
                    ax.errorbar(xp, yp, errp, color=colors[i], label='Data'+str(i), linestyle='', marker='o', alpha=0.75)  # phase fold
                    xpr, ypr, errpr = phasefold(x, yr, yerr, theta_k[1])
                    axr.errorbar(xpr, ypr, errpr, color=colors[i], fmt='o')
                    ax.set_xlim(min(xp), max(xp))
                    axr.set_xlim(min(xpr), max(xpr))

                else:  # full
                    ax.errorbar(x, y, yerr, color=colors[i], label='Data'+str(i), linestyle='', marker='o', alpha=0.75)
                    axr.errorbar(x, yr, yerr, color=colors[i], fmt='o')
                    ax.set_xlim(min(x), max(x))
                    axr.set_xlim(min(x), max(x))

            # best_fit de el modelo completo en linea
            if mode == 1:  # phasefolded
                time_m_p, rv_m_p, err_m_p = phasefold(time_m, rv_m, sp.zeros_like(time_m), theta_k[1])
                ax.plot(time_m_p, rv_m_p, 'k', label='model')
            else:  # full
                ax.plot(time_m, rv_m, '-k', label='model')
            # ax.minorticks_on()
            ax.set_ylabel('Radial Velocity (m/s)', fontsize=24)
            axr.axhline(0, color='k', linewidth=2)
            axr.get_yticklabels()[-1].set_visible(False)
            axr.minorticks_on()
            axr.set_ylabel('Residuals',fontsize=22)
            axr.set_xlabel('Time (Julian Days)',fontsize=22)
            if mode == 1:  # phasefolded
                fig.savefig(saveplace+'/phasefold'+str(k)+'.pdf')
            else:  # full
                fig.savefig(saveplace+'/fullmodel'+str(k)+'.pdf')
            if SHOW:
                plt.show()

        theta_k = sp.roll(theta_k, -5)  # 5 del principio al final
    pass


def plot1_PM(thetas, flattened, temp, kplanets, nins, totcornum, saveplace, setup,
             MOAV, PACC, HISTOGRAMS, CORNER, STARMASS, PNG, PDF, thin, draw_every_n,
             ticknum=10):
    def gaussian(x, mu, sig):
        return sp.exp(-sp.power((x - mu)/sig, 2.)/2.)

    def plot(thetas, flattened, temp, kplanets, CORNER=False, ticknum=ticknum):
        ndim = 1 + 4 * kplanets + nins*2*(MOAV+1) + totcornum + PACC
        ntemps, nwalkers, nsteps = setup

        titles = sp.array(["Period","Amplitude", "Phase","Eccentricity", 'Acceleration', 'Jitter', 'Offset', 'MACoefficient', 'MATimescale', 'Stellar Activity'])
        units = sp.array([" [Days]"," $[\\frac{m}{s}]$", " $[rads]$","", ' $[\\frac{m}{s^2}]$', ' $[\\frac{m}{s}]$', ' $[\\frac{m}{s}]$', ' $[\\frac{m}{s}]$', ' [Days]', ''])

        p_titles = sp.array(['p_Amplitude', 'p_phase', 'p_ecc'])

        thetas = thetas[:-( len(thetas)%nwalkers )]
        flattened = flattened[:-( len(flattened)%nwalkers )]
        quasisteps = len(thetas)//nwalkers

        color = sp.arange(quasisteps)
        colores = sp.array([color for i in range(nwalkers)]).reshape(-1)
        i = 0
        sorting = sp.arange(len(thetas))

        subtitles, namen = sp.array([]), sp.array([])

        for k in range(kplanets):
            subtitles = sp.append(subtitles, [titles[i] + ' '+str(k+1)+units[i] for i in range(4)])
            namen = sp.append(namen, [titles[i] + '_'+str(k) for i in range(4)])

        subtitles = sp.append(subtitles, titles[4]+units[4])  # for acc
        namen = sp.append(namen, titles[4])  # for acc
        if PACC:
            subtitles = sp.append(subtitles, 'Parab Accel $[\\frac{m}{s}]$')
            namen = sp.append(namen, 'Parab Accel')
        for i in range(nins):
            subtitles = sp.append(subtitles, [titles[ii] + ' '+str(i+1)+units[ii] for ii in sp.arange(2)+5])
            namen = sp.append(namen, [titles[ii] + '_'+str(i+1) for ii in sp.arange(2)+5])
            for j in range(MOAV):
                subtitles = sp.append(subtitles, [titles[ii] + ' '+str(i+1) + ' '+str(j+1)+units[ii] for ii in sp.arange(2)+7])
                namen = sp.append(namen, [titles[ii] + '_'+str(i+1) + '_'+str(j+1) for ii in sp.arange(2)+7])

        for h in range(totcornum):
            subtitles = sp.append(subtitles, titles[-1]+' '+str(h+1))
            namen = sp.append(namen, titles[-1]+'_'+str(h+1))

        print('\n PLOTTING CHAINS for temperature '+str(temp)+'\n')
        pbar_chain = tqdm(total=ndim)
        #############
        for i in range(ndim):  # chains
            fig, ax = plt.subplots(figsize=(12, 7))
            if subtitles[i][:3] == 'Per':
                pass

            ydif = (max(thetas[:,i]) - min(thetas[:,i])) / 10.
            ax.set(ylim=(min(thetas[:,i]) - ydif, max(thetas[:,i]) + ydif))

            im = ax.scatter(sorting, thetas[:,i], c=colores, lw=0., cmap='gist_rainbow', alpha=0.8)
            plt.xlabel("N", fontsize=24)
            plt.ylabel(subtitles[i], fontsize=24)

            cb = plt.colorbar(im, ax=ax)
            lab = 'Step Number'

            if thin * draw_every_n != 1:
                lab = 'Step Number * '+str(thin*draw_every_n)

            cb.set_label('Step Number')
            if PNG:
                fig.savefig(saveplace+"/chains"+temp+'_'+str(i+1)+'_'+namen[i]+".png")
            if PDF:
                fig.savefig(saveplace+"/chains"+temp+'_'+str(i+1)+'_'+namen[i]+".pdf")

            pbar_chain.update(1)
            plt.close('all')
        pbar_chain.close()

        print('\n PLOTTING POSTERIORS for temperature '+str(temp)+'\n')
        pbar_post = tqdm(total=ndim)
        for i in range(ndim):  # posteriors
            fig1, ax1 = plt.subplots(figsize=(12, 7))

            xdif1, ydif1 = (max(thetas[:,i]) - min(thetas[:,i])) / 10., (max(flattened) - min(flattened)) / 10.
            ax1.set(xlim=((min(thetas[:,i]) - xdif1), (max(thetas[:,i]) + xdif1)),
                    ylim=((min(flattened) - ydif1), (max(flattened) + ydif1)))

            im = ax1.scatter(thetas[:,i], flattened, s=10 , c=colores, lw=0., cmap='gist_rainbow', alpha=0.8)

            xaxis = ax1.get_xaxis()
            xaxis.set_major_locator(ticker.LinearLocator(numticks=ticknum))
            yaxis = ax1.get_yaxis()
            yaxis.set_major_locator(ticker.LinearLocator(numticks=ticknum))
            #yaxis.set_minor_locator(ticker.LinearLocator(numticks=5))
            '''
            if subtitles[i][:3] == 'Per':
                ax1.set_xscale('log')
                xaxis.set_major_locator(ticker.LogLocator(numticks=ticknum))
            '''
            ax1.axvline(thetas[sp.argmax(flattened), i], color='r', linestyle='--', linewidth=2, alpha=0.70)
            # ax1.invert_yaxis()

            plt.xlabel(subtitles[i], fontsize=24)
            plt.ylabel("Posterior", fontsize=24)

            cb = plt.colorbar(im, ax=ax1)
            lab = 'Step Number'
            if thin * draw_every_n != 1:
                lab = 'Step Number * '+str(thin*draw_every_n)
            cb.set_label(lab)

            if PNG:
                fig1.savefig(saveplace+"/posteriors"+temp+'_'+str(i+1)+'_'+namen[i]+".png")
            if PDF:
                fig1.savefig(saveplace+"/posteriors"+temp+'_'+str(i+1)+'_'+namen[i]+".pdf")
            plt.close('all')

            pbar_post.update(1)
        pbar_post.close()

        if HISTOGRAMS:
            if kplanets == 0:
                print 'Sorry! No histograms here yet! We are working on it ! '
                pass
            print('\n PLOTTING HISTOGRAMS for temperature '+str(temp)+'\n')
            lab=['Period [d]','Amplitude [m/s]',r'$\phi$ [rads]','Eccentricity','a [AU]',r'Msin(i) [$M_{\oplus}$]']
            params=len(lab)
            pbar_hist = tqdm(total=params*kplanets)
            num_bins = 12
            for k in range(kplanets):
                per_s = thetas.T[4*k] * 24. * 3600.
                if STARMASS:
                    semi = ((per_s**2.0) / ( (4.0*sp.pi**2.0) / (6.67e-11 * STARMASS * 1.99e30) ))**(1./3) / 1.49598e11 #AU!!
                    Mass = thetas.T[4*k] / ( (28.4/sp.sqrt(1. - thetas.T[5*k+3]**2.)) * (STARMASS**(-0.5)) * (semi**(-0.5)) ) * 317.8 #Me!!
                else:
                    params = len(lab) - 2
                for ii in range(params):
                    if ii < 4:
                        Per = thetas.T[4*k + ii]
                    if ii == 4:
                        Per = semi
                    if ii == 6:
                        Per = Mass

                    mu,sigma = norm.fit(Per)  # Mean and sigma of distribution!!
                    # first histogram of the data
                    n, bins, patches = plt.hist(Per, num_bins, normed=1)
                    plt.close("all")  # We don't need the plot just data!!

                    #Get the maximum and the data around it!!
                    maxi = Per[sp.where(flattened == sp.amax(flattened))][0]
                    dif = sp.fabs(maxi - bins)
                    his_max = bins[sp.where(dif == sp.amin(dif))]

                    res=sp.where(n == 0)[0]  # Find the zeros!!
                    if res.size:
                        if len(res) > 2:
                            for j in range(len(res)):
                                if res[j+2] - res[j] == 2:
                                    sub=j
                                    break
                        else:
                            sub=res[0]

                        # Get the data subset!!
                        if bins[sub] > his_max:
                            post_sub=flattened[sp.where(Per <= bins[sub])]
                            Per_sub=Per[sp.where(Per <= bins[sub])]
                        else:
                            post_sub=flattened[sp.where(Per >= bins[sub])]
                            Per_sub=Per[sp.where(Per >= bins[sub])]

                    else:
                        Per_sub=Per
                        post_sub=flattened

                    plt.subplots(figsize=(12,7))  # Define the window size!!
                    # redo histogram of the subset of data
                    n, bins, patches = plt.hist(Per_sub, num_bins, normed=1, facecolor='blue', alpha=0.5)
                    mu, sigma = norm.fit(Per_sub)  # add a 'best fit' line
                    var = sigma**2.
                    #Some Stats!!
                    skew='%.4E' % Decimal(sp.stats.skew(Per_sub))
                    kurt='%.4E' % Decimal(sp.stats.kurtosis(Per_sub))
                    gmod='%.4E' % Decimal(bins[sp.where(n == sp.amax(n))][0])
                    med='%.4E' % Decimal(sp.median(Per_sub))
                    # print 'The skewness, kurtosis, mean, and median of the data are {} : {} : {} : {}'.format(skew,kurt,gmod,med)

                    #Make a model x-axis!!
                    span=bins[len(bins)-1] - bins[0]
                    bins_x=((sp.arange(num_bins*100.) / (num_bins*100.)) * span) + bins[0]

                    y = gaussian(bins_x, mu, sigma) * sp.amax(n) #Renormalised to the histogram maximum!!

                    axes = plt.gca()
                    #y = mlab.normpdf(bins, mu, sigma)
                    plt.plot(bins_x, y, 'r-',linewidth=3)

                    # Tweak spacing to prevent clipping of ylabel
                    plt.subplots_adjust(left=0.15)

                    #axes.set_xlim([])
                    axes.set_ylim([0.,sp.amax(n)+sp.amax(n)*0.7])

                    axes.set_xlabel(lab[ii],size=15)
                    axes.set_ylabel('Frequency',size=15)
                    axes.tick_params(labelsize=15)

                    plt.autoscale(enable=True, axis='x', tight=True)

                    #Get the axis positions!!
                    ymin, ymax = axes.get_ylim()
                    xmin, xmax = axes.get_xlim()

                    #Add a key!!
                    mu_o = '%.4E' % Decimal(mu)
                    sigma_o = '%.4E' % Decimal(sigma)
                    var_o = '%.4E' % Decimal(var)

                    axes.text(xmax - (xmax - xmin)*0.65, ymax - (ymax - ymin)*0.1, r"$\mathcal{N}(\mu_1,\sigma^2,\mu_3,\mu_4)$",size=25)
                    axes.text(xmax - (xmax - xmin)*0.8, ymax - (ymax - ymin)*0.180, r"$\mu_1 ={}$".format(mu_o),size=20)
                    axes.text(xmax - (xmax - xmin)*0.8, ymax - (ymax - ymin)*0.255, r"$\sigma^2 ={}$".format(var_o),size=20)
                    axes.text(xmax - (xmax - xmin)*0.8, ymax - (ymax - ymin)*0.330, r"$\mu_3 ={}$".format(skew),size=20)

                    axes.text(xmax - (xmax - xmin)*0.5, ymax - (ymax - ymin)*0.180, r"$\mu_4 ={}$".format(kurt),size=20)
                    axes.text(xmax - (xmax - xmin)*0.5, ymax - (ymax - ymin)*0.255, r"$Median ={}$".format(med),size=20)
                    axes.text(xmax - (xmax - xmin)*0.5, ymax - (ymax - ymin)*0.330, r"$Mode ={}$".format(gmod),size=20)

                    plt.savefig(saveplace+'/hist_test'+temp+'_'+str(k)+'_'+str(ii)+'.pdf') #,bbox_inches='tight')
                    plt.close('all')
                    pbar_hist.update(1)

            '''
                if i < 5*kplanets and i%7==5:
                    plt.savefig(saveplace+"/histogram"+temp+'_'+str(i+1)+'_'+'SMA'+".pdf")
                if i < 5*kplanets and i%7==6:
                    plt.savefig(saveplace+"/histogram"+temp+'_'+str(i+1)+'_'+'Mass'+".pdf")
                else:
                    plt.savefig(saveplace+"/histogram"+temp+'_'+str(i+1)+'_'+namen[i]+".pdf")
            '''

            pbar_hist.close()


        if CORNER:
            try:
                print('Plotting Corner Plot... May take a few seconds')
                fig = corner.corner(thetas, labels=subtitles)
                fig.savefig(saveplace+"/triangle.pdf")
            except:
                print('Corner Plot Failed!!')
                pass # corner
        try:
            plt.close('all')
        except:
            pass
        pass

    ntemps, nwalkers, nsteps = setup

    for i in range(ntemps):
        check_length = len(thetas[i])//nwalkers
        if check_length // draw_every_n < 100:
            draw_every_n = 1

        if i == 0:
            try:
                plot(thetas[0][::draw_every_n], flattened[0][::draw_every_n], '0', kplanets, CORNER=CORNER)
            except:
                print('Sample size insufficient to draw the posterior plots for the cold chain!')
                pass
        else:
            try:
                plot(thetas[i][::draw_every_n], flattened[i][::draw_every_n], str(i), kplanets)
            except:
                print('Sample size insufficient to draw the posterior plots for temp '+str(i)+' ! !')
                pass
    pass


def plot2_PM(setup, all_data, fit, kplanets, nins, totcornum, starflag, staract,
             saveplace, ndat, MOAV, PACC, SHOW=False):

    def phasefold(TIME, RV, ERR, PER):
        phases = foldAt(TIME, PER, T0=0.0)
        sortIndi = sp.argsort(phases)  # sorts the points
        Phases = phases[sortIndi]  # gets the indices so we sort the RVs correspondingly(?)
        rv_phased = RV[sortIndi]
        time_phased = Phases * PER
        err_phased = ERR[sortIndi]
        return time_phased, rv_phased, err_phased

    def clear_noise(RV, theta_acc, theta_k, theta_i, theta_sa, staract):
        '''
        This should clean offset, add jitter to err
        clear acc, red noise and stellar activity
        '''
        time, rv, err, ins = all_data

        JITTER, OFFSET, MACOEF, MATS = sp.zeros(ndat), sp.zeros(ndat), sp.array([sp.zeros(ndat) for i in range(MOAV)]), sp.array([sp.zeros(ndat) for i in range(MOAV)])

        for i in range(ndat):
            jittpos = int(ins[i]*2*(MOAV+1))
            JITTER[i], OFFSET[i] = theta_i[jittpos], theta_i[jittpos + 1]
            for j in range(MOAV):
                MACOEF[j][i], MATS[j][i] = theta_i[jittpos + 2*(j+1)], theta_i[jittpos + 2*(j+1)+1]

        ERR = sp.sqrt(err ** 2 + JITTER ** 2)
        tmin, tmax = sp.amin(time), sp.amax(time)

        #RV0 = RV - OFFSET - ACC * (time - time[0])
        if totcornum:
            COR = sp.array([sp.array([sp.zeros(ndat) for k in range(len(starflag[i]))]) for i in range(len(starflag))])
            assert len(theta_sa) == totcornum, 'error in correlations'
            AR = 0.0  # just to remember to add this
            counter = -1

            for i in range(nins):
                for j in range(len(starflag[i])):
                    counter += 1
                    passer = -1
                    for k in range(ndat):
                        if starflag[i][j] == ins[k]:  #
                            passer += 1
                            COR[i][j][k] = theta_sa[counter] * staract[i][j][passer]

            FMC = 0
            for i in range(len(COR)):
                for j in range(len(COR[i])):
                    FMC += COR[i][j]
        else:
            FMC = 0


        if PACC:
            ACC = theta_acc[0] * (time - time[0]) + theta_acc[1] * (time - time[0]) ** 2
        else:
            ACC = theta_acc[0] * (time - time[0])

        MODEL = OFFSET + ACC + FMC

        for k in sp.arange(kplanets):
            MODEL += empmir.mini_PM_model(theta_k[4*k:4*(k+1)], time)
        residuals, MA = sp.zeros(ndat), sp.zeros(ndat)
        for i in range(ndat):
            residuals = RV - MODEL
            for c in range(MOAV):
                if i > c:
                    MA[i] = MACOEF[c][i] * sp.exp(-sp.fabs(time[i-1] - time[i]) / MATS[c][i]) * residuals[i-1]
                    MODEL[i] += MA[i]
                    residuals[i] -= MA[i]

        RV0 = RV - OFFSET - ACC - FMC - MA

        return RV0, ERR, residuals

    time, rv, err, ins = all_data
    ndim = 1 + 4 * kplanets + nins*2*(MOAV+1) + totcornum + PACC
    colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k', 'xkcd:indigo', 'xkcd:scarlet', 'xkcd:burnt orange', 'xkcd:apple green', 'xkcd:coral']
    #letter = ['a', 'b', 'c', 'd', 'e', 'f']  # 'a' is just a placeholder

    theta_k = fit[:kplanets * 4]
    theta_acc = fit[kplanets * 4:kplanets * 4 + PACC + 1]
    theta_i = fit[kplanets * 4 + PACC + 1:kplanets * 4 + nins*2*(MOAV+1) + PACC + 1]
    theta_sa = fit[kplanets * 4 + nins*2*(MOAV+1) + PACC + 1:]

    for k in range(kplanets):
        rv0, err0, residuals = clear_noise(rv, theta_acc, theta_k, theta_i, theta_sa, staract)
        rvk, errk = rv0, err0
        for kk in sp.arange(kplanets-1)+1:
            rvk -= empmir.mini_PM_model(theta_k[4*kk:4*(kk+1)], time)
        t_p, rv_p, err_p = phasefold(time, rvk, errk, theta_k[1])
        t_p, res_p, err_p = phasefold(time, residuals, errk, theta_k[1])

        time_m = sp.linspace(min(time), max(time), int(1e4))
        rv_m = empmir.mini_PM_model(theta_k[:4], time_m)

        for mode in range(2):
            fig = plt.figure(figsize=(20,10))
            gs = gridspec.GridSpec(3, 4)
            ax = fig.add_subplot(gs[:2, :])
            axr = fig.add_subplot(gs[-1, :])
            plt.subplots_adjust(hspace=0)

            for i in range(nins):  # printea datos separados por instrumento
                x, y, yerr = sp.array([]), sp.array([]), sp.array([])
                yr = sp.array([])
                for j in range(len(ins)):
                    if ins[j] == i:
                        x = sp.append(x, time[j])
                        y = sp.append(y, rvk[j])
                        yerr = sp.append(yerr, errk[j])

                        yr = sp.append(yr, residuals[j])
                if mode == 1:  # phasefolded
                    xp, yp, errp = phasefold(x, y, yerr, theta_k[1])  # phase fold
                    ax.errorbar(xp, yp, errp, color=colors[i], label='Data'+str(i), linestyle='', marker='o', alpha=0.75)  # phase fold
                    xpr, ypr, errpr = phasefold(x, yr, yerr, theta_k[1])
                    axr.errorbar(xpr, ypr, errpr, color=colors[i], fmt='o')
                    ax.set_xlim(min(xp), max(xp))
                    axr.set_xlim(min(xpr), max(xpr))

                else:  # full
                    ax.errorbar(x, y, yerr, color=colors[i], label='Data'+str(i), linestyle='', marker='o', alpha=0.75)
                    axr.errorbar(x, yr, yerr, color=colors[i], fmt='o')
                    ax.set_xlim(min(x), max(x))
                    axr.set_xlim(min(x), max(x))

            # best_fit de el modelo completo en linea
            if mode == 1:  # phasefolded
                time_m_p, rv_m_p, err_m_p = phasefold(time_m, rv_m, sp.zeros_like(time_m), theta_k[1])
                ax.plot(time_m_p, rv_m_p, 'k', label='model')
            else:  # full
                ax.plot(time_m, rv_m, '-k', label='model')
            # ax.minorticks_on()
            ax.set_ylabel('Radial Velocity (m/s)', fontsize=24)
            axr.axhline(0, color='k', linewidth=2)
            axr.get_yticklabels()[-1].set_visible(False)
            axr.minorticks_on()
            axr.set_ylabel('Residuals',fontsize=22)
            axr.set_xlabel('Time (Julian Days)',fontsize=22)
            if mode == 1:  # phasefolded
                fig.savefig(saveplace+'/phasefold'+str(k)+'.pdf')
            else:  # full
                fig.savefig(saveplace+'/fullmodel'+str(k)+'.pdf')
            if SHOW:
                plt.show()

        theta_k = sp.roll(theta_k, -4)  # 5 del principio al final
    pass
