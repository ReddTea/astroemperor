#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
from PyAstronomy.pyasl import MarkleyKESolver
from emperors_library import normal_pdf
import batman

def RV_model(THETA, time, kplanets):
    modelo = 0.0
    if kplanets == 0:
        return 0.0
    for i in range(kplanets):
        P, As, Ac, S, C = THETA[5*i:5*(i+1)]
        A = As ** 2 + Ac ** 2
        ecc = S ** 2 + C ** 2
        w = sp.arccos(C / (ecc ** 0.5))  # longitude of periastron
        phase = sp.arccos(Ac / (A ** 0.5))
        ### test
        if S < 0:
            w = 2 * sp.pi - sp.arccos(C / (ecc ** 0.5))
        if As < 0:
            phase = 2 * sp.pi - sp.arccos(Ac / (A ** 0.5))

        ###
        per = sp.exp(P)
        freq = 2. * sp.pi / per
        M = freq * time + phase  # mean anomaly
        E = sp.array([MarkleyKESolver().getE(m, ecc) for m in M])  # eccentric anomaly
        f = (sp.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * sp.tan(E / 2.)) * 2.)  # true anomaly
        modelo += A * (sp.cos(f + w) + ecc * sp.cos(w))
    return  modelo


def mini_RV_model(params, time):
    P, A, phase, w, ecc = params
    freq = 2. * sp.pi / P
    M = freq * time + phase
    E = sp.array([MarkleyKESolver().getE(m, ecc) for m in M])
    f = (sp.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * sp.tan(E / 2.)) * 2.)
    modelo = A * (sp.cos(f + w) + ecc * sp.cos(w))
    return modelo


def RV4_model(theta, time, kplanets):
    modelo = 0.0
    if kplanets == 0:
        return 0.0
    for i in range(kplanets):
        P, As, Ac, ecc = theta[4*i:4*(i+1)]
        A = As ** 2 + Ac ** 2
        phase = sp.arccos(Ac / (A ** 0.5))
        if As < 0:
            phase = 2 * sp.pi - sp.arccos(Ac / (A ** 0.5))
        per = sp.exp(P)
        freq = 2. * sp.pi / per
        M = freq * time + phase
        E = sp.array([MarkleyKESolver().getE(m, ecc) for m in M])
        f = (sp.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * sp.tan(E / 2.)) * 2.)
        #print '\r '+str(i)+str(per)  # testing purposes
        modelo += A * (sp.cos(f) + ecc)
    return modelo


def mini_RV4_model(params, time):
    P, A, phase, ecc = params
    freq = 2. * sp.pi / P
    M = freq * time + phase
    E = sp.array([MarkleyKESolver().getE(m, ecc) for m in M])
    f = (sp.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * sp.tan(E / 2.)) * 2.)
    modelo = A * (sp.cos(f) + ecc)
    return modelo


def transit_lightCurve(lc_params, time, fsigns, free_params, ld, *theta_j):
    # (t0 radius dist inc), P, time, fsigns
    #print 'flux1\n\n'  # PMPMPM
    flux = 0.0
    for i in range(fsigns):
        params = batman.TransitParams()
        nfp = len(free_params)
        f_params = (nfp + ld) * fsig
        if theta_j:
            params.per = theta_j[3*i]
            params.w = theta_j[3*i + 1]
            params.ecc = theta_j[3*i + 2]
        else:
            params.per = lc_params[(nfp+ld)*(i+1) - 2]
            params.w = lc_params[(nfp+ld)*(i+1) - 1]
            params.ecc = lc_params[(nfp+ld)*(i+1)]
        #params.per = P                       #orbital period in days
        params.t0 = lc_params[(nfp+ld)*i]                       #time of inferior conjunction
        params.rp = lc_params[(nfp+ld*i) + 1]                   #planet radius (in units of stellar radii)
        params.a = lc_params[(nfp+ld*i) + 2]                      #semi-major axis (in units of stellar radii)
        params.inc = lc_params[(nfp+ld*i) + 3]                     #orbital inclination (in degrees)
        #params.ecc = 0.                      #eccentricity

        #params.w = 0.                        #longitude of periastron (in degrees)
        u = []
        for j in range(ld):
            u.append(lc_params[nfp+j + (nfp+ld)*i])
        params.u = u                #limb darkening coefficients [u1, u2]
        params.limb_dark = "quadratic"       #limb darkening model

        m = batman.TransitModel(params, time)    #initializes model
        flux += m.light_curve(params)          #calculates light curve
    #print flux, 'flux2\n\n'  # PMPMPM
    return (flux)

#lc_params, time, fsigns, free_params, ld, *theta_j
def mini_transit(lc_params, time, fsigns, free_params, ld, *theta_j):
    #P, t0, radius, dist, inc = theta
    nfp = len(free_params)
    params = batman.TransitParams()
    if theta_j:
        params.per = theta_j[0]                       #orbital period in days
        params.w = theta_j[1]                        #longitude of periastron (in degrees)
        params.ecc = theta_j[2]                      #eccentricity
    else:
        params.per = nfplc_params[nfp+ld]                       #orbital period in days
        params.w = nfplc_params[nfp+ld+1]                        #longitude of periastron (in degrees)
        params.ecc = nfplc_params[nfp+ld+2]                      #eccentricity
    params.t0 = nfplc_params[0]                       #time of inferior conjunction
    params.rp = nfplc_params[1]                   #planet radius (in units of stellar radii)
    params.a = nfplc_params[2]                      #semi-major axis (in units of stellar radii)
    params.inc = nfplc_params[3]                     #orbital inclination (in degrees)
    u = []
    for i in range(ld):
        u.append(nfplc_params[nfp+i])
    params.u = u                #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"       #limb darkening model

    m = batman.TransitModel(params, time)    #initializes model
    flux = m.light_curve(params)          #calculates light curve

    return (flux)


def pm_model(subtheta, P, time):
     #T0, r, a, b, c = param
    print('transit1')  # PMPMPM
    T0, r = subtheta
    # (t0 radius dist inc), P, time, fsigns
    lc_params = T0, r, 101.1576001138329, 89.912
    transit = transit_lightCurve(lc_params, P, time, fsigns)  # que es cada uno
    #t = time - time[0]
    #print transit, 'transit2\n\n'  # PMPMPM
    return transit #+ a + b*t + c*t*t


def henshin(thetas, kplanets):
    for i in range(kplanets):
        Pk = thetas[:, i*5]
        Ask = thetas[:, i*5 + 1]
        Ack = thetas[:, i*5 + 2]
        Sk = thetas[:, i*5 + 3]
        Ck = thetas[:, i*5 + 4]

        ecck  = Sk ** 2 + Ck ** 2
        Ak = Ask ** 2 + Ack ** 2
        wk = sp.arccos(Ck / (ecck ** 0.5))
        Phasek = sp.arccos(Ack / (Ak ** 0.5))
        for j in range(len(Sk)):
            if Sk[j] < 0:
                wk[j] = 2 * sp.pi - sp.arccos(Ck[j] / (ecck[j] ** 0.5))
            if Ask[j] < 0:
                Phasek[j] = 2 * sp.pi - sp.arccos(Ack[j] / (Ak[j] ** 0.5))

        thetas[:, i*5] = sp.exp(Pk)
        thetas[:, i*5 + 1] = Ak
        thetas[:, i*5 + 2] = Phasek
        thetas[:, i*5 + 3] = wk
        thetas[:, i*5 + 4] = ecck
    return thetas


def henshin4(thetas, kplanets):
    for i in range(kplanets):
        Pk = thetas[:, i*4]
        Ask = thetas[:, i*4 + 1]
        Ack = thetas[:, i*4 + 2]
        ecck = thetas[:, i*4 + 3]

        Ak = Ask ** 2 + Ack ** 2
        Phasek = sp.arccos(Ack / (Ak ** 0.5))
        for j in range(len(Ask)):
            if Ask[j] < 0:
                Phasek[j] = 2 * sp.pi - sp.arccos(Ack[j] / (Ak[j] ** 0.5))

        thetas[:, i*4] = sp.exp(Pk)
        thetas[:, i*4 + 1] = Ak
        thetas[:, i*4 + 2] = Phasek
        thetas[:, i*4 + 3] = ecck
    return thetas


def RV_residuals(theta, rv, time, ins, staract, starflag, kplanets, nins, MOAV, totcornum, PACC):
    ndat = len(time)
    model_params = kplanets * 5
    acc_params = 1 + PACC
    ins_params = nins * 2 * (MOAV + 1)
    jitter, offset, macoef, timescale = sp.zeros(ndat), sp.zeros(ndat), sp.array([sp.zeros(ndat) for i in range(MOAV)]), sp.array([sp.zeros(ndat) for i in range(MOAV)])
    if PACC:
        ACC = theta[model_params] * (time - time[0]) + theta[model_params + 1] * (time - time[0]) ** 2
    else:
        ACC = theta[model_params] * (time - time[0])

    residuals = sp.zeros(ndat)
    for i in range(ndat):
        jitpos = int(model_params + acc_params + ins[i] * 2 * (MOAV+1))
        jitter[i], offset[i] = theta[jitpos], theta[jitpos + 1]  # jitt
        for j in range(MOAV):
            macoef[j][i], timescale[j][i] = theta[jitpos + 2*(j+1)], theta[jitpos + 2*(j+1) + 1]
    a1 = (theta[:model_params])

    if totcornum:
        COR = sp.array([sp.array([sp.zeros(ndat) for k in range(len(starflag[i]))]) for i in range(len(starflag))])
        SA = theta[model_params+acc_params+ins_params:]

        assert len(SA) == totcornum, 'error in correlations'
        AR = 0.0  # just to remember to add this
        counter = -1

        for i in range(nins):
            for j in range(len(starflag[i])):
                counter += 1
                passer = -1
                for k in range(ndat):
                    if starflag[i][j] == ins[k]:  #
                        passer += 1
                        COR[i][j][k] = SA[counter] * staract[i][j][passer]

        FMC = 0
        for i in range(len(COR)):
            for j in range(len(COR[i])):
                FMC += COR[i][j]
    else:
        FMC = 0

    MODEL = RV_model(a1, time, kplanets) + offset + ACC + FMC
    #print MODEL

    for i in range(ndat):
        residuals[i] = rv[i] - MODEL[i]
        for c in range(MOAV):
            if i > c:
                MA = macoef[c][i] * sp.exp(-sp.fabs(time[i-1-c] - time[i]) / timescale[c][i]) * residuals[i-1-c]
                residuals[i] -= MA
    return residuals


### P R I O R S  ###

def logp_rv(theta, params):
    time, kplanets, nins = params[0], params[1], params[2]
    MOAV, totcornum, boundaries = params[3], params[4], params[5]
    inslims, acc_lims, sigmas = params[6], params[7], params[8]
    eccprior, jittprior, jittmean = params[9], params[10], params[11]
    STARMASS, HILL, PACC = params[12], params[13], params[14]
    CHECK = params[15]
    #CHECK = True

    G = 39.5  ##6.67408e-11 * 1.9891e30 * (1.15740741e-5) ** 2  # in Solar Mass-1 s-2 m3
    lp_flat_fix, lp_flat_ins, lp_ecc, lp_jitt = 0., 0., 0., 0.
    lp_correl = 0.
    lp_jeffreys = 0.

    k_params = kplanets * 5
    acc_params = 1 + PACC
    ins_params = nins * 2 * (MOAV + 1)
    MP = sp.zeros(kplanets)
    SMA, GAMMA = sp.zeros(kplanets), sp.zeros(kplanets)
    for k in range(kplanets):
        Pk, Ask, Ack, Sk, Ck = theta[k*5:(k+1)*5]

        if (boundaries[k*10+2] <= Ask <= boundaries[k*10+3] and
            boundaries[k*10+4] <= Ack <= boundaries[k*10+5] and
            boundaries[k*10+6] <= Sk <= boundaries[k*10+7] and
            boundaries[k*10 + 8] <= Ck <= boundaries[k*10 + 9]):
            lp_flat_fix += 0.0
            #print('GOOD_ONE_1')
        else:
            if CHECK:
                print('mark 1', boundaries[k*10+2],Ask, boundaries[k*10+3],
                      boundaries[k*10+4], Ack, boundaries[k*10+5],
                      boundaries[k*10+6], Sk, boundaries[k*10+7],
                      boundaries[k*10 + 8], Ck, boundaries[k*10 + 9])
            return -sp.inf

        Pmin, Pmax = boundaries[k*10], boundaries[k*10+1]
        if Pmin <= Pk <= Pmax:
            #lp_jeffreys += sp.log(Pk**-1 / (sp.log(Pmax/Pmin)))
            lp_jeffreys += 0.  # Because it is logp ??
            #print('GOOD_ONE_2', Pmin, Pk, Pmax)
        else:
            if CHECK:
                print('mark 2', Pmin, Pk, Pmax)
            return -sp.inf

        Ak = Ask ** 2 + Ack ** 2
        ecck = Sk ** 2 + Ck ** 2

        if 0.0 <= ecck <= 1.0:
            lp_ecc += normal_pdf(ecck, 0, eccprior**2)
        else:
            if CHECK:
                print('ecc')
            return -sp.inf



        if kplanets > 1:
            if HILL:
                per_s = Pk * 24. * 3600.
                SMA[k] = ((per_s**2.0) / ( (4.0*sp.pi**2.0) / (6.67e-11 * STARMASS * 1.99e30) ))**(1./3) / 1.49598e11
                MP[k] = Ak / ( (28.4/sp.sqrt(1. - ecck**2.)) * (STARMASS**(-0.5)) * (SMA[k]**(-0.5)) ) * 317.8
                #SMA[k] = (((sp.exp(Pk)/365.242199) ** 2) * G * STARMASS / (4 * sp.pi ** 2)) ** (1/3.)
                #MP[k] = sp.sqrt(1 - ecck ** 2) * Ak * sp.exp(Pk) ** (1/3.) * STARMASS ** (2/3.) / 203.
                GAMMA[k] = sp.sqrt(1 - ecck)
    if kplanets > 1:
        if HILL:
            orden = sp.argsort(SMA)
            SMA = SMA[orden]  # in AU
            MP = MP[orden]  # in Earth Masses
            #M = STARMASS * 1047.56 + sp.sum(MP)  # to Jupyter masses
            M = STARMASS * 332946 + sp.sum(MP)  # to Earth Masses

            mu = MP / M
            for kk in range(kplanets-1):
                alpha = mu[kk] + mu[kk+1]
                delta = sp.sqrt(SMA[kk+1] / SMA[kk])

                if alpha ** -3 * (mu[kk] + (mu[kk+1] / (delta ** 2))) * (mu[kk] * GAMMA[kk] + mu[kk+1] * GAMMA[kk+1] * delta)**2 < 1 + (3./alpha)**(4./3) * (mu[kk] * mu[kk+1]):
                    if CHECK:
                        print('HILL UNSTABLE')
                    return -sp.inf
                else:
                    pass

    acc_k = theta[k_params]
    if acc_lims[0] <= acc_k <= acc_lims[1]:
        lp_flat_fix += 0
    else:
        if CHECK:
            print('ACCEL')
        return -sp.inf

    if PACC:
        if acc_lims[0] <= theta[k_params + PACC] <= acc_lims[1]:
            lp_flat_fix += 0
        else:
            if CHECK:
                print('PACCEL ')
            return -sp.inf

    j = 0
    lp_flat_ins = 0.0
    lp_jitt = 0.0
    for j in range(nins):
        for c in range(MOAV):
            macoef_j = theta[k_params + acc_params + j*2*(MOAV+1) + 2*(c+1)]
            timescale_j = theta[k_params + acc_params + j*2*(MOAV+1) + 2*(c+1) + 1]
            bookmark = 4 * (j*MOAV + j + c + 1)
            if (inslims[bookmark] <= macoef_j <= inslims[bookmark+1] and
                inslims[bookmark+2] <= timescale_j <= inslims[bookmark+3]):
                lp_flat_ins += 0.0
            else:
                if CHECK:
                    print('MOVING AVERAGE')
                return -sp.inf
        jitt_j = theta[k_params + acc_params + j*2*(MOAV+1)]
        offset_j = theta[k_params + acc_params + j*2*(MOAV+1) + 1]
        jittmin, jittmax = inslims[4*(j*MOAV+j)], inslims[4*(j*MOAV+j)+1]
        if jittmin <= jitt_j <= jittmax:
            lp_jitt += normal_pdf(jitt_j, jittmean, jittprior**2)
        else:
            if CHECK:
                print('SI JITT')
            return -sp.inf
        if inslims[4*(j*MOAV+j)+2] <= offset_j <= inslims[4*(j*MOAV+j)+3]:
            lp_flat_ins += 0.0
        else:
            if CHECK:
                print('NO JITT')
            return -sp.inf
    for h in range(totcornum):
        cork = theta[k_params + acc_params + ins_params + h]
        if 0. <= cork <= 1.:  # hmmm
            lp_correl += 0
        else:
            return -sp.inf

    return lp_ecc + lp_flat_fix + lp_flat_ins + lp_jitt + lp_correl


def logp_pm(theta, params):
    '''
    FLAT POR AHORA
    sp.array([self.time_pm, fsig, self.nins_pm, self.MOAV_pm,
    self.totcornum_pm, boundaries, sigmas_raw,
    self.PACC_pm])]
    '''
    '''#
    print 'prior1\n\n'  # PMPMPM
    time, fsig, lenppm  = params[0], params[1], params[2]
    nins, MOAV, totcornum = params[3], params[4], params[5]
    boundaries, sigmas, PACC = params[6], params[7], params[8]

    T0, r, k_a, k_r = theta
    '''#
    '''
    if not (2456915.67 < T0  < 2456915.73 and 0.05 < r  < 0.09 and
        0 < k_a < 5.0e-2 and 0 < k_r < 6.):
        return -np.inf
    '''
    '''#
    prior = 0.
    k = -2
    for j in fsig:
        for i in range(lenppm):
            k += 2
            if boundaries[k] <= theta[i] <= boundaries[k+1]:  # keep an eye on this as different fsigns WILL fail
                prior += 0
            else:
                return -sp.inf

            ###
            #
            mu = 0.07 # mean of the Normal prior
            sigma = 0.004 # standard deviation of the Normal prior
            prob_r = normal_pdf(r - mu, sigma)

            mu = 2456915.70
            sigma =  0.005
            prob_t0 = normal_pdf(T0 - mu, sigma)
            prob = prob_t0
            return prob
            #
            ###
    print prior, 'prior2\n\n'  # PMPMPM
    '''#
    prior = 0.0
    return prior
    pass


def logp_rvpm(theta, params):
    params_rv, params_pm = params
    time, kplanets, nins = params_rv[0], params_rv[1], params_rv[2]  # checked
    MOAV, totcornum, boundaries = params_rv[3], params_rv[4], params_rv[5]  # checked
    PACC = params_rv[14]
    ndim_rv = 5*kplanets + 2*nins*(MOAV+1) + (1 + PACC) + totcornum

    theta_rv = theta[:ndim_rv]
    theta_pm = theta[ndim_rv:]



    LOGP_RV = logp_rv(theta_rv, params_rv)
    if kplanets > 0:
        LOGP_PM = logp_pm(theta_pm, params_pm)
        #print LOGP_RV, LOGP_PM, 'LOGP_RV, LOGP_PM'  # PMPMPM
    else:
        LOGP_PM = 0.0
    return LOGP_RV + LOGP_PM
    pass


def logp_4mod(theta):
    time, kplanets, nins = params[0], params[1], params[2]
    MOAV, totcornum, boundaries = params[3], params[4], params[5]
    inslims, acc_lims, sigmas = params[6], params[7], params[8]
    eccprior, jittprior, jittmean = params[9], params[10], params[11]
    STARMASS, HILL, PACC = params[12], params[13], params[14]
    CHECK = params[15]
    fsig, f2k = 1, 1

    pm_dim = 3


    G = 39.5 ##6.67408e-11 * 1.9891e30 * (1.15740741e-5) ** 2  # in Solar Mass-1 s-2 m3
    lp_flat_fix, lp_flat_ins, lp_ecc, lp_jitt = 0., 0., 0., 0.
    lp_correl = 0.
    lp_jeffreys = 0.

    model_params = kplanets * 5
    acc_params = 1 + PACC
    ins_params = nins * 2 * (MOAV + 1)
    MP = sp.zeros(kplanets)
    SMA, GAMMA = sp.zeros(kplanets), sp.zeros(kplanets)
    for k in range(kplanets):
        Pk, Ask, Ack, Sk, Ck = theta[k*5:(k+1)*5]

        if (boundaries[k*10+2] <= Ask <= boundaries[k*10+3] and
            boundaries[k*10+4] <= Ack <= boundaries[k*10+5] and
            boundaries[k*10+6] <= Sk <= boundaries[k*10+7] and
            boundaries[k*10 + 8] <= Ck <= boundaries[k*10 + 9]):
            lp_flat_fix += 0.0
            #print('GOOD_ONE_1')
        else:
            if CHECK:
                print('mark 1', boundaries[k*10+2],Ask, boundaries[k*10+3],
                      boundaries[k*10+4], Ack, boundaries[k*10+5],
                      boundaries[k*10+6], Sk, boundaries[k*10+7],
                      boundaries[k*10 + 8], Ck, boundaries[k*10 + 9])
            return -sp.inf

        Pmin, Pmax = boundaries[k*10], boundaries[k*10+1]
        if Pmin <= Pk <= Pmax:
            #lp_jeffreys += sp.log(Pk**-1 / (sp.log(Pmax/Pmin)))
            lp_jeffreys += 0.  # Because it is logp ??
            #print('GOOD_ONE_2', Pmin, Pk, Pmax)
        else:
            if CHECK:
                print('mark 2', Pmin, Pk, Pmax)
            return -sp.inf

        Ak = Ask ** 2 + Ack ** 2
        ecck = Sk ** 2 + Ck ** 2

        if 0.0 <= ecck <= 1.0:
            lp_ecc += normal_pdf(ecck, 0, eccprior**2)
        else:
            if CHECK:
                print('ecc')
            return -sp.inf



        if kplanets > 1:
            if HILL:
                per_s = Pk * 24. * 3600.
                SMA[k] = ((per_s**2.0) / ( (4.0*sp.pi**2.0) / (6.67e-11 * STARMASS * 1.99e30) ))**(1./3) / 1.49598e11
                MP[k] = Ak / ( (28.4/sp.sqrt(1. - ecck**2.)) * (STARMASS**(-0.5)) * (SMA[k]**(-0.5)) ) * 317.8
                #SMA[k] = (((sp.exp(Pk)/365.242199) ** 2) * G * STARMASS / (4 * sp.pi ** 2)) ** (1/3.)
                #MP[k] = sp.sqrt(1 - ecck ** 2) * Ak * sp.exp(Pk) ** (1/3.) * STARMASS ** (2/3.) / 203.
                GAMMA[k] = sp.sqrt(1 - ecck)
    if kplanets > 1:
        if HILL:
            orden = sp.argsort(SMA)
            SMA = SMA[orden]  # in AU
            MP = MP[orden]  # in Earth Masses
            #M = STARMASS * 1047.56 + sp.sum(MP)  # to Jupyter masses
            M = STARMASS * 332946 + sp.sum(MP)  # to Earth Masses

            mu = MP / M
            for kk in range(kplanets-1):
                alpha = mu[kk] + mu[kk+1]
                delta = sp.sqrt(SMA[kk+1] / SMA[kk])

                if alpha ** -3 * (mu[kk] + (mu[kk+1] / (delta ** 2))) * (mu[kk] * GAMMA[kk] + mu[kk+1] * GAMMA[kk+1] * delta)**2 < 1 + (3./alpha)**(4./3) * (mu[kk] * mu[kk+1]):
                    if CHECK:
                        print('HILL UNSTABLE')
                    return -sp.inf
                else:
                    pass

    acc_k = theta[kplanets * 5]
    if acc_lims[0] <= acc_k <= acc_lims[1]:
        lp_flat_fix += 0
    else:
        if CHECK:
            print('ACCEL')
        return -sp.inf

    if PACC:
        if acc_lims[0] <= theta[kplanets * 5 + 1] <= acc_lims[1]:
            lp_flat_fix += 0
        else:
            if CHECK:
                print('PACCEL ')
            return -sp.inf

    j = 0
    lp_flat_ins = 0.0
    lp_jitt = 0.0
    for j in range(nins):
        for c in range(MOAV):
            macoef_j = theta[model_params + acc_params + j*2*(MOAV+1) + 2*(c+1)]
            timescale_j = theta[model_params + acc_params + j*2*(MOAV+1) + 2*(c+1) + 1]
            bookmark = 4 * (j*MOAV + j + c + 1)
            if (inslims[bookmark] <= macoef_j <= inslims[bookmark+1] and
                inslims[bookmark+2] <= timescale_j <= inslims[bookmark+3]):
                lp_flat_ins += 0.0
            else:
                if CHECK:
                    print('MOVING AVERAGE')
                return -sp.inf
        jitt_j = theta[model_params + acc_params + j*2*(MOAV+1)]
        offset_j = theta[model_params + acc_params + j*2*(MOAV+1) + 1]
        jittmin, jittmax = inslims[4*(j*MOAV+j)], inslims[4*(j*MOAV+j)+1]
        if jittmin <= jitt_j <= jittmax:
            lp_jitt += normal_pdf(jitt_j, jittmean, jittprior**2)
        else:
            if CHECK:
                print('SI JITT')
            return -sp.inf
        if inslims[4*(j*MOAV+j)+2] <= offset_j <= inslims[4*(j*MOAV+j)+3]:
            lp_flat_ins += 0.0
        else:
            if CHECK:
                print('NO JITT')
            return -sp.inf
    for h in range(totcornum):
        cork = theta[model_params + acc_params + ins_params + h]
        if 0. <= cork <= 1.:  # hmmm
            lp_correl += 0
        else:
            return -sp.inf

    return lp_ecc + lp_flat_fix + lp_flat_ins + lp_jitt + lp_correl



### L I K E L I H O O D S ###



def logl_rv(theta, params):
    time, rv, err = params[0], params[1], params[2]
    ins, staract, starflag = params[3], params[4], params[5]
    kplanets, nins, MOAV = params[6], params[7], params[8]
    totcornum, PACC = params[9], params[10]
    i, lnl = 0, 0
    ndat = len(time)
    model_params = kplanets * 5
    acc_params = 1 + PACC
    ins_params = nins * 2 * (MOAV + 1)
    jitter, offset, macoef, timescale = sp.zeros(ndat), sp.zeros(ndat), sp.array([sp.zeros(ndat) for i in range(MOAV)]), sp.array([sp.zeros(ndat) for i in range(MOAV)])
    if PACC:
        ACC = theta[model_params] * (time - time[0]) + theta[model_params + 1] * (time - time[0]) ** 2
    else:
        ACC = theta[model_params] * (time - time[0])

    residuals = sp.zeros(ndat)
    for i in range(ndat):
        jitpos = int(model_params + acc_params + ins[i] * 2 * (MOAV+1))
        jitter[i], offset[i] = theta[jitpos], theta[jitpos + 1]  # jitt
        for j in range(MOAV):
            macoef[j][i], timescale[j][i] = theta[jitpos + 2*(j+1)], theta[jitpos + 2*(j+1) + 1]
    a1 = (theta[:model_params])

    if totcornum:
        #print 'SE ACTIBOY'
        COR = sp.array([sp.array([sp.zeros(ndat) for k in range(len(starflag[i]))]) for i in range(len(starflag))])
        SA = theta[model_params+acc_params+ins_params:]

        assert len(SA) == totcornum, 'error in correlations'
        AR = 0.0  # just to remember to add this
        counter = -1

        for i in range(nins):
            for j in range(len(starflag[i])):
                counter += 1
                passer = -1
                for k in range(ndat):
                    if starflag[i][j] == ins[k]:  #
                        passer += 1
                        COR[i][j][k] = SA[counter] * staract[i][j][passer]

        FMC = 0
        for i in range(len(COR)):
            for j in range(len(COR[i])):
                FMC += COR[i][j]
    else:
        #print 'NO SE AKTIBOY'
        FMC = 0

    MODEL = RV_model(a1, time, kplanets) + offset + ACC + FMC


    for i in range(ndat):
        residuals[i] = rv[i] - MODEL[i]
        for c in range(MOAV):
            if i > c:
                MA = macoef[c][i] * sp.exp(-sp.fabs(time[i-1-c] - time[i]) / timescale[c][i]) * residuals[i-1-c]
                residuals[i] -= MA

    inv_sigma2 = 1.0 / (err**2 + jitter**2)
    lnl = sp.sum(residuals ** 2 * inv_sigma2 - sp.log(inv_sigma2)) + sp.log(2*sp.pi) * ndat
    return -0.5 * lnl


import george
from george import kernels

def logl_pm(theta, params, P):
    #'''#
    #print 'lnl1\n'  # PMPMPM
    time, flux, err = params[0], params[1], params[2]
    ins, staract, starflag = params[3], params[4], params[5]
    fsig, f2k, nins = params[6], params[7], params[8]
    MOAV, totcornum, PACC = params[9], params[10], params[11]
    kplanets = params[12]

    i, lnl = 0, 0
    ndat = len(time)
    model_params = kplanets * 5
    acc_params = 1 + PACC
    ins_params = nins * 2 * (MOAV + 1)
    #totcornum
    pm_params = 4 * fsig  # PMPMPM
    #def lnlike_gp(param, x, y, yerr):
    radius = 10.**theta[-1]
    gp = george.GP(theta[-2] * kernels.Matern32Kernel(radius))
    #raise ImportError
    '''
    gp.compute(time, err)
    '''
    lnl = gp.lnlikelihood(flux - pm_model(theta[:-4][:2], P, time))
    # pm_model(subtheta, P, time, params)
    print(lnl, 'lnl2\n')  # PMPMPM

    #pm_model(theta, P, time, params)
    #'''#
    lnl = 0.0
    return lnl


def logl_rvpm(theta, params):
    params_rv, params_pm = params
    #time, rv, err = params_rv[0], params_rv[1], params_rv[2]
    #ins, staract, starflag = params_rv[3], params_rv[4], params_rv[5]
    kplanets, nins, MOAV = params_rv[6], params_rv[7], params_rv[8]
    totcornum, PACC = params_rv[9], params_rv[10]

    ndim_rv = 5*kplanets + 2*nins*(MOAV+1) + (1 + PACC) + totcornum
    theta_rv = theta[:ndim_rv]
    P = sp.array([theta[5*k] for k in range(kplanets)])
    #print P, 'periods\n\n'
    LOGL_RV = logl_rv(theta_rv, params_rv)
    #if kplanets == 1:
    #    print LOGL_RV, 'LOGL_RV'  # LOGL_PM, 'LOGL_RV, LOGL_PM\n\n'  # PMPMPM
    #'''
    theta_pm = theta[ndim_rv:]
    if kplanets > 0:
        LOGL_PM = logl_pm(theta_pm, params_pm, P)
    else:
        LOGL_PM = 0.0
    #'''
    return LOGL_RV + LOGL_PM



def uniform(x, lims, args):
    if lims[0] <= x <= lims[1]:
        return 0.
    else:
        return -sp.inf

def uniform_spe(x, lims, args):
    if lims[0] <= x <= lims[1]:
        return 0.
    else:
        return -sp.inf

def flat(x, lims, args):
    if lims[0] <= x <= lims[1]:
        return 0.0
    else:
        return -sp.inf

def jeffreys(x, lims, args):
    if lims[0] <= x <= lims[1]:
        return sp.log(x**-1 / (sp.log(lims[1]/lims[0])))
    else:
        return -sp.inf

def normal(x, lims, args):
    if lims[0] <= x <= lims[1]:
        mean, var = args[0], 2*args[1]
        return ( - (x - mean) ** 2 / var)
    else:
        return -sp.inf

def fixed(x, lims, args):
    if lims[0] <= x <= lims[1]:
        return 0.0
    else:
        return -sp.inf

def joined(x, lims, args):
    if lims[0] <= x <= lims[1]:
        return 0.0
    else:
        return -sp.inf

D = {'uniform':uniform,
     'flat':flat,
     'jeffreys':jeffreys,
     'normal':normal,
     'uniform_spe':uniform_spe,
     'fixed':fixed,
     'joined':joined}

def neo_logp_rv(theta, params):
    _theta, ndim, C = params
    c, lp = 0, 0.

    for j in range(ndim):
        #print(_theta[j+c].name)
        #print(D[_theta[C[j]].prior](theta[j], _theta[C[j]].lims, _theta[C[j]].args))
        lp += D[_theta[C[j]].prior](theta[j], _theta[C[j]].lims, _theta[C[j]].args)

    # add HILL criteria!!
    #G = 39.5  ##6.67408e-11 * 1.9891e30 * (1.15740741e-5) ** 2  # in Solar Mass-1 s-2 m3
    #MP = sp.zeros(kplanets)  # this goes in hill
    #SMA, GAMMA = sp.zeros(kplanets), sp.zeros(kplanets)
    return lp

def neo_logl_rv(theta, paramis):
    # PARAMS DEFINITIONS

    _t, AC, params = paramis

    time, rv, err = params[0], params[1], params[2]
    ins, staract, starflag = params[3], params[4], params[5]
    kplanets, nins, MOAV = params[6], params[7], params[8]
    totcornum, ACC = params[9], params[10]
    i, lnl = 0, 0
    ndat = len(time)
    jitter, offset = sp.zeros(ndat), sp.zeros(ndat)

    macoef, timescale = sp.array([sp.zeros(ndat) for i in range(sp.amax(MOAV))]), sp.array([sp.zeros(ndat) for i in range(sp.amax(MOAV))])

    model_params = kplanets * 5
    ins_params = (nins + sp.sum(MOAV)) * 2
    acc_params = ACC

    # THETA CORRECTION FOR FIXED THETAS
    for a in AC:
        theta = sp.insert(theta, a, _t[a].val)

    if ACC > 0:
        ACC = sp.polyval(sp.r_[0, theta[model_params:model_params+ACC]], (time-time[0]))


    # SETUP

    residuals = sp.zeros(ndat)
    for i in range(ndat):
        jitpos = int(model_params + acc_params + (ins[i] + sp.sum(MOAV[:int(ins[i])])) * 2)
        jitter[i], offset[i] = theta[jitpos], theta[jitpos + 1]  #
        for jj in range(MOAV[int(ins[i])]):
            macoef[jj][i] = theta[jitpos + 2*(jj+1)]
            timescale[jj][i] = theta[jitpos + 2*(jj+1) + 1]
    a1 = (theta[:model_params])

    #raise Exception('dele al debug mijo')
    # CHECK THIS CHECK THIS
    if totcornum:
        #print 'SE ACTIBOY'
        COR = sp.array([sp.array([sp.zeros(ndat) for k in range(len(starflag[i]))]) for i in range(len(starflag))])
        SA = theta[model_params+acc_params+ins_params:]

        assert len(SA) == totcornum, 'error in correlations'
        AR = 0.0  # just to remember to add this
        counter = -1

        for i in range(nins):
            for j in range(len(starflag[i])):
                counter += 1
                passer = -1
                for k in range(ndat):
                    if starflag[i][j] == ins[k]:  #
                        passer += 1
                        COR[i][j][k] = SA[counter] * staract[i][j][passer]

        FMC = 0
        for i in range(len(COR)):
            for j in range(len(COR[i])):
                FMC += COR[i][j]
    else:
        #print 'NO SE AKTIBOY'
        FMC = 0

    MODEL = RV_model(a1, time, kplanets) + offset + ACC + FMC


    for i in range(ndat):
        residuals[i] = rv[i] - MODEL[i]
        for c in range(sp.amax(MOAV)):
            if i > c:
                MA = macoef[c][i] * sp.exp(-sp.fabs(time[i-1-c] - time[i]) / timescale[c][i]) * residuals[i-1-c]
                residuals[i] -= MA

    inv_sigma2 = 1.0 / (err**2 + jitter**2)
    lnl = sp.sum(residuals ** 2 * inv_sigma2 - sp.log(inv_sigma2)) + sp.log(2*sp.pi) * ndat
    return -0.5 * lnl




#
