# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
# -*- coding: utf-8 -*-
import pickle

import scipy as sp

a = sp.array(['RV_dataset1.vels', 'RV_dataset14.vels'])
aa = sp.array(['RV_dataset14.vels'])


def read_data(instruments):
    """Data pre-processing."""
    nins = len(instruments)
    instruments = sp.array([sp.loadtxt('datafiles/' + x) for x in instruments])

    def data(data, ins_no):
        Time, Radial_Velocity, Err = data.T[:3]  # el error de la rv
        # Radial_Velocity -= sp.mean(Radial_Velocity)  # DEL for pm testing
        # marca el instrumento al q pertenece
        Flag = sp.ones(len(Time)) * ins_no
        Staract = data.T[3:]
        return sp.array([Time, Radial_Velocity, Err, Flag, Staract])

    def sortstuff(tryin):
        t, rv, er, flag = tryin
        order = sp.argsort(t)
        return sp.array([x[order] for x in [t, rv, er, flag]])

    fd = sp.array([]), sp.array([]), sp.array([]), sp.array([])

    for k in range(len(instruments)):  # appends all the data in megarg
        t, rv, er, flag, star = data(instruments[k], k)
        fd = sp.hstack((fd, [t, rv, er, flag]))  # ojo this, list not array

    # fd[0] = fd[0] - min(fd[0])  # min t
    alldat = sp.array([])
    staract = []
    for i in range(nins):
        hold = data(instruments[i], i)[4]
        staract.append(hold)
    # staract = sp.array([data(instruments[i], i)[4] for i in range(nins)])
    # print 'alr'
    #    except:
    #        staract = sp.array([sp.array([]) for i in range(nins)])
    #        print 'nr'
    starflag = sp.array([sp.array([i for k in range(len(staract[i]))])
                         for i in range(len(staract))])
    tryin = sortstuff(fd)
    for i in range(len(starflag)):
        for j in range(len(starflag[i])):
            staract[i][j] -= sp.mean(staract[i][j])
    totcornum = 0
    for correlations in starflag:
        if len(correlations) > 0:
            totcornum += len(correlations)
    # print fd[0]  # THISLINE
    # print sp.argsort(fd[0])  # THISLINE
    return tryin, staract, starflag, totcornum


class DATA:
    def __init__(self, instruments):
        # =sp.array(['RV_dataset1.vels', 'RV_dataset14.vels'
        self.nins = len(instruments)
        # all_data[x] pickea dataset
        self.all_data = sp.array(
            [sp.loadtxt('datafiles/' + x) for x in instruments])
        self.rv = sp.array([])
        self.activity = sp.array([])
        self.cornum = sp.array([])

        for i in range(self.nins):  # stack all data
            dat0, dat1, cornum = self.insert_labels(
                self.all_data[i], i)  # rvs, activities
            print(dat1, cornum)
            if i == 0:
                self.rv = dat0
            else:
                self.rv = sp.r_[self.rv, dat0]
            self.cornum = sp.append(self.cornum, cornum)

        self.rv_sorted = self.sortstuff(self.rv)
        self.activity = sp.array(
            [self.insert_labels(self.all_data[i], i)[1] for i in range(self.nins)])
        # print self.activity
        #self.act_sorted = self.sortstuff(self.activity)

    def insert_labels(self, data, ins_no):
        flag = sp.ones_like(data.T[0]) * ins_no  # flags
        holder = sp.c_[data[:, :3], flag]  # rvs
        if data[:, 3:].size > 0:
            holder1 = sp.c_[data[:, 0], data[:, 3:], flag]  # activity
            # print holder1.shape
            cornum = len(holder1.T) - 2
        else:
            # print data[:, 3:]
            holder1 = sp.array([])
            cornum = 0
        return holder, holder1, cornum

    def sortstuff(self, data_all):
        order = sp.argsort(data_all.T[0])  # by time
        return sp.array([x[order] for x in data_all.T])

        '''
        # this goes in __init__ in astroemperor.py
                rvdat = emplib.DATA(stardat[0])
                self.time, self.rv, self.err, self.ins = rvdat.rv_sorted  # time, radial velocities, error and instrument flag
                self.all_data = rvdat.all_data
                self.staract = rvdat.activity  # time, star activity index and flag
                self.totcornum = sp.sum(rvdat.cornum)  # quantity if star activity indices
                self.starflag = sp.array([sp.array([i for _ in range(len(rvdat.cornum[i]))]) for _ in range(len(rvdat.cornum))])  # backwards compatibility
                self.rvfiles = stardat[0]
                self.nins = len(self.rvfiles)  # number of instruments autodefined
                self.ndat = len(self.time)  # number of datapoints
                # PM
                pmdat = emplib.DATA(stardat[1])
                self.time_pm, self.rv_pm, self.err_pm, self.ins_pm = pmdat.rv_sorted  # just the fname its pm rly
                self.all_data_pm = pmdat.all_data
                self.staract_pm = pmdat.activity  # time, star activity index and flag
                self.totcornum_pm = pmdat.cornum  # ?
                self.pmfiles = stardat[1]
                self.nins_pm = len(self.pmfiles)
                self.ndat_pm = len(self.time_pm)
                self.fsig = 1
            else:
                # RV
                rvdat = emplib.DATA(stardat)
                self.time, self.rv, self.err, self.ins = rvdat.rv_sorted  # time, radial velocities, error and instrument flag
                self.all_data = rvdat.all_data
                self.staract = rvdat.activity  # time, star activity index and flag
                self.totcornum = rvdat.cornum  # quantity if star activity indices
                self.rvfiles = stardat
                self.nins = len(self.rvfiles)  # number of instruments autodefined
                self.ndat = len(self.time)  # number of datapoints
                # PM
                self.time_pm, self.rv_pm, self.err_pm, self.ins_pm = 0., 0., 0., 0.
                self.totcornum_pm = 0.
        '''


def read_data_f(instruments):
    """Pre-process data."""
    fnins = len(instruments)
    instruments = sp.array([sp.loadtxt('datafiles/' + x) for x in instruments])

    def data(data, ins_no):
        Time, Radial_Velocity = data.T[:2]  # el error de la rv
        Radial_Velocity -= sp.mean(Radial_Velocity)
        # marca el instrumento al q pertenece
        Flag = sp.ones(len(Time)) * ins_no
        return sp.array([Time, Radial_Velocity, Flag])

    def sortstuff(tryin):
        t, rv, flag = tryin
        order = sp.argsort(t)
        return sp.array([x[order] for x in [t, rv, flag]])

    fd = sp.array([]), sp.array([]), sp.array([])

    for k in range(len(instruments)):  # appends all the data in megarg
        t, rv, flag = data(instruments[k], k)
        fd = sp.hstack((fd, [t, rv, flag]))  # ojo this, list not array

    fd[0] = fd[0] - min(fd[0])
    alldat = sp.array([])

    tryin = sortstuff(fd)

    return fd, staract, starflag, totcornum


def normal_pdf(x, mean, variance):
    var = 2 * variance
    return (- (x - mean) ** 2 / var)


def gaussian(x, sigma):
    coef = -(x * x) / (2 * sigma * sigma)
    return 1 / np.sqrt(2 * np.pi * sigma * sigma) * np.exp(coef)


def hist_gaussian(x, mu, sig):
    return sp.exp(-sp.power((x - mu) / sig, 2.) / 2.)


def pt_pos(setup, *args):

    if args:
        kplanets, nins, boundaries = args[0], args[1], args[2]
        inslims, acc_lims, MOAV = args[3], args[4], args[5]
        totcornum, PACC = args[6], args[7]

    ntemps, nwalkers, nsteps = setup
    k_params = 5 * kplanets
    i_params = nins * 2 * (MOAV + 1)

    ndim = 1 + k_params + i_params + totcornum + PACC
    pos = sp.zeros((nwalkers, ndim))
    k = -2
    l = -2
    ll = -2
    for j in range(ndim):
        if j < k_params:
            k += 2
            if j % 5 == 0:
                fact = sp.absolute(
                    boundaries[k] - boundaries[k + 1]) / nwalkers
            else:
                #fact = sp.absolute(boundaries[k]) / (self.nwalkers)
                fact = (sp.absolute(
                    boundaries[k] - boundaries[k + 1]) * 2) / (5 * nwalkers)
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
            for i in range(nwalkers):
                if j % 5 == 0:
                    pos[i][j] = boundaries[k] + (dif[i] + fact / 2.0)
                else:
                    #pos[i][j] = boundaries[k] * 0.5 + (dif[i] + fact/2.0)
                    pos[i][j] = (boundaries[k + 1] + 3 *
                                 boundaries[k]) / 4 + (dif[i] + fact / 2.0)
        if j == 5 * kplanets:  # acc
            fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
            for i in range(nwalkers):
                pos[i][j] = acc_lims[0] + (dif[i] + fact / 2.0)
        if PACC:
            if j == 5 * kplanets + PACC:  # parabolic accel
                fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers
                dif = sp.arange(nwalkers) * fact * \
                    sp.random.uniform(0.9, 0.999)
                for i in range(nwalkers):
                    pos[i][j] = acc_lims[0] + (dif[i] + fact / 2.0)

        # instruments
        if 5 * kplanets + PACC < j < k_params + i_params + 1 + PACC:
            l += 2
            fact = sp.absolute(inslims[l] - inslims[l + 1]) / nwalkers
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)

            if (j - k_params - 1 - PACC) % i_params == 0:  # ojo aqui
                jitt_ini = sp.sort(
                    sp.fabs(sp.random.normal(0, 1, nwalkers))) * 0.1
                dif = jitt_ini * sp.random.uniform(0.9, 0.999)

            for i in range(nwalkers):
                pos[i][j] = inslims[l] + (dif[i] + fact / 2.0)
            # print(pos[j][:])
        if totcornum:
            if j > k_params + i_params + PACC:
                fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers

                dif = sp.arange(nwalkers) * fact * \
                    sp.random.uniform(0.8, 0.999)
                for i in range(nwalkers):
                    pos[i][j] = acc_lims[0] + (dif[i] + fact / 2.0)
                    # print(pos[i][j])

    pos = sp.array([pos for h in range(ntemps)])
    return pos


def pt_pos_rvpm(setup, *args):
    '''
    MAKE A WAY TO DIFFERENTIATE BETWEEN RV AND RVPM
    '''

    if args:
        kplanets, nins, boundaries = args[0], args[1], args[2]
        inslims, acc_lims, MOAV = args[3], args[4], args[5]
        totcornum, PACC = args[6], args[7]
    ntemps, nwalkers, nsteps = setup

    k_params = 5 * kplanets
    i_params = nins * 2 * (MOAV + 1)
    fsig, lenppm, nins_pm, boundaries_pm = args[-4:]  # PM ONLY
    ndim = k_params + i_params + totcornum + PACC + 1
    if kplanets > 0:
        if args:
            ndim += fsig * lenppm
    pos = sp.zeros((nwalkers, ndim))
    k, kk = -2, -2
    l, ll = -2, -2

    for j in range(ndim):
        if j < k_params:  # planetary params
            k += 2
            if j % 5 == 0:
                fact = sp.absolute(
                    boundaries[k] - boundaries[k + 1]) / nwalkers
                dif = sp.arange(nwalkers) * fact * \
                    sp.random.uniform(0.9, 0.999)
                pos[:, j] = boundaries[k] + (dif + fact / 2.0)
                act0 = pos
            else:
                fact = (sp.absolute(
                    boundaries[k] - boundaries[k + 1]) * 2) / (5 * nwalkers)
                dif = sp.arange(nwalkers) * fact * \
                    sp.random.uniform(0.9, 0.999)
                pos[:, j] = (boundaries[k + 1] + 3 * boundaries[k]
                             ) / 4 + (dif + fact / 2.0)
                act1 = pos
        if j == k_params:  # acc
            fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
            pos[:, j] = acc_lims[0] + (dif + fact / 2.0)
        if PACC:  # pacc
            if j == k_params + PACC:  # parabolic accel
                fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers
                dif = sp.arange(nwalkers) * fact * \
                    sp.random.uniform(0.9, 0.999)
                pos[:, j] = acc_lims[0] + (dif + fact / 2.0)
        act2 = pos
        # instruments
        if k_params + PACC < j < k_params + i_params + 1 + PACC:
            l += 2
            fact = sp.absolute(inslims[l] - inslims[l + 1]) / nwalkers
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)

            if (j - k_params - 1 - PACC) % i_params == 0:  # ojo aqui
                jitt_ini = sp.sort(
                    sp.fabs(sp.random.normal(0, 1, nwalkers))) * 0.1
                dif = jitt_ini * sp.random.uniform(0.9, 0.999)

            for i in range(nwalkers):
                pos[i][j] = inslims[l] + (dif[i] + fact / 2.0)
        if totcornum:
            if k_params + i_params + PACC < j < k_params + i_params + totcornum + PACC + 1:
                fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers

                dif = sp.arange(nwalkers) * fact * \
                    sp.random.uniform(0.8, 0.999)
                for i in range(nwalkers):
                    pos[i][j] = acc_lims[0] + (dif[i] + fact / 2.0)
        if kplanets > 0 and k_params + i_params + totcornum + PACC < j:
            if args:  # pm thingy
                kk += 2
                fact = sp.absolute(
                    boundaries_pm[kk] - boundaries_pm[kk + 1]) / nwalkers
                dif = sp.arange(nwalkers) * fact * \
                    sp.random.uniform(0.9, 0.999)
                pos[:, j] = (boundaries_pm[kk + 1] + 3 *
                             boundaries_pm[kk]) / 4 + (dif + fact / 2.)

    pos = sp.array([pos for h in range(ntemps)])

    return pos


def neo_p0(setup, *args):
    ntemps, nwalkers, nsteps = setup
    t = args[0]
    ndim = args[1]
    C = args[2]

    pos = sp.zeros((nwalkers, ndim))

    for j in range(ndim):
        boundaries = t[C[j]].lims
        fact = sp.absolute(boundaries[0] - boundaries[1]) / nwalkers
        rnd = sp.random.uniform(0.9, 0.9999)
        dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.9999)

        if (t[C[j]].prior=='uniform_spe_a' or t[C[j]].prior=='uniform_spe_b' or
            t[C[j]].prior=='uniform_spe_c' or t[C[j]].prior=='uniform_spe_d'):
            for i in range(nwalkers):
                pos[i][j] = (boundaries[1] + 3 * boundaries[0]) / \
                    4 + (dif[i] * 2. / 5. + fact / 2.0)
        elif t[C[j]].tag() == 'Jitter':
            jitt_ini = sp.sort(sp.fabs(sp.random.normal(0, 1, nwalkers))) * 0.1
            dif = jitt_ini * sp.random.uniform(0.9, 0.9999)
            for i in range(nwalkers):
                pos[i][j] = boundaries[0] + (dif[i] + fact/2.0)
                pos[i][j] *= 0.1
        elif t[C[j]].tag()=='MACoefficient':
            for i in range(nwalkers):
                pos[i][j] = boundaries[0] + (dif[i] + fact/2.0)
                #pos[i][j] *= 1
#                #print('bobos', boundaries[0], boundaries[1], pos[i][j])
        else:
            for i in range(nwalkers):
                pos[i][j] = boundaries[0] + (dif[i] + fact/2.0)
    #pos[:, 8] = pos[:, 8] ** 0.5
    #pos[:, 12] = pos[:, 12] ** 0.5
    pos = sp.array([pos for h in range(ntemps)])
    return pos


def ensure(condition, warning, MUSIC):
    try:
        assert condition
    except Exception as e:
        if MUSIC:
            MUSIC.play()
        assert condition, warning
    pass


def instigator(cherry_chain, cherry_post, all_data, saveplace):
    """Save chains and posteriors in a pickle file for later use."""
    save_chains(cherry_chain, saveplace)
    save_posteriors(cherry_post, saveplace)
    save_rv_data(all_data, saveplace)
    pass


def save_chains(chains, out_dir):
    """Pickle the chains."""
    pickle_out = open(out_dir + '/chains.pkl', 'wb')
    pickle.dump(chains, pickle_out)
    pickle_out.close()
    pass


def save_posteriors(posteriors, out_dir):
    """Pickle the posteriors."""
    pickle_out = open(out_dir + '/posteriors.pkl', 'wb')
    pickle.dump(posteriors, pickle_out)
    pickle_out.close()
    pass


def save_rv_data(all_data, out_dir):
    """Save radial-velocity data."""
    # TODO: save bisector.
    pickle_out = open(out_dir + '/rv_data.pkl', 'wb')
    pickle.dump(all_data, pickle_out)
    pickle_out.close()
    pass


def read_chains(in_dir):
    """Read chains file."""
    pickle_in = open(in_dir, 'rb')
    chains = pickle.load(pickle_in)
    pickle_in.close()
    return chains


def read_posteriors(in_dir):
    """Read posteriors file."""
    pickle_in = open(in_dir, 'rb')
    posteriors = pickle.load(pickle_in)
    pickle_in.close()
    return posteriors


def read_rv_data(in_dir):
    """Read radial-velocity pickle file."""
    pickle_in = open(in_dir, 'rb')
    all_rv_data = pickle.load(pickle_in)
    pickle_in.close()
    return all_rv_data


def phasefold(time, rv, err, period):
    """Phasefold an rv timeseries with a given period.

    Parameters
    ----------
    time : array_like
        An array containing the times of measurements.
    rv : array_like
        An array containing the radial-velocities.
    err : array_like
        An array containing the radial-velocity uncertainties.
    period : float
        The period with which to phase fold.

    Returns
    -------
    time_phased : array_like
        The phased timestamps.
    rv_phased : array_like
        The phased RVs.
    err_phased : array_like
        The phased RV uncertainties.

    """
    phases = (time / period) % 1
    sortIndi = sp.argsort(phases)  # sorts the points
    # gets the indices so we sort the RVs correspondingly(?)
    time_phased = phases[sortIndi]
    rv_phased = rv[sortIndi]
    err_phased = err[sortIndi]
    return time_phased, rv_phased, err_phased


def credibility_interval(post, alpha=.68):
    """Calculate bayesian credibility interval.

    Parameters:
    -----------
    post : array_like
        The posterior sample over which to calculate the bayesian credibility
        interval.
    alpha : float, optional
        Confidence level.
    Returns:
    --------
    med : float
        Median of the posterior.
    low : float
        Lower part of the credibility interval.
    up : float
        Upper part of the credibility interval.

    """
    lower_percentile = 100 * (1 - alpha) / 2
    upper_percentile = 100 * (1 + alpha) / 2
    low, med, up = sp.percentile(
        post, [lower_percentile, 50, upper_percentile]
    )
    return med, low, up
