# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp

a=sp.array(['RV_dataset1.vels', 'RV_dataset14.vels'])
aa=sp.array(['RV_dataset14.vels'])

def read_data(instruments):
    '''
    Data pre-processing
    '''
    nins = len(instruments)
    instruments = sp.array([sp.loadtxt('datafiles/'+x) for x in instruments])
    def data(data, ins_no):
        Time, Radial_Velocity, Err = data.T[:3]  # el error de la rv
        #Radial_Velocity -= sp.mean(Radial_Velocity)  # DEL for pm testing
        Flag = sp.ones(len(Time)) * ins_no  # marca el instrumento al q pertenece
        Staract = data.T[3:]
        return sp.array([Time, Radial_Velocity, Err, Flag, Staract])

    def sortstuff(tryin):
        t, rv, er, flag = tryin
        order = sp.argsort(t)
        return sp.array([x[order] for x in [t, rv, er, flag]])

    fd = sp.array([]), sp.array([]), sp.array([]), sp.array([])

    for k in range(len(instruments)):  # appends all the data in megarg
        t, rv, er, flag, star = data(instruments[k], k)
        fd = sp.hstack((fd, [t, rv, er, flag] ))  # ojo this, list not array

    #fd[0] = fd[0] - min(fd[0])  # min t
    alldat = sp.array([])
    #    try:
    staract = []
    for i in range(nins):
        hold = data(instruments[i], i)[4]
        staract.append(hold)
    #staract = sp.array([data(instruments[i], i)[4] for i in range(nins)])
    #print 'alr'
    #    except:
    #        staract = sp.array([sp.array([]) for i in range(nins)])
    #        print 'nr'
    starflag = sp.array([sp.array([i for k in range(len(staract[i]))]) for i in range(len(staract))])
    tryin = sortstuff(fd)
    for i in range(len(starflag)):
        for j in range(len(starflag[i])):
            staract[i][j] -= sp.mean(staract[i][j])
    totcornum = 0
    for correlations in starflag:
        if len(correlations) > 0:
            totcornum += len(correlations)
    #print fd[0]  # THISLINE
    #print sp.argsort(fd[0])  # THISLINE
    return tryin, staract, starflag, totcornum

class DATA:
    def __init__(self, instruments):
        # =sp.array(['RV_dataset1.vels', 'RV_dataset14.vels'
        self.nins = len(instruments)
        self.all_data = sp.array([sp.loadtxt('datafiles/'+x) for x in instruments])  # all_data[x] pickea dataset
        self.rv = sp.array([])
        self.activity = sp.array([])
        self.cornum = sp.array([])

        for i in range(self.nins):  # stack all data
            dat0, dat1, cornum = self.insert_labels(self.all_data[i], i)  # rvs, activities
            print(dat1, cornum)
            if i == 0:
                self.rv = dat0
            else:
                self.rv = sp.r_[self.rv, dat0]
            self.cornum = sp.append(self.cornum, cornum)

        self.rv_sorted = self.sortstuff(self.rv)
        self.activity = sp.array([self.insert_labels(self.all_data[i], i)[1] for i in range(self.nins)])
        #print self.activity
        #self.act_sorted = self.sortstuff(self.activity)


    def insert_labels(self, data, ins_no):
        flag = sp.ones_like(data.T[0]) * ins_no  # flags
        holder = sp.c_[data[:, :3], flag]  # rvs
        if data[:, 3:].size > 0:
            holder1 = sp.c_[data[:, 0], data[:, 3:], flag]  # activity
            #print holder1.shape
            cornum = len(holder1.T) - 2
        else:
            #print data[:, 3:]
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
    '''
    Data pre-processing
    '''
    fnins = len(instruments)
    instruments = sp.array([sp.loadtxt('datafiles/'+x) for x in instruments])
    def data(data, ins_no):
        Time, Radial_Velocity = data.T[:2]  # el error de la rv
        Radial_Velocity -= sp.mean(Radial_Velocity)
        Flag = sp.ones(len(Time)) * ins_no  # marca el instrumento al q pertenece
        return sp.array([Time, Radial_Velocity, Flag])

    def sortstuff(tryin):
        t, rv, flag = tryin
        order = sp.argsort(t)
        return sp.array([x[order] for x in [t, rv, flag]])

    fd = sp.array([]), sp.array([]), sp.array([])

    for k in range(len(instruments)):  # appends all the data in megarg
        t, rv, flag = data(instruments[k], k)
        fd = sp.hstack((fd, [t, rv, flag] ))  # ojo this, list not array

    fd[0] = fd[0] - min(fd[0])
    alldat = sp.array([])

    tryin = sortstuff(fd)

    return fd, staract, starflag, totcornum


def normal_pdf(x, mean, variance):
    var = 2 * variance
    return ( - (x - mean) ** 2 / var)


def gaussian(x, sigma):
    coef = -(x*x)/(2*sigma*sigma)
    return 1/np.sqrt(2*np.pi*sigma*sigma) * np.exp(coef)


def pt_pos(setup, *args):

    if args:
        kplanets, nins, boundaries = args[0], args[1], args[2]
        inslims, acc_lims, MOAV = args[3], args[4], args[5]
        totcornum, PACC = args[6], args[7]

    ntemps, nwalkers, nsteps = setup
    k_params = 5 * kplanets
    i_params = nins*2*(MOAV+1)

    ndim = 1 + k_params + i_params + totcornum + PACC
    pos = sp.zeros((nwalkers, ndim))
    k = -2
    l = -2
    ll = -2  ##
    for j in range(ndim):
        if j < k_params:
            k += 2
            if j%5==0:
                fact = sp.absolute(boundaries[k] - boundaries[k+1]) / nwalkers
            else:
                #fact = sp.absolute(boundaries[k]) / (self.nwalkers)
                fact = (sp.absolute(boundaries[k] - boundaries[k+1]) * 2) / (5 * nwalkers)
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
            for i in range(nwalkers):
                if j%5==0:
                    pos[i][j] = boundaries[k] + (dif[i] + fact/2.0)
                else:
                    #pos[i][j] = boundaries[k] * 0.5 + (dif[i] + fact/2.0)
                    pos[i][j] = (boundaries[k+1]+3*boundaries[k])/4 + (dif[i] + fact/2.0)
        if j == 5 * kplanets:  # acc
            fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
            for i in range(nwalkers):
                pos[i][j] = acc_lims[0] + (dif[i] + fact/2.0)
        if PACC:
            if j == 5 * kplanets + PACC:  # parabolic accel
                fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers
                dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
                for i in range(nwalkers):
                    pos[i][j] = acc_lims[0] + (dif[i] + fact/2.0)

        # instruments
        if 5 * kplanets + PACC < j < k_params + i_params + 1 + PACC:
            l += 2
            fact = sp.absolute(inslims[l] - inslims[l+1]) / nwalkers
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)

            if (j-k_params-1-PACC) % i_params == 0:  # ojo aqui
                jitt_ini = sp.sort(sp.fabs(sp.random.normal(0, 1, nwalkers))) * 0.1
                dif = jitt_ini * sp.random.uniform(0.9, 0.999)

            for i in range(nwalkers):
                pos[i][j] = inslims[l] + (dif[i] + fact/2.0)
            #print(pos[j][:])
        if totcornum:
            if j > k_params + i_params + PACC:
                fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers

                dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.8, 0.999)
                for i in range(nwalkers):
                    pos[i][j] = acc_lims[0] + (dif[i] + fact/2.0)
                    #print(pos[i][j])

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
    i_params = nins*2*(MOAV+1)
    fsig, lenppm, nins_pm, boundaries_pm = args[-4:]  # PM ONLY
    ndim = k_params + i_params + totcornum + PACC + 1
    if kplanets > 0:
        if args:
            ndim += fsig*lenppm
    pos = sp.zeros((nwalkers, ndim))
    k, kk = -2, -2
    l, ll = -2, -2

    for j in range(ndim):
        if j < k_params:  # planetary params
            k += 2
            if j%5==0:
                fact = sp.absolute(boundaries[k] - boundaries[k+1]) / nwalkers
                dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
                pos[:, j] = boundaries[k] + (dif + fact/2.0)
                act0 = pos
            else:
                fact = (sp.absolute(boundaries[k] - boundaries[k+1]) * 2) / (5 * nwalkers)
                dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
                pos[:, j] = (boundaries[k+1]+3*boundaries[k])/4 + (dif + fact/2.0)
                act1 = pos
        if j == k_params:  # acc
            fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
            pos[:, j] = acc_lims[0] + (dif + fact/2.0)
        if PACC:  # pacc
            if j == k_params + PACC:  # parabolic accel
                fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers
                dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
                pos[:, j] = acc_lims[0] + (dif + fact/2.0)
        act2 = pos
        # instruments
        if k_params + PACC < j < k_params + i_params + 1 + PACC:
            l += 2
            fact = sp.absolute(inslims[l] - inslims[l+1]) / nwalkers
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)

            if (j-k_params-1-PACC) % i_params == 0:  # ojo aqui
                jitt_ini = sp.sort(sp.fabs(sp.random.normal(0, 1, nwalkers))) * 0.1
                dif = jitt_ini * sp.random.uniform(0.9, 0.999)

            for i in range(nwalkers):
                pos[i][j] = inslims[l] + (dif[i] + fact/2.0)
        if totcornum:
            if k_params + i_params + PACC < j < k_params + i_params + totcornum + PACC + 1:
                fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers

                dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.8, 0.999)
                for i in range(nwalkers):
                    pos[i][j] = acc_lims[0] + (dif[i] + fact/2.0)
        if kplanets > 0 and k_params + i_params + totcornum + PACC < j:
            if args:  # pm thingy
                kk += 2
                fact = sp.absolute(boundaries_pm[kk] - boundaries_pm[kk+1]) / nwalkers
                dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
                pos[:, j] = (boundaries_pm[kk+1] + 3*boundaries_pm[kk])/4 + (dif + fact/2.)

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
        fact = sp.absolute(boundaries[0]-boundaries[1]) / nwalkers
        rnd = sp.random.uniform(0.9, 0.9999)
        dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.9999)

        if t[C[j]].prior=='uniform_spe' or t[C[j]].prior=='joined':
            for i in range(nwalkers):
                pos[i][j] = (boundaries[1]+3*boundaries[0])/4 + (dif[i]*2./5. + fact/2.0)
        elif t[C[j]].tag()=='Jitter':
            jitt_ini = sp.sort(sp.fabs(sp.random.normal(0, 1, nwalkers))) * 0.1
            dif = jitt_ini * sp.random.uniform(0.9, 0.9999)
            for i in range(nwalkers):
                pos[i][j] = boundaries[0] + (dif[i] + fact/2.0)

        else:
            for i in range(nwalkers):
                pos[i][j] = boundaries[0] + (dif[i] + fact/2.0)
    pos = sp.array([pos for h in range(ntemps)])
    return pos


def ensure(condition, warning, MUSIC):
    try:
        assert condition
    except:
        if MUSIC:
            MUSIC.play()
        assert condition, warning
    pass
#
