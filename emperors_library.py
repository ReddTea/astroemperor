# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
# -*- coding: utf-8 -*-
import pickle
import scipy as sp
from scipy.special import erf

a = sp.array(['RV_dataset1.vels', 'RV_dataset14.vels'])
aa = sp.array(['RV_dataset14.vels'])


def read_data(instruments, data_type='rv_file'):
    """Data pre-processing."""
    nins = len(instruments)
    instruments = sp.array([sp.loadtxt('datafiles/' + x) for x in instruments])

    def data(data, ins_no):
        Time, Radial_Velocity, Err = data.T[:3]  # el error de la rv
        if data_type == 'rv_file':
            Radial_Velocity -= sp.mean(Radial_Velocity)  # DEL for pm testing
        # marca el instrumento al q pertenece
        Flag = sp.ones(len(Time)) * ins_no
        Staract = data.T[3:].tolist()
        return sp.array([Time, Radial_Velocity, Err, Flag, Staract])

    def sortstuff(tryin):
        t, rv, er, flag = tryin
        order = sp.argsort(t)
        return sp.array([x[order] for x in [t, rv, er, flag]])

    fd = sp.array([]), sp.array([]), sp.array([]), sp.array([])

    for k in range(len(instruments)):  # appends all the data in megarg
        t, rv, er, flag, star = data(instruments[k], k)
        fd = sp.hstack((fd, [t, rv, er, flag]))  # ojo this, list not array
    tryin = sortstuff(fd)

    alldat = sp.array([])

    staract = []
    starflag = sp.array([])
    for i in range(nins):
        hold = data(instruments[i], i)[4]
        if sp.any(hold):
            starflag = sp.append(starflag, sp.ones(len(hold)) * i)
            for j in range(len(hold)):
                something = hold[j] - sp.mean(hold[j])  # normalize by rms
                something /= sp.sqrt(sp.mean(something**2))
                staract.append(something)

    totcornum = len(starflag)
    return tryin, staract, starflag, totcornum


def normal_pdf(x, mean, variance):
    var = 2 * variance
    return (- (x - mean) ** 2 / var)


def gaussian(x, sigma):
    coef = -(x * x) / (2 * sigma * sigma)
    return 1 / np.sqrt(2 * np.pi * sigma * sigma) * np.exp(coef)


def hist_gaussian(x, mu, sig):
    return sp.exp(-sp.power((x - mu) / sig, 2.) / 2.)


def neo_p0(setup, *args):
    ntemps, nwalkers, nsteps = setup
    t = args[0]
    ndim = args[1]
    C = args[2]

    pos = sp.zeros((ntemps, nwalkers, ndim))
    for temp in range(ntemps):
        for j in range(ndim):
            boundaries = t[C[j]].lims
            fact = sp.absolute(boundaries[0] - boundaries[1]) / nwalkers
            rnd = sp.random.uniform(0.9, 0.9999)
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.9999)
            if (t[C[j]].cv and t[C[j]].tag()!='Period'):
                for i in range(nwalkers):
                    pos[temp][i][j] = (boundaries[1] + 3 * boundaries[0]) / \
                        4 + (dif[i] * 2. / 5. + fact / 2.0)
            elif t[C[j]].tag() == 'Jitter':
                jitt_ini = sp.sort(sp.fabs(sp.random.normal(0, 1, nwalkers))) * 0.1
                dif = jitt_ini * sp.random.uniform(0.9, 0.9999)
                for i in range(nwalkers):
                    pos[temp][i][j] = boundaries[0] + (dif[i] + fact/2.0)
                    pos[temp][i][j] *= 0.1

            else:
                for i in range(nwalkers):
                    pos[temp][i][j] = boundaries[0] + (dif[i] + fact/2.0)

    return pos


def ensure(condition, warning, MUSIC):
    try:
        assert condition
    except Exception as e:
        if MUSIC:
            MUSIC.play()
        assert condition, warning
    pass


def instigator(setup, theta, cherry_chain, cherry_post, all_data, saveplace):
    """Save chains and posteriors in a pickle file for later use."""
    save(setup, saveplace, 'setup')
    save(theta, saveplace, 'theta')
    save(cherry_chain, saveplace, 'chains')
    save(cherry_post, saveplace, 'posteriors')
    save(all_data, saveplace, 'rv_data')
    pass


def save(obj, out_dir, type):
    """Pickle the given object.

    It pickles either the setup (nx1 arr), theta object, cherry_chain,
    cherry_post or all rv data into a pkl file for later use.
    """
    pickle_out = open(out_dir + '/' + type + '.pkl', 'wb')
    pickle.dump(obj, pickle_out)
    pickle_out.close()


def read(in_dir):
    """Read pickled files."""
    pickle_in = open(in_dir, 'rb')
    obj = pickle.load(pickle_in)
    pickle_in.close()
    return obj


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


def credibility_interval(post, alpha=1.):
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
    z = erf(alpha/sp.sqrt(2))

    lower_percentile = 100 * (1 - z) / 2
    upper_percentile = 100 * (1 + z) / 2
    low, med, up = sp.percentile(
        post, [lower_percentile, 50, upper_percentile]
    )
    return med, low, up
