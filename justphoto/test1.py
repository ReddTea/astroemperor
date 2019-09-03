#!/usr/bin/env python
# -*- coding: utf-8 -*-

if True:  # imports
    import pyfits as py
    import numpy as np
    import matplotlib.pyplot as plt
    from numba import jit

    import george
    from george import kernels
    import emcee
    import corner

    import batman

    from tqdm import tqdm

    Neptune_Density = 1638 #kg/m3.
    Jupiter_Density = 1326 #kg/m3.
    Earth_Density = 5520 #kg/m3.
    Earth_radius = 6371000 #m
    Jupiter_Radius = 69911000 #m
    Jupiter_Mass = 1.898e+27 #kg
    Sun_radius = 695700000 #m
    au = 149597870700 #m

# points to add
madness = 0#12000
plot_new = True
plot_all = True

nwalkers = 50
nsteps = 1000

burn_out1 = nsteps//5
# some functions

def init_batman(time):
    p = batman.TransitParams()
    n = {'t0': 0., 'per': 1., 'rp': 0.1, 'a': 15.,
         'inc': 87., 'ecc':0., 'w':90.}
    for x in n:
        setattr(p, x, n[x])

    p.limb_dark = 'quadratic'
    p.u = [0.1, 0.3]
    m = batman.TransitModel(p, time)
    return p, m

def transit_lightCurve(time, t0, radius, dist, P, inc):
    p = batman.TransitParams()
    p.t0 = t0                       #time of inferior conjunction
    p.per = P                       #orbital period in days
    p.rp = radius                   #planet radius (in units of stellar radii)
    p.a = dist                      #semi-major axis (in units of stellar radii)
    p.inc = inc                     #orbital inclination (in degrees)
    p.ecc = 0.                      #eccentricity
    p.w = 0.                        #longitude of periastron (in degrees)
    p.u = [0.1, 0.3]                #limb darkening coefficients [u1, u2]
    p.limb_dark = "quadratic"       #limb darkening model

    m = batman.TransitModel(p, time)    #initializes model
    flux = m.light_curve(p)          #calculates light curve

    return (flux)

def RadiusPlanet(density, Mass, StellarRadius): # in jupiter mass and solar radius
    SR = StellarRadius*Sun_radius
    M = Mass*Jupiter_Mass
    r = (3*(M/density) /(4*np.pi))**(1/3)
    return r/SR  # radius in star ratio

def distancePlanet(distance, StellarRadius): # in au, and solar radius
    d = distance*au
    SR = StellarRadius*Sun_radius
    return d/SR  # distance in star ratio

def gaussian(x, sigma):
    coef = -(x*x)/(2*sigma*sigma)
    return 1/np.sqrt(2*np.pi*sigma*sigma) * np.exp(coef)

# loads data
x, y, x_error, y_error = np.load('transit_ground_r.npy')

# initialize batman
P, M = init_batman(x)

# creates new points n=madness
xa = np.sort(np.random.uniform(min(x), max(x), madness))
xarr = np.random.normal(np.mean(x_error), np.std(x_error), madness)
yarr = np.random.normal(np.mean(y_error), np.std(y_error), madness)

ya = transit_lightCurve(xa, 2456915.6997, 0.0704, 101.1576001138329, 24.73712, 89.912)
ya *= np.random.uniform(0.998, 1.003, madness)

if plot_new:
    plt.subplots(figsize=(16,8))
    plt.grid(True)
    plt.xlim( (min(x)-0.01) , (max(x+0.01)))
    plt.ylim(0.99, 1.015)

    plt.errorbar(x, y, xerr=x_error, yerr=y_error, fmt='b.', alpha=1/1.)
    plt.errorbar(xa, ya, xerr=xarr, yerr=yarr, fmt='r.', alpha=1/1.)
    plt.show()

xx = np.append(x, xa)
xx_error = np.append(x_error, xarr)
yy = np.append(y, ya)
yy_error = np.append(y_error, yarr)

orden = np.argsort(xx)

xx, yy, xx_error, yy_error = xx[orden], yy[orden], xx_error[orden], yy_error[orden]

if plot_all:
    plt.subplots(figsize=(16,8))
    plt.grid(True)
    plt.xlim( (min(xx)-0.01) , (max(xx+0.01)))
    plt.ylim(0.99, 1.015)

    plt.plot((xx[0], xx[-1]), (1.0 - 0.1/100, 1.0 - 0.1/100), 'k--', linewidth=2, alpha = 0.5)
    plt.plot((xx[0], xx[-1]), (1.0 + 0.1/100 , 1. + 0.1/100), 'k--', linewidth=2, alpha = 0.5)
    plt.plot((xx[0], xx[-1]), (1., 1.), 'k', linewidth=4)

    #print(xx)
    plt.errorbar(xx, yy, xerr=xx_error, yerr=yy_error, fmt='b.', alpha=1/1.)
                             #time, t0,     radius,        dist,            P,        inc
    y_transit = transit_lightCurve(x, 2456915.6997, 0.0704, 101.1576001138329, 24.73712, 89.912)

    yy_transit = transit_lightCurve(xx, 2456915.6997, 0.0704, 101.1576001138329, 24.73712, 89.912)
    #print(yy_transit)
    plt.plot(xx, yy_transit, 'r',  linewidth=2)

    plt.ylabel('Normalized Flux', fontsize=15)
    plt.xlabel('JD', fontsize=15)
    plt.title('GROND in g')
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)

    plt.show()

# more funcs to feed emcee

def give_initial(nwalkers):
    ndim = 4
    T0 = 2456915.70
    r = 0.0704
    k_a = 1.0e-3 # amplitud kernel
    k_r = 3 # radial kernel

    param = []
    for i in range(nwalkers):
        t1 = T0  + 1e-3*np.random.randn() # Normal distribution 1-1 4-4
        t2 = r   + 1e-4*np.random.randn()
        t6 = k_a + 1*1e-4*np.random.randn()
        t7 = k_r + 1*1e-0*np.random.randn()
        param.append((t1,t2,np.abs(t6),np.abs(t7)))
    return param, ndim

def give_initial_pt(nwalkers):
    ndim = 4
    ntemp = 2
    T0 = 2456915.70
    r = 0.0704
    k_a = 1.0e-3 # amplitud kernel
    k_r = 3 # radial kernel

    param = []
    for i in range(nwalkers):
        t1 = T0  + 1e-3*np.random.randn() # Normal distribution 1-1 4-4
        t2 = r   + 1e-4*np.random.randn()
        t6 = k_a + 1*1e-4*np.random.randn()
        t7 = k_r + 1*1e-0*np.random.randn()
        param.append((t1,t2,np.abs(t6),np.abs(t7)))
    param = np.array([param for _ in range(ntemp)])
    return param, ndim

def plus_random_initial(param, nwalkers):
    T0, r, k_a, k_r = param
    t = []
    for i in range(nwalkers):
        t1 = T0  + 1e-4*np.random.randn() # Normal distribution 1-1 4-4
        t2 = r   + 1e-5*np.random.randn()
        t6 = k_a + 1*1e-4*np.random.randn()
        t7 = k_r + 1*1e-0*np.random.randn()
        t.append((t1,t2, np.abs(t6),np.abs(t7)))
    return t

def lnprob_gp(param, x, y, yerr):
    lp = lnprior(param)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(param, x, y, yerr)

def lnprior(param):
    T0, r, k_a, k_r = param
    if not (2456915.67 < T0  < 2456915.73 and 0.05 < r  < 0.09 and
        0 < k_a < 5.0e-2 and 0 < k_r < 6.):
        return -np.inf

    mu = 0.07 # mean of the Normal prior
    sigma = 0.004 # standard deviation of the Normal prior
    prob_r = np.log(gaussian(r - mu, sigma))

    mu = 2456915.70
    sigma =  0.005
    prob_t0 = np.log(gaussian(T0 - mu, sigma))
    prob = prob_t0
    return prob

def lnlike_gp(param, x, y, yerr):
    radius = 10.**param[-1]
    gp = george.GP(param[-2] * kernels.Matern32Kernel(radius))
    gp.compute(x, yerr)
    return gp.lnlikelihood(y - Model(param[:-2], x))

def Model(param, x):
    T0, r = param
    transit = transit_lightCurve(x, T0, r, 101.1576001138329, 24.73712, 89.912)
    t = x - x[0]
    return transit


data = (xx, yy, yy_error)

p0, ndim = give_initial_pt(nwalkers)
sampler = emcee.PTSampler(2, nwalkers, ndim, lnlike_gp, lnprior, loglargs=[xx, yy, yy_error])

print("Running burn-in")

pbar = tqdm(burn_out1)
for p, lnprob, lnlike in sampler.sample(p0, iterations=burn_out1):
    pass


if plot_chain1:
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 12))
    import matplotlib.gridspec as gridspec
    gs1 = gridspec.GridSpec(4, 1)
    gs1.update(wspace=0.025)

    axes[0].plot(sampler.chain[0, :, :, 0].T, color="k", alpha=0.2)
    #axes[0].yaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_ylabel("$T0$")

    axes[1].plot(sampler.chain[0, :, :, 1].T, color="k", alpha=0.2)
    #axes[1].yaxis.set_major_locator(MaxNLocator(5))
    axes[1].set_ylabel("$r$")

    axes[2].plot(sampler.chain[0, :, :, 2].T, color="k", alpha=0.2)
    #axes[5].yaxis.set_major_locator(MaxNLocator(5))
    axes[2].set_ylabel("$k a$")

    axes[3].plot(sampler.chain[0, :, :, 3].T, color="k", alpha=0.2)
    #axes[6].yaxis.set_major_locator(MaxNLocator(5))
    axes[3].set_ylabel("$k r$")

    fig.tight_layout(h_pad=0.0)
    plt.show()













#
