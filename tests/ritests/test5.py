# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python
# -*- coding: utf-8 -*-
if True:
    import pyfits as py
    import numpy as np
    import matplotlib.pyplot as plt
    from numba import jit
    import scipy as sp
    import george
    from george import kernels
    import emcee
    import corner
    import batman


PLOT1 = False
PLOT2 = False
PLOT3 = False

import test5_sup as t5s


def neo_init_george(kernels):
    '''
    kernels should be a matrix
    rows +, columns *, ie
    [[k1, k2], [k3, k4]]
    k_out = c*k1*k2+c*k3*k4
    '''
    k_out = K['Constant']
    for func in kernels[0]:
        k_out *= K[func]
    for i in range(len(kernels)):
        if i == 0:
            pass
        else:
            k = K['Constant']
            for func in kernels[i]:
                k *= K[func]
            k_out += k
    return k_out


def gaussian(x, sigma):
    coef = -(x*x)/(2*sigma*sigma)
    return 1/sp.sqrt(2*sp.pi*sigma*sigma) * sp.exp(coef)


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

x, y, x_error, y_error = sp.load('transit_ground_r.npy')

# initialize batman
M, P = t5s.neo_init_batman(x)
create = 15000
if create:

    madness = create
    xa = np.sort(np.random.uniform(2458042, 2458048, madness))
    yarr = np.random.normal(np.mean(y_error), np.std(y_error), madness)


    Ma, Pa = t5s.neo_init_batman(xa)
    b_pa = sp.array([xa, 1, [2], [Ma], [Pa]])
    # 52-
          # t0      per   rp    a      inc    ecc  w    a1   a2
    t_ = [2458042.0, 3.1, 0.07, 0.025, 89.8, 0.1, 90., 0.1, 0.3]
    ya = t5s.neo_lightcurve(t_, b_pa)

                                #t0         rp        # a
    #ya = transit_lightCurve(xa, 2456915.6997, 0.0704, 101.1576001138329, 24.73712, 89.912)
    ya *= np.random.uniform(0.999, 1.001, madness)
    xa1 = sp.linspace(min(xa), max(xa), 10000)
    Ma1, Pa1 = t5s.neo_init_batman(xa1)
    b_pa1 = sp.array([xa, 1, [2], [Ma1], [Pa1]])
    y_transit = t5s.neo_lightcurve(t_, b_pa1)


    plt.subplots(figsize=(10,8))
    plt.xlim( (min(xa)-0.01) , (max(xa+0.01)))
    plt.ylim(min(ya)-0.001, 2-(min(ya)-0.001))
    plt.plot((xa[0], xa[-1]), (1.0 - 0.1/100, 1.0 - 0.1/100), 'k--', linewidth=2, alpha = 0.5)
    plt.plot((xa[0], xa[-1]), (1.0 + 0.1/100 , 1. + 0.1/100), 'k--', linewidth=2, alpha = 0.5)
    plt.plot((xa[0], xa[-1]), (1., 1.), 'k', linewidth=4)


    plt.errorbar(xa, ya, yerr=yarr, fmt='b.', alpha=0.5/1.)
    plt.ylabel(' Flux', fontsize=26)
    plt.xlabel('JD', fontsize=26)
    plt.title('LP 834-9', fontsize=30)
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.savefig('synth_3_dat.png')

    plt.plot(xa1, y_transit, 'r',  linewidth=2)
    plt.show()
    sp.savetxt('synth3.flux', sp.array([xa,ya,yarr]).T)


if PLOT1:
    b_pa = sp.array([x, 1, [2], [M], [P]])
    t_ = [2456915.6997, 24.73712, 0.0704, 101.1576001138329, 89.912, 0., 90., 0.1, 0.3]
    y_ = t5s.neo_lightcurve(t_, b_pa)

    plt.subplots(figsize=(16,8))
    plt.grid(True)
    plt.xlim( (min(x)-0.01) , (max(x+0.01)))
    plt.ylim(0.99, 1.015)

    plt.errorbar(x, y_, yerr=y_error, fmt='b.', alpha=1/1.)
    plt.show()

if PLOT2:
    plt.subplots(figsize=(16,8))
    plt.grid(True)
    plt.xlim( (min(x)-0.01) , (max(x+0.01)))
    plt.ylim(0.99, 1.015)

    plt.plot((x[0], x[-1]), (1.0 - 0.1/100, 1.0 - 0.1/100), 'k--', linewidth=2, alpha = 0.5)
    plt.plot((x[0], x[-1]), (1.0 + 0.1/100 , 1. + 0.1/100), 'k--', linewidth=2, alpha = 0.5)
    plt.plot((x[0], x[-1]), (1., 1.), 'k', linewidth=4)

    plt.errorbar(x, y, yerr=y_error, fmt='b.', alpha=1/1.1)

    b_pa = sp.array([x, 1, [2], [M], [P]])
    t_ = [2456915.6997, 24.73712, 0.0704, 101.1576001138329, 89.912, 0., 90., 0.1, 0.3]
    y_transit = t5s.neo_lightcurve(t_, b_pa)

    plt.plot(x, y_transit, 'r',  linewidth=2)

    plt.ylabel('Normalized Flux', fontsize=15)
    plt.xlabel('JD', fontsize=15)
    plt.title('GROND in g')
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.show()
    pass

data = (x, y, y_error)
nwalkers = 50
p0, ndim = give_initial_pt(nwalkers)

def lnprior(param):
    T0, r, k_a, k_r = param
    if not (2456915.67 < T0  < 2456915.73 and 0.05 < r  < 0.09 and
        0 < k_a < 5.0e-2 and 0 < k_r < 6.):
        return -np.inf

    '''
    mu = 0.07 # mean of the Normal prior
    sigma = 0.004 # standard deviation of the Normal prior
    prob_r = np.log(gaussian(r - mu, sigma))

    mu = 2456915.70
    sigma =  0.005
    prob_t0 = np.log(gaussian(T0 - mu, sigma))
    prob = prob_t0
    '''
    return 0

K = {'Constant': 2. ** 2,
     'ExpSquaredKernel': kernels.ExpSquaredKernel(metric=1.**2),
     'ExpSine2Kernel': kernels.ExpSine2Kernel(gamma=1.0, log_period=1.0),
     'Matern32Kernel': kernels.Matern32Kernel(2.)}

def neo_init_george(kernels):
    k_out = K['Constant']
    for func in kernels[0]:
        k_out *= K[func]
    for i in range(len(kernels)):
        if i == 0:
            pass
        else:
            k = K['Constant']
            for func in kernels[i]:
                k *= K[func]
            k_out += k
    gp = george.GP(k_out)
    return gp

G = neo_init_george(sp.array([['Matern32Kernel']]))

def lnlike_gp(param, x, y, yerr, l_param, b_param):
    radius = 10.**param[-1]
    theta_g = sp.array([param[-2], radius])
    gp = l_param

    bm, bp = b_param

    #gp = george.GP(param[-2] * kernels.Matern32Kernel(radius))
    gp.set_parameter_vector(theta_g)

    gp.compute(x, yerr)
    return gp.lnlikelihood(y - t5s.Model(param[:-2], x, bm, bp))




'''
sampler = emcee.PTSampler(2, nwalkers, ndim, lnlike_gp, lnprior, loglargs=[x, y, y_error, G, [M, P]])
print("Running burn-in")
#%time p0, lnp, _ = sampler.run_mcmc(p0, 200)

for p, lnprob, lnlike in sampler.sample(p0, iterations=200):
    pass


sampler.reset()

p0, lnprob0, lnlike0 = p, lnprob, lnlike
sampler.reset()
print('\n ---------------------- CHAIN ---------------------- \n')
for p, lnprob, lnlike in sampler.sample(p0, lnprob0=lnprob0,
                                        lnlike0=lnlike0,
                                        iterations=1000):
    pass

samples = sampler.chain[0, :, -1000:, :].reshape((-1, ndim))
sampler.lnprobability

T0_f, r_f, ka_f, kr_f = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
print('T0 = ' + str(T0_f))
print('r = ' + str(r_f))
print('k_a = ' + str(ka_f))
print('k_r = ' + str(kr_f))
'''
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
        }

x2 = np.linspace(min(x), max(x), 1000)
M1, P1 = t5s.neo_init_batman(x2)

if PLOT3:
    plt.subplots(figsize=(16,8))
    plt.grid(True)
    plt.xlim( (min(x)-0.01) , (max(x+0.01)))
    plt.ylim(0.99, 1.015)
    plt.plot((x[0], x[-1]), (1., 1.), 'k', linewidth=4)

    plt.errorbar(x, y, yerr=y_error, fmt='b.', alpha=1/1.)

    y_transit = t5s.Model([T0_f[0], r_f[0]], x, M, P)

    M2, P2 = t5s.neo_init_batman(x)
    y_transit2 = t5s.Model([T0_f[0]*1.01, r_f[0]*1.01], x, M2, P2)


    plt.plot(x, y_transit, 'r',  linewidth=2)

    plt.ylabel('Normalized Flux', fontsize=15)
    plt.xlabel('JD', fontsize=15)
    plt.title('GROND in i band', fontsize=40)
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.subplots_adjust(left=0.15)

    r_t = str(np.around(r_f[0], decimals=4))
    r_tp = str(np.around(r_f[1], decimals=4))
    r_tm = str(np.around(r_f[2], decimals=4))
    plt.text(x[0]+0.06, 1.007, 'r = '+ r_t , fontdict=font)
    plt.text(x[0]+0.10, 1.0075, '+ '+ r_tp, fontdict=font)
    plt.text(x[0]+0.102, 1.0065, '-  '+ r_tm, fontdict=font)

    if True:
        for s in samples[np.random.randint(len(samples), size=24)]:
            radius = 10.**s[-1]
            t_g = sp.array([s[-2], radius])
            gp = george.GP(s[-2]* kernels.Matern32Kernel(radius))
            #G.set_parameter_vector(t_g)
            #gp = george.GP(0.2* kernels.Matern32Kernel(radius))
            gp.compute(x, y_error)
            m = gp.sample_conditional(y - t5s.Model(s[:-2], x, M, P), x2) + t5s.Model1(s[:-2], x2, M1, P1)
            plt.plot(x2, m, '-', color="#4682b4", alpha=0.2)
        plt.show()


#
