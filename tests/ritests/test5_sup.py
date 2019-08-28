# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
import batman

def neo_init_batman(t):
    '''
    initializes batman
    '''
    n = {'t0': min(t), 'per': 1., 'rp': 0.1, 'a': 15.,
         'inc': 87., 'ecc':0., 'w':90.}
    params = batman.TransitParams()
    for x in n:
        setattr(params, x, n[x])

    params.limb_dark = 'quadratic'  # limb darkening model
    ld_coefs = sp.ones(2) * 0.5  # dummy coefficients  # not 1 # DEL
    params.u = ld_coefs

    model = batman.TransitModel(params, t)
    return model, params

def Model(param, x, bm, bp):
    T0, r = param
    b_pa = sp.array([x, 1, [2], [bm], [bp]])
    t_ = [T0, 24.73712, r, 101.1576001138329, 89.912, 0., 90., 0.1, 0.3]
    transit = neo_lightcurve(t_, b_pa)
    return transit

def neo_lightcurve(theta, params):
    time = params[0]
    kplanets = params[1]
    ldn, batman_m, batman_p = params[2], params[3], params[4]

    flux = 0.0
    #  thetas go in corrected

    for k in range(kplanets):
        np = int(sp.sum(ldn[:k])) + 7 * k
        # no ser huaso, usar setattr(params, 'n', v)

        batman_p[k].t0 = theta[np]
        batman_p[k].per = theta[np + 1]
        batman_p[k].rp = theta[np + 2]
        batman_p[k].a = theta[np + 3]
        batman_p[k].inc = theta[np + 4]
        batman_p[k].ecc = theta[np + 5]
        batman_p[k].w = theta[np + 6]
        batman_p[k].u = theta[np + 7:np + 7 + ldn[k]]

        flux += batman_m[k].light_curve(batman_p[k])  # calculates light curve
    return flux

def Model1(param, x2, bm, pm):
    T01, r1 = param
    b_pa1 = sp.array([x2, 1, [2], [bm], [pm]])
    t_1 = [T01, 24.73712, r1, 101.1576001138329, 89.912, 0., 90., 0.1, 0.3]
    transit1 = neo_lightcurve(t_1, b_pa1)
    return transit1


#
