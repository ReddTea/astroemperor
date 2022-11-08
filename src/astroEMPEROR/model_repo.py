# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# version 0.1.0
# date 19 jan 2021

import numpy as np
import kepler


def _model_sinusoid(theta, time, rv, err2, flags, *modargs):
    per, amp, pha = theta
    mod = amp * np.cos((2 * np.pi / per) * time + pha)

    ferr2 = np.zeros_like(mod)

    return mod, ferr2


def _model_acc(theta, time, rv, err2, flags, *modargs):
    mod = np.polyval(np.r_[theta, 0], time-time[0])
    #res = rv - mod

    ferr2 = np.zeros_like(mod)
    return mod, ferr2


def _model_instrument(theta, time, rv, err2, flags, *modargs):
    ins_no = modargs[0]

    offset, jitter = theta[:2]
    my_mask = (flags == ins_no)

    mod = offset * my_mask  # OFFSET

    new_err = my_mask * jitter ** 2

    return mod, new_err


def _model_staract(theta, time, rv, err2, flags, *modargs):
    ins_no = modargs[0][0]
    staracts = modargs[0][1]
    my_mask = (flags == ins_no)

    mod = np.zeros_like(flags)
    for j in range(len(theta)):
        mod += my_mask * theta[j] * staracts[j]

    new_err = np.zeros_like(mod)

    return mod, new_err


def _model_moav(theta, time, rv, err2, flags, *modargs):
    # time and residuals should exclusively be the ones for this instrument
    ins_no, maorder = modargs[0][:2]

    # apply offset
    offset, jitter = theta[:2]
    #print('in model 2', offset*[flags == ins_no])

    my_mask = (flags == ins_no)
    mod = offset * my_mask  # OFFSET
    #residuals = rv - mod


    if maorder > 0:
        residuals = modargs[0][2]
        theta_ma = theta[2:]
        t_ = time[my_mask]
        for i in range(len(t_)):
            for c in range(maorder):
                if i > c:
                    dt = abs(t_[i] - t_[i - 1 - c])
                    macoef = theta_ma[2 * c]
                    matime = theta_ma[2 * c + 1]
                    MA = macoef * np.exp(-dt / matime) * residuals[my_mask][i - 1 - c]
                    mod[my_mask][i] += MA
                    residuals[my_mask][i] -= MA

    new_err = my_mask * jitter ** 2
    return mod, new_err


def _model_moav_sa(theta, time, rv, err2, flags, *modargs):
    # time and residuals should exclusively be the ones for this instrument
    ins_no, maorder, staracts, cornum = modargs[0][:4]

    # apply offset
    offset, jitter = theta[:2]
    #print('in model 2', offset*[flags == ins_no])

    my_mask = (flags == ins_no)
    mod = offset * my_mask  # OFFSET
    #residuals = rv - mod

    if cornum:
        for j in range(cornum):
            mod[my_mask] += theta[2 * (maorder + 1) + j] * staracts[j]

    if maorder > 0:
        residuals = modargs[0][-1]
        theta_ma = theta[2:]
        t_ = time[my_mask]
        for i in range(len(t_)):
            for c in range(maorder):
                if i > c:
                    dt = abs(t_[i] - t_[i - 1 - c])
                    macoef = theta_ma[2 * c]
                    matime = theta_ma[2 * c + 1]
                    MA = macoef * np.exp(-dt / matime) * residuals[my_mask][i - 1 - c]
                    mod[my_mask][i] += MA
                    residuals[my_mask][i] -= MA

    new_err = np.zeros_like(mod)

    return mod, new_err



    new_err = my_mask * jitter ** 2
    return mod, new_err


def _model_scale(theta, time, rv, err2, flags, *modargs):
    ins_no = modargs[0]
    scale_coef = theta[0]

    #mod = np.zeros_like(time)
    my_mask = (flags == ins_no)
    mod = rv * (1 - scale_coef) * my_mask

    #new_err = my_mask * (scale_coef ** 2 - 1) * err2
    new_err = np.zeros_like(mod)
    return mod, new_err


def _model_keplerian(theta, time, rv, err2, flags, *modargs):
    per, A, phase, ecc, w = theta
    freq = 2. * np.pi / per
    M = freq * time + phase
    E = np.array([kepler.solve(m, ecc) for m in M])  # eccentric anomaly
    f = (np.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * np.tan(E / 2.)) * 2.)  # true anomaly
    model = A * (np.cos(f + w) + ecc * np.cos(w))

    ferr2 = np.zeros_like(model)
    return model, ferr2


def _model_keplerian_scale(theta, time, rv, err2, flags, *modargs):
    nins, kplan = modargs[0]
    masks = [flags == i for i in range(nins)]

    scales = theta[-nins:]

    model = np.zeros_like(rv)
    err2 = np.zeros_like(err2)

    # keplerians
    for k in range(kplan):
        model0, err20 = _model_keplerian(theta[k*5:(k+1)*5], time, rv, err2, flags, *modargs)
        model += model0

    # scales
    for i in range(nins):
        model[masks[i]] *= scales[i]

    # errors, may wanna multiply
    ferr2 = np.zeros_like(model)
    return model, ferr2


def _model_keplerian_hou(theta, time, rv, err2, flags, *modargs):
    P, As, Ac, S, C = theta

    per = np.exp(P)
    A = As ** 2 + Ac ** 2
    ecc = S ** 2 + C ** 2

    if ecc < 1e-5:
        w = 0
    else:
        w = np.arccos(C / (ecc ** 0.5))  # longitude of periastron
        if S < 0:
            w = 2 * np.pi - np.arccos(C / (ecc ** 0.5))

    phase = np.arccos(Ac / (A ** 0.5))
    if As < 0:
        phase = 2 * np.pi - np.arccos(Ac / (A ** 0.5))


    freq = 2. * np.pi / per
    M = freq * time + phase
    E = np.array([kepler.solve(m, ecc) for m in M])
    f = (np.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * np.tan(E / 2.)) * 2.)  # true anomaly

    model = A * (np.cos(f + w) + ecc * np.cos(w))

    ferr2 = np.zeros_like(model)

    return model, ferr2


def _model_keplerian_tp(theta, time, rv, err2, flags, *modargs):
    #model = 0
    per, A, tp, ecc, w = theta
    freq = 2. * np.pi / per

    M = freq * (time - tp)
    E = np.array([kepler.solve(m, ecc) for m in M])
    f = (np.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * np.tan(E / 2.)) * 2.)  # true anomaly
    model = A * (np.cos(f + w) + ecc * np.cos(w))

    ferr2 = np.zeros_like(model)
    return  model, ferr2


def _model_keplerian_houtp(theta, time, rv, err2, flags, *modargs):
    per, A, tp, S, C = theta

    ecc = S ** 2 + C ** 2

    if ecc < 1e-5:
        w = 0
    else:
        w = np.arccos(C / (ecc ** 0.5))  # longitude of periastron
        if S < 0:
            w = 2 * np.pi - np.arccos(C / (ecc ** 0.5))

    freq = 2. * np.pi / per
    M = freq * (time - tp)
    E = np.array([kepler.solve(m, ecc) for m in M])
    f = (np.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * np.tan(E / 2.)) * 2.)  # true anomaly
    model = A * (np.cos(f + w) + ecc * np.cos(w))

    ferr2 = np.zeros_like(model)
    return  model, ferr2


def _model_keplerian_pymc3(theta, time, rv, err2, flags, *modargs):
    import aesara_theano_fallback.tensor as tt
    import exoplanet as xo

    per, A, phase, ecc, w = theta
    freq = 2. * np.pi / per  # sometimes as n in literature
    M = freq * time + phase  # in exoplanet as n*t - (f + w)


    f = xo.orbits.get_true_anomaly(M, ecc + tt.zeros_like(M))
    model = A * (tt.cos(f + w) + ecc * tt.cos(w))

    ferr2 = tt.zeros_like(model)
    return  model, ferr2


def _model_keplerian2_pymc3(theta, time, rv, err2, flags, *modargs):
    import aesara_theano_fallback.tensor as tt
    import exoplanet as xo

    P, K, phi, C, S = theta


    pass


def _model_keplerian_joined(theta, time, rv, err2, flags, *modargs):
    pass


def _model_keplerian_transit(theta, time, rv, err2, flags, *modargs):
    import batman
    #params = batman.TransitParams()

    ldn = modargs[0][0]
    transit_bool = modargs[0][1]
    transit_model = modargs[0][2]
    bparams = modargs[0][3]
    if True:
        parnames = ['per', 'a', 't0', 'ecc', 'w', 'rp', 'inc']
        for i in range(7):
            setattr(bparams, parnames[i], theta[i])

        if ldn:
            bparams.limb_dark = ldn
            bparams.u = theta[7:]

    model = transit_model.light_curve(bparams) - transit_bool
    ferr2 = np.zeros_like(time)

    #print(model)
    #print(theta)
    #print('--------------------------------')
    return model, ferr2


def _model_gaussian_process(theta, time, rv, err2, flags, *modargs):
    import celerite
    from celerite import terms as cterms

    pass





#
