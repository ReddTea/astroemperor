import george
george.__version__

import numpy as np
import matplotlib.pyplot as pl
from statsmodels.datasets import co2
import pandas as pd


df = pd.DataFrame.from_records(co2.load().data)
df['date'] = df.date.apply(lambda x: x.decode('utf-8'))
df['date'] = pd.to_datetime(df.date.values, format='%Y%m%d').to_julian_date()
#df['date'] = pd.to_datetime(df['date'].values).to_julian_date()

t = 2000 + (df.date.values.astype(float) - 2451545.0) / 365.25
y = np.array(df.co2.values.astype(float))
m = np.isfinite(t) & np.isfinite(y) & (t < 1996)
t, y = t[m][::4], y[m][::4]

pl.plot(t, y, ".k")
pl.xlim(t.min(), t.max())
pl.xlabel("year")
pl.ylabel("CO$_2$ in ppm");

pl.show()

# create kernels
from george import kernels

k1 = 66**2 * kernels.ExpSquaredKernel(metric=67**2)
k2 = 2.4**2 * kernels.ExpSquaredKernel(90**2) * kernels.ExpSine2Kernel(gamma=2/1.3**2, log_period=0.0)
k3 = 0.66**2 * kernels.RationalQuadraticKernel(log_alpha=np.log(0.78), metric=1.2**2)
k4 = 0.18**2 * kernels.ExpSquaredKernel(1.6**2)
kernel = k1 + k2 + k3 + k4

gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
               white_noise=np.log(0.19**2), fit_white_noise=True)
gp.compute(t)
print('LOGL1')
print(gp.log_likelihood(y))
#print(gp.grad_log_likelihood(y))

##################
'''
import scipy.optimize as op

# Define the objective function (negative log-likelihood in this case).
def nll(p):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)

# You need to compute the GP once before starting the optimization.
gp.compute(t)

# Print the initial ln-likelihood.
print(gp.log_likelihood(y))

# Run the optimization routine.
p0 = gp.get_parameter_vector()
results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

# Update the kernel and print the final log-likelihood.
gp.set_parameter_vector(results.x)
print(gp.log_likelihood(y))
##################

x = np.linspace(max(t), 2025, 2000)
mu, var = gp.predict(y, x, return_var=True)
std = np.sqrt(var)

pl.plot(t, y, ".k")
pl.fill_between(x, mu+std, mu-std, color="g", alpha=0.5)

pl.xlim(t.min(), 2025)
pl.xlabel("year")
pl.ylabel("CO$_2$ in ppm");
pl.show()
'''
# WITH EMCEE

def lnprob(p):
    # Trivial uniform prior.
    if np.any((-100 > p[1:]) + (p[1:] > 100)):
        return -np.inf

    # Update the kernel and compute the lnlikelihood.
    gp.set_parameter_vector(p)
    #print(p)
    print(gp.lnlikelihood(y, quiet=True))
    return gp.lnlikelihood(y, quiet=True)

import emcee

#gp.compute(t)

def pt_lnprob(p):
    gp.set_parameter_vector(p)
    print(gp.lnlikelihood(y, quiet=True))
    return gp.lnlikelihood(y, quiet=True)
def pt_lp(p):
    if np.any((-100 > p[1:]) + (p[1:] > 100)):
        return -np.inf
    else:
        return 0.0

# Set up the sampler.
nwalkers, ndim = 36, len(gp)
ntemps = 2

#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
sampler = emcee.PTSampler(ntemps, nwalkers, ndim, pt_lnprob, pt_lp)

# Initialize the walkers.
p0 = gp.get_parameter_vector() + 1e-4 * np.random.randn(nwalkers, ndim)
pt_p0 = np.array([p0 for _ in range(ntemps)])

print("Running burn-in")
#p0, _, _ = sampler.run_mcmc(p0, 200)
pt_p0, _, _ = sampler.run_mcmc(pt_p0, 100)

print("Running production chain")
#sampler.run_mcmc(p0, 200);
sampler.run_mcmc(pt_p0, 100);
'''
x = np.linspace(max(t), 2025, 250)
for i in range(50):
    # Choose a random walker and step.
    w = np.random.randint(sampler.chain.shape[0])
    n = np.random.randint(sampler.chain.shape[1])
    gp.set_parameter_vector(sampler.chain[w, n])

    # Plot a single sample.
    pl.plot(x, gp.sample_conditional(y, x), "g", alpha=0.1)

pl.plot(t, y, ".k")

pl.xlim(t.min(), 2025)
pl.xlabel("year")
pl.ylabel("CO$_2$ in ppm");
pl.show()
'''
#








'''



def transit_noise():
    log_like = get_sq_exp_likelihood(xt,residuals,yerrt*1e6,
                                     parameters['sigma_w']['object'].value,
                                     parameters['lnh']['object'].value,
                                     parameters['lnlambda']['object'].value)
    return log_like



def lnlike_transit(gamma=1.0):
    coeff1,coeff2 = reverse_ld_coeffs(options['photometry'][the_instrument]['LD_LAW'], \
    parameters['q1']['object'].value,parameters['q2']['object'].value)
    params[the_instrument].t0 = parameters['t0']['object'].value
    params[the_instrument].per = parameters['P']['object'].value
    params[the_instrument].rp = parameters['p']['object'].value
    params[the_instrument].a = parameters['a']['object'].value
    params[the_instrument].inc = parameters['inc']['object'].value
    params[the_instrument].ecc = parameters['ecc']['object'].value
    params[the_instrument].w = parameters['omega']['object'].value
    params[the_instrument].u = [coeff1,coeff2]
    model = m[the_instrument].light_curve(params[the_instrument])
    residuals = (yt-model)*1e6
    log_like = get_sq_exp_likelihood(xt,residuals,yerrt*1e6,
                                     parameters['sigma_w']['object'].value,
                                     parameters['lnh']['object'].value,
                                     parameters['lnlambda']['object'].value)


    return log_like

'''




#
