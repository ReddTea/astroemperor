# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/ /^\s*elif/
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# DEPENDENCIES
if True:
    import george
    from george.modeling import Model
    import numpy as np
    import matplotlib.pyplot as pl
    from george import kernels
    np.random.seed(1234)
    import emcee
    from tqdm import tqdm

# inputs
PLOT = False

RUN_1 = False
burn1 = 300
chain1 = 600

RUN_2 = True
burn2 = 400
chain2 = 1000

class Model(Model):
    parameter_names = ("amp", "location", "log_sigma2")

    def get_value(self, t):
        return self.amp * np.exp(-0.5*(t.flatten()-self.location)**2 * np.exp(-self.log_sigma2))


# simulate data

def generate_data(params, N, rng=(-5, 5)):
    gp = george.GP(0.1 * kernels.ExpSquaredKernel(3.3))
    t = rng[0] + np.diff(rng) * np.sort(np.random.rand(N))
    y = gp.sample(t)
    y += Model(**params).get_value(t)
    yerr = 0.05 + 0.05 * np.random.rand(N)
    y += yerr * np.random.randn(N)
    return t, y, yerr

truth = dict(amp=-1.0, location=0.1, log_sigma2=np.log(0.4))
t, y, yerr = generate_data(truth, 50)

if PLOT:
    pl.figure()
    pl.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
    pl.ylabel(r"$y$")
    pl.xlabel(r"$t$")
    pl.xlim(-5, 5)
    pl.title("simulated data");
    pl.show()

class PolynomialModel(Model):
    parameter_names = ("m", "b", "amp", "location", "log_sigma2")
    def get_value(self, t):
        t = t.flatten()
        return (t * self.m + self.b +
                self.amp * np.exp(-0.5*(t-self.location)**2*np.exp(-self.log_sigma2)))
    pass


def lnprob(p):
    model.set_parameter_vector(p)
    return model.log_likelihood(y, quiet=True) + model.log_prior()

if RUN_1:
    print('initialize run1')
    model = george.GP(mean=PolynomialModel(m=0, b=0, amp=-1, location=0.1, log_sigma2=np.log(0.4)))
    model.compute(t, yerr)


    initial = model.get_parameter_vector()
    ndim, nwalkers = len(initial), 32
    p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, burn1)
    sampler.reset()

    print("Running production...")
    sampler.run_mcmc(p0, chain1);

    # plot results
    if PLOT:
        # Plot the data.
        pl.figure()
        pl.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)

        # The positions where the prediction should be computed.
        x = np.linspace(-5, 5, 500)

        # Plot 24 posterior samples.
        samples = sampler.flatchain
        for s in samples[np.random.randint(len(samples), size=24)]:
            model.set_parameter_vector(s)
            pl.plot(x, model.mean.get_value(x), color="#4682b4", alpha=0.3)

        pl.ylabel(r"$y$")
        pl.xlabel(r"$t$")
        pl.xlim(-5, 5)
        pl.title("fit assuming uncorrelated noise");
        pl.show()


##
def lnprob2(p):
    gp.set_parameter_vector(p)
    return gp.log_likelihood(y, quiet=True) + gp.log_prior()
if RUN_2:
    print('init run2')
    kwargs = dict(**truth)
    kwargs["bounds"] = dict(location=(-2, 2))
    mean_model = Model(**kwargs)
    gp = george.GP(np.var(y) * kernels.Matern32Kernel(10.0), mean=mean_model)
    gp.compute(t, yerr)


    initial = gp.get_parameter_vector()
    ndim, nwalkers = len(initial), 32
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2)


    print("Running first burn-in...")
    p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
    p0, lp, _ = sampler.run_mcmc(p0, chain2)

    print("Running second burn-in...")
    p0 = p0[np.argmax(lp)] + 1e-8 * np.random.randn(nwalkers, ndim)
    sampler.reset()
    p0, _, _ = sampler.run_mcmc(p0, chain2)
    sampler.reset()

    print("Running production...")
    sampler.run_mcmc(p0, chain2);



from corner import corner

tri_cols = ["amp", "location", "log_sigma2"]
tri_labels = [r"$\alpha$", r"$\ell$", r"$\ln\sigma^2$"]
tri_truths = [truth[k] for k in tri_cols]


names = gp.get_parameter_names()
inds = np.array([names.index("mean:"+k) for k in tri_cols])
corner(sampler.flatchain[:, inds], truths=tri_truths, labels=tri_labels);
pl.show()

gp.get_parameter_vector()








#
