# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from celerite.modeling import Model

PLOT = False


# Define the model
class MeanModel(Model):
    parameter_names = ("alpha", "ell", "log_sigma2")

    def get_value(self, t):
        return self.alpha * np.exp(-0.5*(t-self.ell)**2 * np.exp(-self.log_sigma2))

    # This method is optional but it can be used to compute the gradient of the
    # cost function below.
    def compute_gradient(self, t):
        e = 0.5*(t-self.ell)**2 * np.exp(-self.log_sigma2)
        dalpha = np.exp(-e)
        dell = self.alpha * dalpha * (t-self.ell) * np.exp(-self.log_sigma2)
        dlog_s2 = self.alpha * dalpha * e
        return np.array([dalpha, dell, dlog_s2])

mean_model = MeanModel(alpha=-1.0, ell=0.1, log_sigma2=np.log(0.4))
true_params = mean_model.get_parameter_vector()

# Simuate the data
np.random.seed(42)
x = np.sort(np.random.uniform(-5, 5, 50))
yerr = np.random.uniform(0.05, 0.1, len(x))
K = 0.1*np.exp(-0.5*(x[:, None] - x[None, :])**2/10.5)
K[np.diag_indices(len(x))] += yerr**2
y = np.random.multivariate_normal(mean_model.get_value(x), K)

# Plot the data
if PLOT:
    plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
    plt.ylabel(r"$y$")
    plt.xlabel(r"$t$")
    plt.xlim(-5, 5)
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.title("simulated data");
    plt.show()



from scipy.optimize import minimize

import celerite
from celerite import terms

# Set up the GP model
kernel = terms.RealTerm(log_a=np.log(np.var(y)), log_c=-np.log(10.0))
gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
gp.compute(x, yerr)
print("Initial log-likelihood: {0}".format(gp.log_likelihood(y)))

# Define a cost function
def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

def grad_neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.grad_log_likelihood(y)[1]

# Fit for the maximum likelihood parameters
initial_params = gp.get_parameter_vector()
bounds = gp.get_parameter_bounds()
soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                method="L-BFGS-B", bounds=bounds, args=(y, gp))
gp.set_parameter_vector(soln.x)
print("Final log-likelihood: {0}".format(-soln.fun))

# Make the maximum likelihood prediction
t = np.linspace(-5, 5, 500)
mu, var = gp.predict(y, t, return_var=True)
std = np.sqrt(var)

# Plot the data
if PLOT:
    color = "#ff7f0e"
    plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
    plt.plot(t, mu, color=color)
    plt.fill_between(t, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
    plt.ylabel(r"$y$")
    plt.xlabel(r"$t$")
    plt.xlim(-5, 5)
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.title("maximum likelihood prediction");
    plt.show()




#
