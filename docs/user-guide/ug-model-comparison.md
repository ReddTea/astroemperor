# Model Comparison and Next Run

## Model Matching

Multiple methods of model comparison are supported, which are calculated based on the statistics of the current chain compared to the previous one. After choosing a statistic and a tolerance (the default is BIC and tolerance 5), if the run complies with the required tolerance, the algorithm adds more complexity to the model and initiates another run.

Model comparison is done in `emperor` via the `emp_stats` object, and works with any of the calculated statistics.

The available methods are:


| Statistic           | name            | Comparison Criteria |
|---------------------|-----------------|---------------------|
| \\(\chi^2\\)        | ```chi2```      | ```Min```           |
| \\(\chi^2_{\nu}\\)  | ```chi2_red```  | ```Min```           |
| AIC                 | ```AIC```       | ```Min```           |
| BIC                 | ```BIC```       | ```Min```           |
| DIC                 | ```DIC```       | ```Min```           |        
| HQIC                | ```HQIC```      | ```Min```           |        
| RMSE                | ```RMSE```      | ```Min```           |        
| Maximum Posterior   | ```post_max```  | ```Max```           |
| Maximum Likelihood  | ```like_max```  | ```Max```           |
| Evidence            | ```Evidence```  | ```Max```           |
| Pass                | ```Pass```      | ```Pass```          |


The comparison statistic can be set with the `set_comparison_criteria` method, and the tolerance changed with the `set_tolerance` method.

The comparison criteria `min` is:
\\[\mathrm{BIC_{old}} - \mathrm{BIC_{new}} > \mathrm{tol}\\]

On the other hand, `max` corresponds to:
\\[\mathrm{BIC_{new}} - \mathrm{BIC_{old}} > \mathrm{tol}\\]

While `pass` always continues.


For example, if we want to find Keplerian signals up to 5. We set the Evidence as comparison criteria, with a tolerance of 10. The \\(K_0\\) had a log Z value of -1338. Since \\(-1338 - -\inf > 10\\) passes.
The \\(K_1\\) run had an Evidence value of -901, the comparison \\(-901 - -1338 > 10\\) passes again. The \\(K_2\\) run had an Evidence value of -948, the comparison fails \\(-948 - -901 \not> 10\\), stopping the program. 

```python
sim.set_comparison_criteria('Evidence')
sim.set_tolerance(10)

sim.autorun(0, 5)
```

## Subsequent Steps

If there is a following run, there are three things that `EMPEROR` will consider. First, adding a block to the model. Second, selecting the priors for the new parameters. And third, selecting the priors for the pre-existent parameters.

The added block will usually be a Keplerian Block. The priors of the new parameters, can be chosen like in the [previous section](ug-the-model.md#specs-and-priors):

```python
sim.add_condition(['Amplitude 1', 'limits', [40, 60]])  # ~Uniform[40, 60]

sim.add_condition(['Amplitude 2', 'limits', [20, 30]])  # ~Uniform[40, 60]

sim.add_condition(['Amplitude 3', 'limits', [1, 10]])  # ~Uniform[40, 60]
```

Each condition will be applied when that parameter comes to the table.


The subsequent run can optionally employ the previous posterior as its prior. Notably, the exploration space won't be sliced around the solution; rather a new prior will be placed upon it. \emp\ includes two approaches for modelling the posterior of each parameter-- Kernel Density Estimation and Bayesian Gaussian Mixtures. Both methods are employed through custom objects built upon the `sklearn` library, accepting any native variables as an input dictionary.

Options to constrain subsequent runs based on parameter uncertainties are also available. The available methods include constraining the next run's parameters around:

1. \\(n\sigma\\) of each chain's standard deviation.
2. [\\(\mu - n\sigma\\), \\(\mu + n\sigma\\)]-th percentiles of the chain, matching the corresponding \\(n\sigma\\) Gaussian range.
3. High-Density Interval (HDI), matching the \\(n\sigma\\) Gaussian range (e.g. \\(2\sigma\\) will constrain around the 95% HDI). By default, `emperor` applies a 99% HDI interval.

These methods are accesible through the `sigma`, `percentile`, and `range` keywords. Alternatively, using previous posteriors can be accessed with the `GM` or `KDE` keywords. The \\(n\sigma\\) value can be accessed through `constrain_sigma`, and to which blocks this will be applied through `constrain_types` (the default is just for the Keplerian blocks and Jitter's upper boundaries).


```python
sim.constrain_sigma = 3
sim.constrain_method = 'range'  # 'sigma', 'GM', 'range'
```