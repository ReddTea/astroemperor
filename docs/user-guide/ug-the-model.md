# Model Management

Modelling is done via `blocks`, which are objects containing a specific part of the overall model. Each block represents a single function, and contains its associated parameters (in `specs` objects), and all relevant metadata (e.g. model name, \LaTeX representation, cardinality, dimensionality, python dependencies, and script-writing routines). `Specs`, in turn, hold their own metadata for each parameter (e.g. parameter name, units, boundaries, and prior distribution). After a run is executed, its results are stored locally and accessed by the main script for post-processing.

There are two ways of adding `blocks` in `emperor`, manually, and through the automatic mode. We start by making a short summary of each available block for the automatic mode, and we show how to use them in the code.

## Blocks




### Keplerian Block

One parameterisation extensively used in the literature, corresponds to (P, K, \\(\phi\\), $e$, \\(\bar{\omega}\\)), representing the period, semi-amplitude, phase of pericentre passage, eccentricity and longitude of periastron, respectively \citep{exo_handbook}.

Alternatively, [Hou et al.](https://arxiv.org/abs/1104.2612) propose an equivalent set (P, \\(K_c\\), \\(K_s\\), \\(e_c\\), \\(e_s\\)), where 


\\[K_c = \sqrt{K}\cos(\phi), \,\, K_s = \sqrt{K}\sin(\phi) \\]

\\[e_c = \sqrt{e}\cos(\bar{\omega}), \,\,\,\,\,\, e_s = \sqrt{e}\sin(\bar{\omega})\\]

This re-parameterisation is used to bound periodically both \\(\bar{\omega}\\) and \\(\phi\\), improving the performance of the sampler while linearising these circular parameters.

In photometric studies, the time of inferior conjunction \\(T_0\\) is often preferred over the phase \\(\phi\\). Consequently, the set (P, K, \\(T_0\\), \\(e_c\\), \\(e_s\\)) is recommended for such applications.

Different parameterisations for the Keplerian, such as the one shown before, are supported in `emperor`:

| parameterisation                                      | value  |
|-------------------------------------------------------|--------|
| (P, K, \\(\phi\\), e, \\(\bar{\omega}\\))             | 0      |
| (P, K, \\(\phi\\), \\(e_{s}\\), \\(e_{c}\\))          | 1      |
| (logP, \\(K_{s}\\), \\(K_{c}\\), \\(e_{s}\\), \\(e_{c}\\))| 2      |
| (P, K, \\(T_0\\), e, \\(\bar{\omega}\\))              | 3      |
| (P, K, \\(T_0\\), \\(e_{s}\\), \\(e_{c}\\))           | 4      |
| (P, K, \\(M_0\\), e, \\(\bar{\omega}\\))              | 5      |


Adding Keplerians is done when calling the `autorun` method. Furthermore, `emperor` also estimates the minimum-mass and semi-major axis of each planet if the mass of the star is provided.


For example, if we want to use the (P, A, \\(T_0\\), \\(e_c\\), \\(e_s\\)) parameterisation, and evaluate the model with one, then two, and then three Keplerian signals, we simply do:


```python
sim.starmass = 1.12  # mass for 51 Peg
sim.keplerian_parameterisation = 4  # as seen in the table
sim.autorun(1, 3)  # from 1 Keplerian, to 3 Keplerians
```



### Sinusoid Block

This sinusoid can be used for modelling the magnetic cycle.
\\[S_{i}(t) = K_i \cos{(\omega_i t + \phi_i)}\\]

```python
sim.sinusoid = True  # adds a sinusoid
sim.autorun(0, 0)  # no Keplerians, it will evaluate just the sinusoid
```



### Offset Block
The acceleration function \\(\Gamma\\) models offsets and accelerations:
\\[\Gamma_{i, \mathrm{INS}}(t_i, a) = \sum{\frac{\partial^a \gamma}{\partial^a dt} \cdot t^a}\\]

For orders greater than zero, it is assumed that the linear trend is mainly caused by the observed system, and contributions per instrument are detrended in the image reduction. In this way, accelerations are shared among instruments, giving the more familiar notation \\[\Gamma_{i, \mathrm{INS}}(t_i, a=1) = \gamma_{\mathrm{INS}} + \dot{\gamma} \cdot t_i\\].

This is why the first term is handled by the Offset Block, while the rest by the Acceleration Block. The Offset Block will have one parameter per instrument.


### Acceleration Block
The acceleration can be set as:

```python
sim.acceleration = 0
```

A value of 0 will use only offsets. A value of 1, offsets plus a common linear acceleration. And so on.


### Jitter Block
Jitter represents additional white noise beyond the instrumental error \\(\sigma_{i}\\) , often linked to stellar activity.
Jitter increases the uncertainty in the hypothesis, providing a more conservative posterior estimate:
\\[p(D | \pmb{\theta}) = \prod_{\mathrm{INS}}\prod_i^N \sqrt{\frac{1}{2 \pi (\sigma_i^2 + \sigma_{\mathrm{INS}}^2)}} \exp{(- \frac{\xi_i^2}{2 (\sigma_i^2 + \sigma_{\mathrm{INS}}^2)})}\\]


### Moving Average Block
A low-pass filter to smooth the signal by removing correlated noise with the previous q residuals, where q is the order of the moving average:

\\[R_{i,\mathrm{INS}} = \sum_{q}\Phi_{\mathrm{INS},q} \exp(\frac{-|t_{i-q}-t_i|}{\tau_{\mathrm{INS},q}}) \xi_{i-q,\mathrm{INS}}\\]

With \\(\Phi_{\ins,q}\\) the Moving Average Coefficient, \\(\tau_{\ins,q}\\) the characteristic timescale (decay over time factor) and \\(\xi\\) the residuals. If \\(q=0\\), only white noise is modelled, and for \\(q \geq 1\\) higher orders of correlated noise can be captured.

The moving average customisation options are handled through a dictionary. To select if you want to use it per instrument or shared, rely on the `global` keyword, and for the order, `order`.

```python
sim.moav['global'] = False
sim.moav['order'] = 0
```


### Stellar Activity Block

This `Block` represents linear correlations with activity indices, which could be chromospheric stellar activity proxies (e.g. the S-index or log RHK) or line asymmetry measurements (e.g. full width at half maximum, bisector index slope). Indices need to be included in the input data, and they enter the model via

\\[ A\_{i,\mathrm{INS}} = \sum\_{\mathcal{A}} \mathcal{C}\_{\mathcal{A},\mathrm{INS}} \cdot \mathcal{A}\_{i,\mathrm{INS}} \\]

where \\(\mathcal{A}\_{i,\mathrm{INS}}\\) denotes each measured activity index, and \\(\mathcal{C}\_{\mathcal{A},\mathrm{INS}}\\) is the corresponding coefficient (the first-order Taylor expansion of the RV dependence on that index).

To use this model simply add the line:

```python
sim.switch_SA = True
```

### Magnetic Cycle Block
Used to model Magnetic Cycles:

\\[M(t) = K_1 \cos{(\omega_1 t + \phi_1)} + K_2 \cos{(2\omega_1 t + \phi_2)}\\]


To use, add the line:
```python
sim.magnetic_cycle = True
```

### Celerite Block

[In construction!]
The celerite mode must be turned on or off with `switch_celerite`.
The `celerite` kernel must be added through a dictionary. Terms are additive, and they go in as a list. Their respective parameterisation must be defined as in:

```python
sim.switch_celerite = True
sim.my_kernel = {'terms':['RotationTerm'],
                 'params':[{'period':None,
                            'sigma':None,
                            'Q0':None,
                            'dQ':None,
                            'f':None}]
                            }
```

Please refer to the excellent documentation of [celerite2](https://celerite2.readthedocs.io/en/latest/api/python/#model-building).


## Specs and Priors

`Specs` are `EMPEROR`'s representation of the parameters. They contain all the metadata corresponding to each parameter, such as name, units, boundaries, prior distribution, etc.

Modifying parameters is most easily done via the `add_condition` function.
For example, let's say we examined the periodogram of our RVs, and we want to set a Normal prior on the period, constrain the amplitude in a certain range, and fix the eccentricity to 0:


```python
sim.add_condition(['Period 1', 'prior', 'Normal'])  # Use the Normal function
sim.add_condition(['Period 1', 'prargs', [4.23, 1]])  # Use ~Normal(4.23, 1)

sim.add_condition(['Amplitude 1', 'limits', [40, 60]])  # ~Uniform[40, 60]

sim.add_condition(['Eccentricity 1', 'fixed', 0])
sim.add_condition(['Longitude 1', 'fixed', 0])
```

The `add_condition` function receives a list with three items. The parameter name as `str`, what to change as `str`, and the new value.

Properties that are worth keeping in mind for adjustments:


| property        | value       | Example                                   |
|-----------------|-------------|-------------------------------------------|
| ```limits```    | ```list```  | ```'Amplitude 1', 'limits', [40, 60]```   |
| ```prior```     | ```str```   | ```'Period 1', 'prior', 'Normal'```       |
| ```prargs```    | ```list```  | ```'Period 1', 'prargs', [4.23, 1]```     |
| ```init_pos```  | ```list```  | ```'Period S1', 'init_pos', [4.1, 4.3]``` |
| ```fixed```     | ```float``` | ```'Eccentricity 1', 'fixed', 0```        |


The out-of-the-box available priors are:
Uniform, Normal, Beta, Jeffreys, and GaussianMixture.

