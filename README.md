# hdim-opt: High-Dimensional Optimization Toolkit

`hdim_opt` is a lightweight yet comprehensive Python toolkit to streamline numerical optimization, sampling, and analysis of high-dimensional function landscapes and datasets. It features several methods for sample initialization, derivative & derivative-free optimization, and statistical analyses & visualizations of the solution landscape. 

`hdim_opt` provides single-line execution to streamline and improve upon common tasks that are inefficient to implement manually; particularly effective for black-box, high-dimensional, non-differentiable, non-parametric, multi-modal, multi-objective, and non-convex objective functions.

All core functions, listed below, are single-line executable and depend on three essential parameters: [obj_function, bounds, n_samples]:

### Sampling
* **uniform**: Generate uniform QMC sample sequences (via Scipy.stats.qmc).
* **hyperellipsoid**: Generate hyperellipsoidal sample sequence; may accelerate optimization.
* **isotropize**: Isotropize the input data via zero-phase component analysis (ZCA).
* **encode_bipolar**: Bipolar-logarithmic transform, when negative values and exponents are present.

### Optimization
* **quasar**: Optimization using the QUASAR evolutionary algorithm.
* **minimize**: Optimization using gradient-based minimization (via SciPy.minimize).
* **symbolic**: Symbolic regression to approximate the input data or function (via gplearn).
* **stepAIC**: Stepwise feature selection for linear or logistic regression (R's MASS:stepAIC).

### Analysis
* **sensitivity**: Sensitivity analysis to quantify each dimension's influence (via SALib).
* **hyperslice**: Create a hyperslice of the function's underlying solution space.
* **waveform**: Decompose any 2D waveform.
* **analyze**: Analyze any input dataset.


## Installation
Install `hdim_opt` from PyPI:

```bash
pip install hdim_opt
```

## Example Usage
```python
import hdim_opt as h

### Parameter Space
n_samples = 2**10
n_dimensions = 10
bounds = [(-100,100)] * n_dimensions # Parameter bounds
obj_func = h.test_functions.rastrigin # Test function

### Sampling
uniform_samples = h.uniform(n_samples, bounds, method='sobol') # Uniform sampling
ellipsoid_samples = h.hyperellipsoid(n_samples, bounds, verbose=True) # Hyperellipsoid sampling
iso_samples, iso_params = h.isotropize(ellipsoid_samples) # Isotropize data (ZCA)
bipolar_log_samples = h.encode_bipolar(iso_samples, [b[0] for b in bounds]) # Bipolar-logarithm transform

### Optimization
solution, fitness = h.quasar(obj_func, bounds, init=ellipsoid_samples) # Evolutionary optimization
local_sol, local_fit = h.minimize(obj_func, bounds, init=solution) # Gradient-based optimization
all_expr, best_expr = h.symbolic(obj_func, bounds) # Symbolic regression
opt_features, opt_model = h.stepAIC(iso_samples, solution) # R's stepAIC for feature selection & linear regression

### Analysis
Si, S2 = h.sensitivity(obj_func, bounds) # Sensitivity analysis
slice_data, stats = h.hyperslice(obj_func, bounds, slice_dims=(0,1), n_samples=2**7) # Estimate solution space
signal_results = h.waveform(uniform_samples[:,0], slice_data.iloc[:,1]) # Analyze 2D waveform
h.analyze(slice_data) # Analyze any numerical dataset
```

## QUASAR Optimization
Quasi-Adaptive Search with Asymptotic Reinitialization is an evolutionary algorithm highly efficient for minimizing high-dimensional, non-differentiable, and non-parametric objective functions. ([https://arxiv.org/abs/2511.13843]).

## Hyperellipsoid Sampling
Hyperellipsoid Density Sampling is a non-uniform Quasi-Monte Carlo sampling method, designed to exploit promising regions of the parameter space ([https://arxiv.org/abs/2511.07836]).

## Issues & Support
Bug reports are highly appreciated and will be handled timely.