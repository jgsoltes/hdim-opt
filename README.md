# hdim-opt: High-Dimensional Optimization Toolkit

Numerical optimization package for complex, high-dimensional problems. Includes the QUASAR evolutionary algorithm, Hyperellipsoid QMC sampling, and several useful functions streamlined from existing libraries.

All core functions, listed below, are single-line executable and depend on three essential parameters: [obj_function, bounds, n_samples]:

### Sampling
* **hyperellipsoid**: Generate hyperellipsoidal sample sequence; may accelerate optimization.
* **uniform**: Generate uniform QMC sample sequences (via Scipy.stats.qmc).
* **isotropize**: Isotropize the input data via zero-phase component analysis (ZCA).
* **lorentzian**: Fit a Lorentzian/Cauchy kernel density estimation to the data.

### Optimization
* **quasar**: Optimization using the QUASAR evolutionary algorithm.
* **minimize**: Optimization using gradient-based minimization (via SciPy.minimize).
* **symbolic**: Symbolic regression to approximate the input data or function (via gplearn).

### Analysis
* **sensitivity**: Sensitivity analysis to quantify each dimension's influence on the objective (via SALib).
* **hyperslice**: Generate a 1D or 2D hyperslice of a function's underlying solution space.
* **waveform**: Decompose the input signal waveform.
* **analyze**: Numerically analyze any given dataset.



## Installation

Install `hdim_opt` directly from PyPI:

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
ellipsoid_samples = h.hyperellipsoid(n_samples, bounds, verbose=True) # Hyperellipsoid sampling
uniform_samples = h.uniform(n_samples, bounds, method='sobol') # Uniform sampling
iso_samples, iso_params = h.isotropize(ellipsoid_samples) # Isotropize data
kde = h.lorentzian(iso_samples, 150.0, iso_samples, verbose=True) # Lorentzian multivariate KDE

### Optimization
solution, fitness = h.quasar(obj_func, bounds, init=ellipsoid_samples) # Evolutionary optimization
local_sol, local_fit = h.minimize(obj_func, bounds, init=solution) # Gradient-based optimization
all_expr, best_expr = h.symbolic(obj_func, bounds) # Symbolic regression

### Analysis
Si, S2 = h.sensitivity(obj_func, bounds) # Sensitivity analysis
slice_data, stats = h.hyperslice(obj_func, bounds, slice_dims=()) # Hyperslice of the solution space
h.analyze(slice_data) # Analyze any numerical dataset
```

## QUASAR Optimizer
**QUASAR** (Quasi-Adaptive Search with Asymptotic Reinitialization) is a quantum-inspired evolutionary algorithm, highly efficient for minimizing high-dimensional, non-differentiable, and non-parametric objective functions.

* Benefit: Significant improvements in convergence speed and solution quality for high-dimensional spaces compared to standard optimization algorithms like Differential Evolution and L-SHADE. (Reference: [https://arxiv.org/abs/2511.13843]).

## HDS Sampler
**HDS** (Hyperellipsoid Density Sampling) is a non-uniform Quasi-Monte Carlo sampling method, specifically designed to exploit promising regions of the parameter space.

* Benefit: Provides control over high-dimensional sample distributions. Results in higher average solution quality when initializing optimization. (Reference: [https://arxiv.org/abs/2511.07836]).