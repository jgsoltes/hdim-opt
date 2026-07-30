# hdim-opt: High-Dimensional Optimization Toolkit

Numerical optimization package for complex, high-dimensional problems. Includes the QUASAR evolutionary algorithm, Hyperellipsoid QMC sampling, and several streamlined functions derived from existing libraries for ease of use.

All core functions, listed below, are single-line executable and require three essential parameters: [obj_function, bounds, n_samples]:

### Optimization
* **quasar**: QUASAR optimization.
* **minimize**: Optimization using gradient-based minimization (via SciPy.minimize).
* **sensitivity**: Sensitivity analysis to quantify each variable's influence on the objective (via SALib).

### Sampling
* **hyperellipsoid**: Generate hyperellipsoidal sample sequence; may accelerate optimization.
* **uniform**: Generate uniform QMC sample sequences (via Scipy.stats.qmc).
* **isotropize/deisotropize**: Isotropize the input data using zero-phase component analysis (ZCA).
* **lorentzian**: Fit a Lorentzian/Cauchy kernel density estimation to the data.

### Analysis
* **analyze**: Numerically analyze any given dataset.
* **waveform**: Decompose the input waveform signal array into a diagnostic summary.


## Installation

Installed via `hdim_opt` directly from PyPI:

```bash
pip install hdim_opt
```

## Example Usage:

```python
import hdim_opt as h

### Parameter Space
n_dimensions = 10
n_samples = 2**10
bounds = [(-100,100)] * n_dimensions
obj_func = h.test_functions.rastrigin # Test function
t, signal = h.waveform_analysis.e1_waveform() # Signal

### Sampling
ellipsoid_samples = h.hyperellipsoid(n_samples, bounds, verbose=True) # Hyperellipsoid sampling
uniform_samples = h.uniform(n_samples, bounds, method='sobol') # Uniform sampling
iso_samples, iso_params = h.isotropize(ellipsoid_samples) # Isotropize data
kde = h.lorentzian(ellipsoid_samples, 150.0, ellipsoid_samples, verbose=True) # KDE

### Optimization
solution, fitness = h.quasar(obj_func, bounds, init=ellipsoid_samples) # evolutionary optimization
local_sol, local_fit = h.minimize(obj_func, bounds, init=solution) # evolutionary optimization
Si, S2 = h.sensitivity(obj_func, bounds) # sensitivity analysis

### Analysis
h.analyze(ellipsoid_samples) # Analyze any numerical dataset
summary = h.waveform(t, signal) # Waveform analysis
```

## QUASAR Optimizer
**QUASAR** (Quasi-Adaptive Search with Asymptotic Reinitialization) is a quantum-inspired evolutionary algorithm, highly efficient for minimizing high-dimensional, non-differentiable, and non-parametric objective functions.

* Benefit: Significant improvements in convergence speed and solution quality compared to contemporary optimizers. (Reference: [https://arxiv.org/abs/2511.13843]).

## HDS Sampler
**HDS** (Hyperellipsoid Density Sampling) is a non-uniform Quasi-Monte Carlo sampling method, specifically designed to exploit promising regions of the parameter space.

* Benefit: Provides control over high-dimensional sample distributions. Results in higher average solution quality when initializing optimization. (Reference: [https://arxiv.org/abs/2511.07836]).