# hdim-opt: High-Dimensional Optimization Toolkit

A modern optimization suite for complex, high-dimensional problems. This package provides algorithms to accelerate convergence, including the QUASAR evolutionary algorithm and HDS non-uniform QMC sampler.

All core functions, listed below, are single-line executable and require only three essential parameters: [obj_function, bounds, n_samples].
* **quasar**: QUASAR optimization.
* **hds**: Generate a non-uniform HDS sequence, for a dense sample distribution.
* **sobol**: Generate a highly uniform sample sequence (via SciPy).
* **sensitivity**: Perform Sobol sensitivity analysis to measure each variable's importance on objective function results (via SALib).


---

## Installation

Installed via `hdim_opt` directly from PyPI:

```bash
pip install hdim_opt
```

Quick Use Example:

```python
import hdim_opt as h

# Parameter Space
n_dimensions = 100
bounds = [(-100,100)] * n_dimensions
n_samples = 1000
obj_func = h.test_functions.rastrigin

# hdim_opt Function Usage
solution, fitness = h.quasar(obj_func, bounds)
sens_matrix = h.sensitivity(obj_func, bounds)
hds_samples = h.hds(n_samples, bounds)
sobol_samples = h.sobol(n_samples, bounds)
```

#### QUASAR Optimizer (Quasi-Adaptive Search with Asymptotic Reinitialization)
**QUASAR** is a quantum-inspired evolutionary algorithm, efficient for minimizing complex high-dimensional, non-differentiable, and non-parametric objective functions.

* Benefit: Statistically significant improvements in convergence speed and solution quality compared to contemporary optimizers.

* Reference: See experimental trials and analysis: [https://arxiv.org/abs/2511.13843].


### HDS Sampler (Hyperellipsoid Density Sampling)
**HDS** is a non-uniform Quasi-Monte Carlo sampling method, specifically designed to exploit promising regions of the search space.

* Benefit: Provides control over the sample distribution. Results in higher average optimization solution quality when used for population initialization compared to uniform QMC methods. 

* Reference: See experimental trials and analysis: [https://arxiv.org/abs/2511.07836].
