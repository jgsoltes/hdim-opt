# hdim-opt: High-Dimensional Optimization Toolkit

Modern optimization package to accelerate convergence in complex, high-dimensional problems. Includes the QUASAR evolutionary algorithm, HDS exploitative QMC sampler, Sobol sensitivity analysis, signal waveform decomposition, and data transformations.

All core functions, listed below, are single-line executable and require three essential parameters: [obj_function, bounds, n_samples].
* **quasar**: QUASAR optimization for high-dimensional problems.
* **hyperellipsoid**: Generate a non-uniform hyperellipsoid density sequence.
* **sensitivity**: Sensitivity analysis to quantify each variable's influence on the objective (via SALib).
* **pareto**: Easily create a multi-objective Pareto front trade-off analysis using QUASAR.

* **lorentzian**: Fit a Lorentzian/Cauchy kernel density estimation to the data ensemble.
* **isotropize/deisotropize**: Isotropize the input data using ZCA.
* **waveform**: Decompose the input waveform signal array into a diagnostic summary.


---

## Installation

Installed via `hdim_opt` directly from PyPI:

```bash
pip install hdim_opt
```

## Example Usage:

```python
import hdim_opt as h

# Parameter Space
n_dimensions = 30
bounds = [(-100,100)] * n_dimensions
n_samples = 1000
obj_func = h.test_functions.rastrigin

# Optimization
solution, fitness = h.quasar(obj_func, bounds)
sens_matrix = h.sensitivity(obj_func, bounds)
pareto_front = h.pareto(obj_func, bounds, [])

# Sampling
hds_samples = h.hyperellipsoid(n_samples, bounds)
iso_samples, params = h.isotropize(hds_samples)
kde = h.lorentzian(solution, 3.0, iso_samples)
```

## QUASAR Optimizer
**QUASAR** (Quasi-Adaptive Search with Asymptotic Reinitialization) is a quantum-inspired evolutionary algorithm, highly efficient for minimizing high-dimensional, non-differentiable, and non-parametric objective functions.

* Benefit: Significant improvements in convergence speed and solution quality compared to contemporary optimizers. (Reference: [https://arxiv.org/abs/2511.13843]).

## HDS Sampler
**HDS** (Hyperellipsoid Density Sampling) is a non-uniform Quasi-Monte Carlo sampling method, specifically designed to exploit promising regions of the search space.

* Benefit: Provides control over the sample distribution. Results in higher average optimization solution quality when used for population initialization compared to uniform QMC methods. (Reference: [https://arxiv.org/abs/2511.07836]).