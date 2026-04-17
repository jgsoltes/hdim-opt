# hdim-opt: High-Dimensional Optimization Toolkit

Modern optimization package to accelerate convergence in complex, high-dimensional problems. Includes the QUASAR evolutionary algorithm, HDS exploitative QMC sampler, Sobol sensitivity analysis, signal waveform decomposition, and data transformations.

All core functions, listed below, are single-line executable and require three essential parameters: [obj_function, bounds, n_samples]:

* **quasar**: QUASAR optimization for high-dimensional problems.
* **hyperellipsoid**: Generate a non-uniform hyperellipsoid density sequence.
* **sensitivity**: Sensitivity analysis to quantify each variable's influence on the objective (via SALib).

* **lorentzian**: Fit a Lorentzian/Cauchy kernel density estimation to the data ensemble.
* **isotropize/deisotropize**: Isotropize the input data using zero-phase component analysis (ZCA).
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
n_samples = 1000
bounds = [(-100,100)] * n_dimensions
obj_func = h.test_functions.rastrigin # Test function

# Sampling
ellipsoid_samples = h.hyperellipsoid(n_samples, bounds, verbose=True) # Hyperellipsoid sampling
iso_samples, iso_params = h.isotropize(ellipsoid_samples) # Isotropize data
h.analyze(ellipsoid_samples) # Analyze any dataset

# Optimization
solution, fitness = h.quasar(obj_func, bounds, init=iso_samples) # QUASAR evolutionary optimization
Si, S2 = h.sensitivity(obj_func, bounds) # Sobol sensitivity analysis
kde = h.lorentzian(solution, sigma=150.0, ensemble=ellipsoid_samples, verbose=True) # Lorentzian KDE

# Waveforms
t, signal = h.waveform_analysis.e1_waveform(noise=0.1) # Waveform generation
summary = h.waveform(t,signal) # Waveform analysis
```

## QUASAR Optimizer
**QUASAR** (Quasi-Adaptive Search with Asymptotic Reinitialization) is a quantum-inspired evolutionary algorithm, highly efficient for minimizing high-dimensional, non-differentiable, and non-parametric objective functions.

* Benefit: Significant improvements in convergence speed and solution quality compared to contemporary optimizers. (Reference: [https://arxiv.org/abs/2511.13843]).

## HDS Sampler
**HDS** (Hyperellipsoid Density Sampling) is a non-uniform Quasi-Monte Carlo sampling method, specifically designed to exploit promising regions of the search space.

* Benefit: Provides control over the sample distribution. Results in higher average optimization solution quality when used for population initialization compared to uniform QMC methods. (Reference: [https://arxiv.org/abs/2511.07836]).