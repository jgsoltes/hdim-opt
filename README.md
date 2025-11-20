# hdim-opt: High-Dimensional Optimization Toolkit

A modern optimization suite for complex, high-dimensional problems. This package provides algorithms to accelerate convergence, including the QUASAR evolutionary algorithm and HDS non-uniform QMC sampler.

---

## Installation

Installed via `hdim-opt` directly from PyPI:

```bash
pip install hdim-opt
```

#### QUASAR Optimizer (Quasi-Adaptive Search with Asymptotic Reinitialization)
**QUASAR** is a quantum-inspired evolutionary algorithm, efficient for minimizing complex high-dimensional, non-differentiable, and non-parametric objective functions.

* Benefit: Statistically significant improvements in convergence speed and solution quality compared to contemporary optimizers.

* Reference: See experimental trials and analysis: [https://arxiv.org/abs/2511.13843].

Quick Use Example:

```python
import hdim_opt
import numpy as np

# define search space
n_dim = 100
bounds = [(-100,100)] * n_dim

def obj_func(x):
    y = np.sum(x**2)
    return y

# run QUASAR
solution, fitness = hdim_opt.quasar(func=obj_func, bounds=bounds)
```

### HDS Sampler (Hyperellipsoid Density Sampling)
**HDS** is a non-uniform Quasi-Monte Carlo sampling method, specifically designed to exploit promising regions of the search space.

* Benefit: Provides control over the sample distribution. Results in higher average optimization solution quality when used for population initialization compared to uniform QMC methods. 

* Reference: See experimental trials and analysis: [https://arxiv.org/abs/2511.07836].

Quick Use Example:
```python
import hdim_opt

# define search space
n_dim = 2
bounds = [(0,1)] * n_dim

# optional weights
weights = {
            0 : {'center': 0.25, 'std': 0.33},
            1 : {'center': 0.25, 'std': 0.33}
            }

# generate HDS samples
hds_samples = hdim_opt.hds(n_samples=10000, bounds=bounds, 
                    weights=weights, verbose=True)
```

Additional functions include: 
* sobol() to generate uniform Sobol samples (via SciPy)
* sensitivity() to perform Sobol sensitivity analysis (via SALib) (work in progress)