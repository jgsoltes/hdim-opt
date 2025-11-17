## High-Dimensional Optimization Suite
Optimization suite for complex, high-dimensional problems.

**QUASAR** Optimizer (Quasi-Adaptive Search with Asymptotic Reinitialization):

* Quantum-inspired evolutionary algorithm, efficient for minimizing high-dimensional, non-differentiable, and non-parametric functions.
* Statistically significant improvements over contemporary optimizers.
* Called via hdim_opt.quasar()

**HDS** Sampler (Hyperellipsoid Density Sampling):
* Non-uniform Quasi-Monte Carlo sampling algorithm to exploit promising regions.
* Statistically significant improvements over uniform initialization samplers [https://arxiv.org/abs/2511.07836].
* Called via hdim_opt.hds()
