"""

# hdim-opt: High-Dimensional Optimization Toolkit

Functions:
	- quasar: QUASAR optimization for high-dimensional problems.
	- hyperellipsoid: Generate a non-uniform hyperellipsoid density sequence.
	- sensitivity: Sensitivity analysis to quantify each variable's influence on the objective (via SALib).
	- pareto: Easily create a multi-objective Pareto front trade-off analysis using QUASAR.

	- lorentzian: Fit a Lorentzian/Cauchy kernel density estimation to the data ensemble.
	- isotropize/deisotropize: Isotropize the input data using ZCA.
	- waveform: Decompose the input waveform signal array into a diagnostic summary.

Modules:
	- test_functions: Contains test functions for local optimization testing.
	- waveform_analysis: Contains pulse signal generation functions.

Example Usage:

	# Import
	>>> import hdim_opt as h

	# Parameter Space
	>>> n_dimensions = 30
	>>> bounds = [(-100,100)] * n_dimensions
	>>> n_samples = 1000
	>>> obj_func = h.test_functions.rastrigin

	# Optimization
	>>> solution, fitness = h.quasar(obj_func, bounds)
	>>> Si, S2 = h.sensitivity(obj_func, bounds)

	# Sampling
	>>> hds_samples = h.hyperellipsoid(n_samples, bounds)
	>>> iso_samples, params = h.isotropize(hds_samples)
	>>> kde = h.lorentzian(solution, 3.0, hds_samples)

	# Waveform
	>>> t, signal = h.waveform_analysis.e1_waveform(noise=0.1)
	>>> summary = h.waveform(t,signal)
"""

# package version
__version__ = "1.3.7"

# import core components
from .quasar_optimization import optimize as quasar
from . import quasar_helpers

### optional imports
# hyperellipsoid and sobol sequences
try:
    from .hyperellipsoid_sampling import sample as hyperellipsoid
    from .hyperellipsoid_sampling import sobol_sample as sobol
except ImportError:
    hyperellipsoid = sobol = None

# sensitivity analysis, lorentzian KDE, pareto front, isotropization
try:
    from .sens_analysis import sensitivity, lorentzian, pareto, isotropize, deisotropize
except ImportError:
    sensitivity = lorentzian = pareto = isotropize = deisotropize = None

# waveform analysis
try:
    from .waveform_analysis import analyze_waveform as waveform
    from . import waveform_analysis
except ImportError:
    waveform = waveform_analysis = None

# test functions
try:
    from . import test_functions
except ImportError:
    test_functions = None

# define full list
__all__ = [
    'quasar', 'hyperellipsoid', 'sensitivity', 'pareto',
    'lorentzian', 'isotropize', 'deisotropize', 'waveform', 'sobol',
    'test_functions', 'quasar_helpers', 'waveform_analysis'
]