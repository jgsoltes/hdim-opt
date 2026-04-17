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
	>>> n_samples = 1000
	>>> bounds = [(-100,100)] * n_dimensions
	>>> obj_func = h.test_functions.rastrigin # Test function

	# Sampling
	>>> ellipsoid_samples = h.hyperellipsoid(n_samples, bounds, verbose=True) # Hyperellipsoid sampling
	>>> iso_samples, iso_params = h.isotropize(ellipsoid_samples) # Isotropize data
	>>> h.analyze(ellipsoid_samples) # Analyze any dataset

	# Optimization
	>>> solution, fitness = h.quasar(obj_func, bounds, init=iso_samples) # QUASAR evolutionary optimization
	>>> Si, S2 = h.sensitivity(obj_func, bounds) # Sobol sensitivity analysis
	>>> kde = h.lorentzian(solution, sigma=150.0, ensemble=ellipsoid_samples) # Lorentzian KDE

	# Waveforms
	>>> t, signal = h.waveform_analysis.e1_waveform(noise=0.1) # Waveform generation
	>>> summary = h.waveform(t,signal) # Waveform analysis
"""

# package version
__version__ = "1.4.1"

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

# sensitivity analysis, lorentzian KDE, data analysis, isotropization
try:
    from .sens_analysis import sensitivity, lorentzian, analyze, isotropize, deisotropize, pareto
except ImportError:
    sensitivity = lorentzian = analyze = isotropize = deisotropize = pareto = None

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
    'quasar', 'hyperellipsoid', 'sensitivity', 'analyze'
    'lorentzian', 'isotropize', 'deisotropize', 'waveform', 'sobol',
    'test_functions', 'quasar_helpers', 'waveform_analysis', 'pareto'
]