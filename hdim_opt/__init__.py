"""
# hdim-opt: High-Dimensional Optimization Toolkit

Functions:
	- quasar: QUASAR optimization for high-dimensional problems.
	- hyperellipsoid: Generate a non-uniform hyperellipsoid density sequence.
	- sensitivity: Sensitivity analysis to quantify each variable's influence on the objective (via SALib).
	- minimize: Local optimization wrapper for scipy.optimize.
	
	- analyze: Generate a statistical summary of the input dataset.
	- lorentzian: Fit a Lorentzian/Cauchy kernel density estimation to the data ensemble.
	- isotropize/deisotropize: Isotropize the input data using ZCA.
	- waveform: Decompose the input waveform signal array into a diagnostic summary.

Modules:
	- test_functions: Contains test functions for local optimization testing.
	- waveform_analysis: Contains pulse signal decompositions.

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
	>>> uniform_samples = h.uniform(n_samples, bounds, method='sobol') # Uniform sampling
	>>> iso_samples, iso_params = h.isotropize(ellipsoid_samples) # Isotropize data
	>>> h.analyze(ellipsoid_samples) # Analyze any dataset

	# Optimization
	>>> solution, fitness = h.quasar(obj_func, bounds, init=ellipsoid_samples) # Evolutionary optimization
	>>> local_sol, local_fit = h.minimize(obj_func, bounds, init=solution) # local minimization
	>>> Si, S2 = h.sensitivity(obj_func, bounds) # Sensitivity analysis
	>>> kde = h.lorentzian(solution, sigma=150.0, ensemble=ellipsoid_samples) # Lorentzian KDE

	# Waveforms
	>>> t, signal = h.waveform_analysis.e1_waveform(noise=0.1) # Waveform generation
	>>> summary = h.waveform(t,signal) # Waveform analysis
"""

# package version
__version__ = "1.4.72"

# import core components
from .quasar_optimization import optimize as quasar
from .quasar_helpers import minimize
from . import quasar_helpers

### optional imports
# hyperellipsoid and sobol sequences
try:
    from .hyperellipsoid_sampling import sample as hyperellipsoid
    from .hyperellipsoid_sampling import uniform as uniform
except ImportError:
    hyperellipsoid = sobol = None

# sensitivity analysis, lorentzian KDE, data analysis, isotropization
try:
    from .sens_analysis import sensitivity, lorentzian, analyze, isotropize, deisotropize
except ImportError:
    sensitivity = lorentzian = analyze = isotropize = deisotropize = None

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
    'test_functions', 'quasar_helpers', 'waveform_analysis'
]