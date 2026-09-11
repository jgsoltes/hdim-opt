"""
# hdim-opt: High-Dimensional Optimization Toolkit

Functions:
	Sampling:
	- uniform: Generate uniform QMC sample sequences (via Scipy.stats.qmc).
	- hyperellipsoid: Generate hyperellipsoidal sample sequence; may accelerate optimization.
	- isotropize: Isotropize the input data via zero-phase component analysis (ZCA).
	- encode_bipolar: Bipolar-logarithmic transform, when negative values and exponents are present.

	Optimization:
	- quasar: Optimization using the QUASAR evolutionary algorithm.
	- minimize: Optimization using gradient-based minimization (via SciPy.minimize).
	- symbolic: Symbolic regression to approximate the input data or function (via gplearn).
	- stepAIC: Stepwise feature selection for linear or logistic regression (R's MASS:stepAIC).

	Analysis:
	- sensitivity: Sensitivity analysis to quantify each dimension's influence (via SALib).
	- hyperslice: Generate a hyperslice of a function's underlying solution space.
	- waveform: Decompose a 2-D input signal.
	- analyze: Numerically analyze any given dataset.

	Modules:
	- waveform_analysis: Contains analysis functions for signal decomposition.
	- test_functions: Contains test functions for local optimization testing.

Example Usage:
	### Import
	>>> import hdim_opt as h

	### Parameter Space
	>>> n_samples = 2**10
	>>> n_dimensions = 10
	>>> bounds = [(-100,100)] * n_dimensions # Parameter bounds
	>>> obj_func = h.test_functions.rastrigin # Test function

	### Sampling
	>>> uniform_samples = h.uniform(n_samples, bounds, method='sobol') # Uniform sampling
	>>> ellipsoid_samples = h.hyperellipsoid(n_samples, bounds, verbose=True) # Hyperellipsoid sampling
	>>> iso_samples, iso_params = h.isotropize(ellipsoid_samples) # Isotropize data (ZCA)
	>>> bipolar_log_samples = h.encode_bipolar(iso_samples, [b[0] for b in bounds]) # Bipolar-logarithm transform

	### Optimization
	>>> solution, fitness = h.quasar(obj_func, bounds, init=ellipsoid_samples) # Evolutionary optimization
	>>> local_sol, local_fit = h.minimize(obj_func, bounds, init=solution) # Gradient-based optimization
	>>> all_expr, best_expr = h.symbolic(obj_func, bounds) # Symbolic regression
	>>> opt_features, opt_model = h.stepAIC(iso_samples, solution) # R's stepAIC for linear/logistic regression

	### Analysis
	>>> Si, S2 = h.sensitivity(obj_func, bounds) # Sensitivity analysis
	>>> slice_data, stats = h.hyperslice(obj_func, bounds, slice_dims=(1,2)) # Estimate/hyperslice the solution space
	>>> signal_results = h.waveform(uniform_samples[:,0], slice_data.iloc[:,1]) # Analyze 2D waveform
	>>> h.analyze(slice_data) # Analyze any numerical dataset
"""

# package version
__version__ = "1.6.4"

# import core components
from .quasar_optimization import optimize as quasar
from .quasar_helpers import minimize
from . import quasar_helpers

### optional imports
# hyperellipsoid and uniform QMC sequences
try:
    from .hyperellipsoid_sampling import sample as hyperellipsoid
    from .hyperellipsoid_sampling import uniform as uniform
except ImportError as e:
    hyperellipsoid = uniform = None
    print(e)

# sensitivity analysis, lorentzian KDE, data analysis, isotropization
try:
    from .sens_analysis import (sensitivity, lorentzian, analyze, hyperslice, symbolic, stepAIC, 
        isotropize, deisotropize, encode_bipolar, decode_bipolar)
except ImportError as e:
    sensitivity = lorentzian = analyze = hyperslice = symbolic = stepAIC = None
    isotropize = deisotropize = encode_bipolar = decode_bipolar = None
    print(e)

# waveform analysis
try:
    from .waveform_analysis import analyze_waveform as waveform
    from . import waveform_analysis
except ImportError as e:
    waveform = waveform_analysis = None
    print(e)

# test functions
try:
    from . import test_functions
except ImportError as e:
    test_functions = None
    print(e)

# define full list
__all__ = [
    'quasar', 'hyperellipsoid', 'sensitivity', 'analyze',
    'lorentzian', 'uniform', 'waveform', 'hyperslice', 'symbolic', 'stepAIC',
    'isotropize', 'deisotropize', 'encode_bipolar', 'decode_bipolar',
    'test_functions', 'quasar_helpers', 'waveform_analysis'
]