"""

# hdim-opt: High-Dimensional Optimization Toolkit

Functions:
	- quasar: QUASAR optimization for high-dimensional, non-differentiable problems.
	- hds: Generate an exploitative HDS sequence, to distribute samples in focused regions.
	- sobol: Generate a uniform Sobol sequence (via SciPy).
	- sensitivity: Perform Sobol sensitivity analysis to measure each variable's importance on objective function results (via SALib).
	- waveform: Decompose the input waveform array (handles time- and frequency-domain via FFT / IFFT) into a diagnostic summary.

Modules:
	- test_functions: Contains test functions for local optimization testing.
	- waveform_analysis: Contains pulse generation functions.

Example Usage:

	# Import
	>>> import hdim_opt as h

	# Parameter Space
	>>> n_dimensions = 30
	>>> bounds = [(-100,100)] * n_dimensions
	>>> n_samples = 1000
	>>> obj_func = h.test_functions.rastrigin
	>>> time, pulse = h.waveform_analysis.e1_waveform()

	# Functions
	>>> solution, fitness = h.quasar(obj_func, bounds)
	>>> sens_matrix = h.sensitivity(obj_func, bounds)

	>>> hds_samples = h.hds(n_samples, bounds)
	>>> sobol_samples = h.sobol(n_samples, bounds)
	>>> isotropic_samples = h.isotropize(sobol_samples)

	>>> signal_data = h.waveform(x=time,y=pulse)
"""

# package version
__version__ = "1.3.0"
__all__ = ['quasar', 'hds', 'sobol', 'sensitivity', 'test_functions', 'quasar_helpers'] # available for star imports

# import core components
from .quasar_optimization import optimize as quasar
from .hyperellipsoid_sampling import sample as hds
from .sobol_sampling import sobol_sample as sobol
from .sobol_sensitivity import sens_analysis as sensitivity
from .waveform_analysis import analyze_waveform as waveform
from .quasar_helpers import isotropize
from .quasar_helpers import deisotropize
from . import test_functions
from . import quasar_helpers
from . import waveform_analysis