from scipy import stats
import numpy as np

def sobol_sample(n_samples, bounds, normalize=False, seed=None):
    '''
    Objective:
        - Generate a uniform scrambled Sobol sample sequence.
    Inputs:
        - n_samples: Number of samples to generate.
        - bounds: Range to sample over.
        - normalize: Boolean, if True keeps samples normalized to [0,1].
        - seed: Random seed.
    Outputs:
        - sobol_sequence: Sobol sample sequence.
    '''
    
    # clean bounds & n_dimensions
    bounds = np.array(bounds)
    n_dimensions = bounds.shape[0]
    
    sobol_sampler = stats.qmc.Sobol(d=n_dimensions, scramble=True, seed=seed)
    sobol_samples_unit = sobol_sampler.random(n=n_samples)
    
    if not normalize:
        sobol_sequence = stats.qmc.scale(sobol_samples_unit, bounds[:, 0], bounds[:, 1])
    else:
        sobol_sequence = sobol_samples_unit

    return sobol_sequence