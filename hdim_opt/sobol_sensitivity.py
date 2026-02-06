import numpy as np
from scipy import stats

### sensitivity analysis
def sens_analysis(func, bounds, n_samples=2**7, 
                  args=None, kwargs=None, 
                  param_names=None, calc_second_order=True, 
                  log_scale=False, num_to_plot=10, verbose=True):
    '''
    Objective:
        - Perform global Sobol sensitivity analysis on the objective function.
        - Utilizes the SALib package.
    Inputs:
        - func: Objective function (Problem) to analyze.
        - bounds: Parameter space bounds, as an array of tuples.
        - n_samples: Number of Sobol samples to generate.
        - kwargs: Keyword arguments (dictionary) for objective function.
        - param_names: Optional parameter names for each dimension.
        - calc_second_order: Boolean to calculate second-order interactions. Disable to improve computation speed.
        - log_scale: Boolean to log-scale plots.
        - num_to_plot: Number of dimensions to plot.
        - verbose: Boolean to display plots.
    Outputs:
        - Si: Matrix of first- and total-order sensitivity indices and confidences.
        - S2_matrix: Matrix of second-order interactions.
    '''

    ### imports
    try:
        import numpy as np
        from SALib.sample import sobol as sobol_sample
        from SALib.analyze import sobol as sobol_analyze
        import pandas as pd
        import time
    except ImportError as e:
        raise ImportError(f'Sensitivity analysis requires dependencies: (SALib, pandas).') from e
    
    ### extract inputs
    start_time = time.time()
    bounds = np.array(bounds)
    n_params = bounds.shape[0]
    if param_names == None:
        param_names = range(0,n_params)
    elif len(param_names) != len(bounds):
        raise ValueError('Length of param_names does not match length of bounds.')
    
    ### define problem for SALib
    problem = {
        'num_vars': n_params,
        'names': param_names,
        'bounds' : bounds
        }

    ### generate samples
    if verbose:
        print(f'Generating Sobol samples (N={n_samples:,.0f}, D={n_params}).')
    param_values = sobol_sample.sample(problem, n_samples, calc_second_order=calc_second_order)

    ### args / kwargs for the objective function
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    def wrapped_func(x_samples):
        return func(x_samples, *args, **kwargs)

    ### evaluate samples
    # vectorized evaluation
    n_expected = param_values.shape[0]
    try:
        values = wrapped_func(param_values)
        values = np.asarray(values).flatten()
        if values.shape[0] != n_expected:
            raise ValueError('Non-vectorized objective function.')

    # loop-based evaluation
    except ValueError as e:
        if verbose:
            print(f'Non-vectorized objective function; loop-based evaluation.')
        values = np.array([wrapped_func(sample) for sample in param_values])

    # run sensitivity analysis
    print('Running sensitivity analysis.')
    Si = sobol_analyze.analyze(problem, values, calc_second_order=calc_second_order, print_to_console=False)

    # create Si output dataframe
    Si_keys = ['S1', 'S1_conf', 'ST', 'ST_conf']
    Si_filtered = {k: Si[k] for k in Si_keys if k in Si} # filter for output
    Si_df = pd.DataFrame(Si_filtered, index=param_names)

    # create S2 output dataframe
    if calc_second_order:
        S2_matrix = Si['S2']
        S2_df = pd.DataFrame(S2_matrix, index=param_names, columns=param_names)
        S2_df = S2_df.fillna(S2_df.T)
    else:
        S2_df = pd.DataFrame()

    ### end of calculations
    end_time = time.time()
    run_time = end_time - start_time
    if verbose:
        num_to_plot = np.minimum(num_to_plot, n_params)
        print(f'\nRun time: {run_time:.2f}s')
        # plotting imports
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError as e:
            raise ImportError(f'Plotting requires dependencies: (matplotlib, seaborn).') from e

        # sort by S1
        sort_idx = np.argsort(Si['S1'])
        s1_sorted = Si['S1'][sort_idx][-num_to_plot:]
        st_sorted = Si['ST'][sort_idx][-num_to_plot:]
        s1_conf_sorted = Si['S1_conf'][sort_idx][-num_to_plot:]
        st_conf_sorted = Si['ST_conf'][sort_idx][-num_to_plot:]
        names_sorted = [np.array(param_names)[i] for i in sort_idx][-num_to_plot:]
        index = np.arange(len(names_sorted))

        ### plot 1: first-order (S1) and total-order (ST) indices
        fig, ax = plt.subplots(1,1,figsize=(9, 7))
        
        bar_width = 0.35
        ax.barh(index + bar_width/2, s1_sorted, bar_width, xerr=s1_conf_sorted, 
                   label='First-order ($S_1$)',
                   alpha=1,
                   capsize=2.5)
        ax.set_yticks(index)
        ax.set_yticklabels(names_sorted)
    
        ax.barh(index - bar_width/2, st_sorted, bar_width,
                   xerr=st_conf_sorted, 
                   label='Total-order ($S_T$)',
                   alpha=0.75, 
                   capsize=2.5)
        if log_scale:
            ax.set_xscale('log')
        ax.set_title('Sensitivity Indices ($S_1$, $S_T$)')
        ax.legend()
        plt.tight_layout()
        plt.show()

        if calc_second_order:
            ### plot 2: heatmap of second order indices
            s2_plot, ax = plt.subplots(1,1,figsize=(9, 7))
            
            top_idx_to_show = sort_idx[-num_to_plot:]
            S2_filtered = S2_df.iloc[top_idx_to_show, top_idx_to_show]
            mask_filtered = np.tril(np.ones_like(S2_filtered, dtype=bool))
            sns.heatmap(data=S2_filtered, mask=mask_filtered, annot=True, vmin=0.0, fmt='.2f')
            ax.set_title('Second-order Interactions ($S_2$)')
            ax.invert_yaxis()
            plt.tight_layout()
            plt.show()
    
    return Si_df, S2_df

### sobol sampling
def sobol_sample(n_samples, bounds, normalize=False, seed=None):
    '''
    Objective:
        - Generates a uniform scrambled Sobol sample sequence.
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