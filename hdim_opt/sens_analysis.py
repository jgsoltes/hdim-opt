import numpy as np
from scipy import stats
epsilon = 1e-12

### sensitivity analysis
def sensitivity(func, bounds, n_samples=2**7, 
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
        values = np.array([wrapped_func(sample) for sample in param_values])

    # run sensitivity analysis
    print('Analyzing sensitivities.')
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
        fig, ax = plt.subplots(1,1,figsize=(8, 5.5))
        
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
            s2_plot, ax = plt.subplots(1,1,figsize=(8, 5.5))
            
            top_idx_to_show = sort_idx[-num_to_plot:]
            S2_filtered = S2_df.iloc[top_idx_to_show, top_idx_to_show]
            mask_filtered = np.tril(np.ones_like(S2_filtered, dtype=bool))
            sns.heatmap(data=S2_filtered, mask=mask_filtered, annot=True, vmin=0.0, fmt='.2f')
            ax.set_title('Second-order Interactions ($S_2$)')
            ax.invert_yaxis()
            plt.tight_layout()
            plt.show()
    
    return Si_df, S2_df

### data transforms
def isotropize(data):
    '''
    Objective: 
        - Isotropizes the input matrix using Zero-Phase Component Analysis (ZCA).
            - Maintains original parameter orientation while removing correlations.
        - 'deisotropize' function inverse transforms to the original parameter space.
    '''
    from scipy.linalg import eigh
    # convert to array
    X = np.array(data)
    
    # standard scaling (mean = 0, var = 1)
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0) + epsilon # add epsilon to avoid div0
    X_centered = (X - mean) / stdev
    
    # eigen-decomposition of the correlation matrix
    cov = np.cov(X_centered, rowvar=False) + np.eye(X_centered.shape[1]) * epsilon
    eigenvalues, eigenvectors = eigh(cov) # eigh is more stable for symmetric matrices like covariance
    
    # ZCA whitening matrix: W_zca = U @ diag(1/sqrt(lambda)) @ U.T
    diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, epsilon))) # use maximum to avoid div0
    W_zca = eigenvectors @ diag_inv_sqrt @ eigenvectors.T # whitening matrix
    W_zca_inv = (eigenvectors * np.sqrt(np.maximum(eigenvalues, epsilon))) @ eigenvectors.T # save for deisotropization

    # transform: y = X_centered @ W_zca.T
    data_iso = np.dot(X_centered, W_zca) # no .T needed because W_zca is symmetric
    
    # store parameters for deisotropization
    params = {
        'mean': mean,
        'stdev': stdev,
        'W_zca': W_zca,
        'W_zca_inv': W_zca_inv
    }
    return data_iso, params

def deisotropize(data_iso, params):
    '''De-isotropize data to its original parameter space via inverse ZCA.'''
    
    # inverse ZCA: X_centered = y @ W_zca_inv.T
    data_centered = np.dot(data_iso, params['W_zca_inv'].T)
    
    # inverse scaling: X = (X_centered * std) + mean
    data_original = (data_centered * params['stdev']) + params['mean']
    return data_original

### pareto front
def pareto(func, bounds, targets=(), 
                 n_points=11, maxiter=33,
                 vectorized=False, seed=None,
                 log_scale=False, verbose=True):
    '''
    Objective:
        - Generates a Pareto front of optimization results.
        - Analyzes the cost trade-off between two objectives in a multi-objective cost function.
    Inputs:
        - func: Objective function to minimize.
        - bounds: Parameter bounds to search.
        - targets: Target variable names to analyze in the objective function (list of strings, match the variable names in func).
        
        - n_points: Number of points to generate.
        - maxiter: Number of iterations for the QUASAR optimization.
        - vectorized: Boolean for vectorized (matrix-level) objective functions.
        - seed: Random seed for reproducibility.
        
        - log_scale: Plot with log-scaled axes.
        - verbose: Boolean to display outputs.
    Outputs:
        - pareto_df: Dataframe of results from the evolutionary trials.

    Example Usage:
        >>> def obj_func(x, w1, w2):
        >>>    f1 = np.sum(x**2)
        >>>    f2 = np.sum((x - 5)**2)
        >>> return (w1*f1) + (w2*f2)
        >>> results = pareto(obj_func, bounds, ['w1','w2'])
    '''
    ### imports
    try:
        import numpy as np
        import inspect
        import functools
        import pandas as pd
        try:
            from quasar_optimization import optimize as quasar
        except:
            from .quasar_optimization import optimize as quasar
    except:
        raise ImportError('Failed to import a required package: pandas, inspect, functools, quasar_optimization.')
    np.random.seed(seed)

    # get variable names from objective function input
    sig = inspect.signature(func)

    # generate weight samples
    w1_vals = np.linspace(1.0, 0.0, n_points)
    w2_vals = 1.0 - w1_vals
    
    # converting None input
    if targets==None:
        targets = []

    # iterate through n_points to generate QUASAR results
    print('Generating Pareto front:')
    pareto_results = []
    for i in range(n_points):
        w1, w2 = w1_vals[i], w2_vals[i]

        # if targets are given
        if len(targets) > 1:
            weight_map = {targets[0]: w1, targets[1]: w2}
    
            @functools.wraps(func)
            def weighted_func(x, *args, **kwargs):
                for key, w in weight_map.items():
                    kwargs[key] = w
                return func(x, *args, **kwargs)
        else: # if empty target list, assume single-objective
            weighted_func = func

        # QUASAR optimization
        res_sol, res_fit = quasar(
            func=weighted_func, 
            bounds=bounds, 
            kwargs={t: 1.0 for t in targets},
            maxiter=maxiter,
            seed=seed,
            vectorized=vectorized,
            verbose=False
        )

        if len(targets) > 1:
            # raw scores
            if vectorized:
                # match QUASAR input shape
                eval_sol = res_sol.reshape(1, -1)
                f1_raw = func(eval_sol, **{targets[0]: 1.0, targets[1]: 0.0})
                f2_raw = func(eval_sol, **{targets[0]: 0.0, targets[1]: 1.0})
                
                # extract value from result array
                f1_score = f1_raw[0] if np.ndim(f1_raw) > 0 else f1_raw
                f2_score = f2_raw[0] if np.ndim(f2_raw) > 0 else f2_raw
            else:
                f1_score = func(res_sol, **{targets[0]: 1.0, targets[1]: 0.0})
                f2_score = func(res_sol, **{targets[0]: 0.0, targets[1]: 1.0})
            # total fitness
            total_cost = (f1_score * w1) + (f2_score * w2)
        
        else:
            total_cost = res_fit
            f1_score = res_fit
            f2_score = res_fit
        
        pareto_results.append({
            'obj1': f1_score,
            'obj2': f2_score,
            'total_cost': total_cost,
            'w1': w1,
            'w2': w2
        })
        if verbose:
            print(f'Trial {i+1}/{n_points} | w1: {w1:.2f} | f(x): {total_cost:.2e}')

    # convert to dataframe
    pareto_df = pd.DataFrame(pareto_results)

    ### plot
    if verbose:
        print('\nBest solution found:')
        print(pareto_df[pareto_df['total_cost'] == pareto_df['total_cost'].min()].head(1))
        print()

        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            import seaborn as sns
        except:
            raise ImportError('Failed to import visualization packages: matplotlib, seaborn.')
            
        plt.figure(figsize=(10, 6))
        
        # scatter of pareto front
        scatter = sns.scatterplot(data=pareto_df, x='obj1', y='obj2', 
                                    hue='w1', palette='magma', 
                                    size='w1', sizes=(20, 100),)
        
        # connect with small line
        plt.plot(pareto_df['obj1'], pareto_df['obj2'], color='lightgray', linestyle='--', alpha=0.5, zorder=0)
        
        plt.title('Pareto Front')
        plt.xlabel('Objective 1 Cost')
        plt.ylabel('Objective 2 Cost')
        plt.legend(title='Weight 1')
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
        plt.show()
        
    return pareto_df

### lorentzian KDE
def lorentzian(x, sigma, ensemble, verbose=True):
    '''
    Objective:
        - Lorentzian KDE with internal plotting.
    Inputs:
        - x: Coordinate to sample from the underlying KDE.
        - sigma: KDE bandwidth.
        - ensemble: Data ensemble to generate the KDE.
        - verbose: Boolean to display stats and plots.
    Outputs:
        - log_intensity: Logarithmic un-normalized intensity.
    '''
    # imports
    from scipy.spatial.distance import cdist
    from scipy.special import logsumexp, gammaln
    
    # clean input shapes
    ensemble = np.array(ensemble)
    if ensemble.ndim == 1:
        ensemble = ensemble.reshape(-1, 1) # if data is 1D, reshape it to (N, 1)
    N = ensemble.shape[0]
    n_dim = ensemble.shape[1]
    x = np.atleast_2d(x)
    if x.shape[1] != n_dim and x.shape[0] == n_dim: # if data is 1D, reshape it to (N, 1)
        x = x.T

    # calculate distances
    dists_sq = cdist(x, ensemble, metric='sqeuclidean')

    # log-Lorentzian kernels
    log_norm = (gammaln((n_dim + 1) / 2) - 
                (gammaln(1/2) + (n_dim/2) * np.log(np.pi) + n_dim * np.log(sigma)))
    
    log_kernels = log_norm - ((n_dim + 1) / 2) * np.log1p(dists_sq / (sigma**2))

    # integrate and normalize
    log_intensity = (logsumexp(log_kernels, axis=1) - np.log(N)).squeeze()

    ### plotting
    if verbose:
        print('KDE Coordinate:')
        display_vals = np.atleast_1d(log_intensity)        
        if len(display_vals) > 3:
            formatted_display = ', '.join([f'{val:.2e}' for val in display_vals[:3]])
            print(f'- Log Density: [{formatted_display}, ...]')
        else:
            formatted_display = ', '.join([f'{val:.2e}' for val in display_vals])
            print(f'- Log Density: [{formatted_display}]')

        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except:
            return pareto_df
        
        # set bounds
        mins = ensemble.min(axis=0)
        maxs = ensemble.max(axis=0)
        bounds = np.column_stack([mins, maxs])

        # 1D plot
        if n_dim == 1:
            x_range = np.linspace(mins[0], maxs[0], 500)
            grid_points = x_range[:, np.newaxis]
            
            # generate samples
            log_dens = lorentzian(grid_points, sigma, ensemble, verbose=False)
            y_vals = np.exp(log_dens)
            
            plt.figure(figsize=(8, 5))
            plt.plot(x_range, y_vals, lw=2, label='Lorentzian KDE')
            plt.fill_between(x_range, y_vals, alpha=0.2)
            plt.title('1D Density')
            plt.xlabel('')
            plt.ylabel('Density')
            plt.legend()
            plt.show()
        
        # 2d contour plot
        else:
            # PCA projection for D >= 2
            if n_dim > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                ensemble_2d = pca.fit_transform(ensemble)
                comp = pca.components_
                mean = pca.mean_
            else:
                ensemble_2d = ensemble

            # set bounds based on 2D projection
            mins = ensemble_2d.min(axis=0)
            maxs = ensemble_2d.max(axis=0)
            pad = (maxs - mins) * 0.2
            
            x_range = np.linspace(mins[0]-pad[0], maxs[0]+pad[0], 100)
            y_range = np.linspace(mins[1]-pad[1], maxs[1]+pad[1], 100)
            X_grid, Y_grid = np.meshgrid(x_range, y_range)
            grid_2d = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

            if n_dim > 2:
                # map 2d grid back to high-d for grid calculation
                grid_high_d = grid_2d @ comp + mean
                x_2d = pca.transform(x)
            else:
                grid_high_d = grid_2d
                x_2d = x

            log_dens = lorentzian(grid_high_d, sigma, ensemble, verbose=False)
            Z = np.exp(log_dens).reshape(X_grid.shape)

            ### plots
            fig = plt.figure(figsize=(16, 5))
            
            # 1d density plot (PCA 1)
            ax1 = fig.add_subplot(131)
            y_idx = np.argmin(np.abs(y_range - x_2d[0, 1]))

            # slice at y-coordinate of 'x' value
            ax1.plot(x_range, Z[y_idx, :], lw=2)
            ax1.fill_between(x_range, Z[y_idx, :], alpha=0.1, color='cornflowerblue')
            
            ax1.set_title('1D Density')
            ax1.set_xlabel('')
            ax1.set_ylabel('')
            ax1.set_yticks([])
            
            # 2D contour plot
            ax2 = fig.add_subplot(132)
            contour = ax2.contourf(X_grid, Y_grid, Z, levels=67, cmap='bone')
            ax2.scatter(ensemble_2d[:, 0], ensemble_2d[:, 1], s=2.0, alpha=0.33, color='white')
            ax2.set_title('2D Density Contour')
            
            # 3D topography plot
            ax3 = fig.add_subplot(133, projection='3d')
            ax3.plot_surface(X_grid, Y_grid, Z, cmap='bone', edgecolor='none', alpha=0.95)
            ax3.set_title('3D Topology')
            ax3.view_init(elev=35, azim=-60)
            ax3.set_zticks([])
            plt.show()
        
    return log_intensity