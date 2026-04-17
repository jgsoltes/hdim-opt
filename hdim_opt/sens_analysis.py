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
            sns.heatmap(data=S2_filtered, mask=mask_filtered, annot=True, fmt='.2f')
            ax.set_title('Second-order Interactions ($S_2$)')
            ax.invert_yaxis()
            plt.yticks(rotation=0)
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
try:
    from numba import njit, prange
    @njit(parallel=True, fastmath=True)
    def _numba_lorentzian(x, ensemble, inv_sigma_sq, log_norm, eff_dim_plus_1_over_2, is_balloon):
        '''Numba wrapped for fast kernel evaluation.'''
        N_queries = x.shape[0]
        N_ensemble = ensemble.shape[0]
        results = np.zeros(N_queries)
        log_N = np.log(float(N_ensemble))
    
        for i in prange(N_queries):
            # if Balloon: bandwidth is fixed for the query (outer loop)
            # if Pointwise: extract inside the inner loop
            if is_balloon:
                s_inv_fixed = inv_sigma_sq[i]
                l_norm_fixed = log_norm[i]
            
            temp_kernels = np.zeros(N_ensemble)
            for j in range(N_ensemble):
                # distance calculation
                dist_sq = 0.0
                for k in range(x.shape[1]):
                    diff = x[i, k] - ensemble[j, k]
                    dist_sq += diff * diff
                
                # select bandwidth logic
                if is_balloon:
                    s_inv = s_inv_fixed
                    l_norm = l_norm_fixed
                else:
                    s_inv = inv_sigma_sq[j]
                    l_norm = log_norm[j]
                
                ### kernel calculation
                temp_kernels[j] = l_norm - eff_dim_plus_1_over_2 * np.log1p(dist_sq * s_inv)
            
            # LogSumExp for numerical stability
            max_val = np.max(temp_kernels)
            sum_exp = 0.0
            for j in range(N_ensemble):
                sum_exp += np.exp(temp_kernels[j] - max_val)
            
            results[i] = max_val + np.log(sum_exp) - log_N
            
        return results

    def lorentzian(x, sigma, ensemble, eff_dim=None, estimator='balloon', verbose=False):
        '''
        Objective:
            - Lorentzian KDE with internal plotting.
        Inputs:
            - x: Coordinate(s) to sample from the underlying KDE.
            - sigma: KDE bandwidth.
            - ensemble: Kernel ensemble for KDE.
            - eff_dim: Effective dimension for multivariate calculation.
            - estimator: For variable-bandwidth KDEs ('balloon', 'pointwise').
                - Balloon applies query bandwidth; pointwise applies kernel bandwidths.
            - verbose: Boolean to display stats and plots.
        Outputs:
            - log_intensity: Logarithmic un-normalized intensity.
        '''
        from scipy.special import gammaln
        ensemble = np.ascontiguousarray(np.atleast_2d(ensemble))
        x = np.ascontiguousarray(np.atleast_2d(x))
    
        # extract parameters
        N_ensemble, n_dim = ensemble.shape
        M_queries = x.shape[0]
        eff_dim = eff_dim if eff_dim is not None else n_dim
    
        # ensure sigma matches the shape of the estimator
        sigma_vec = np.atleast_1d(sigma).flatten()
        
        ### pre-calculate constants
        log_pi = np.log(np.pi)
        eff_dim_plus_1_over_2 = (eff_dim + 1) / 2
        inv_sigma_sq = 1.0 / (sigma_vec**2)
        log_norm = (gammaln(eff_dim_plus_1_over_2) - 
                   (gammaln(0.5) + (eff_dim/2) * log_pi + eff_dim * np.log(sigma_vec)))
    
        if estimator == 'pointwise':
            # sigma_vec must match ensemble
            if sigma_vec.size != N_ensemble:
                # if scalar, expand to match ensemble size
                if sigma_vec.size == 1:
                    inv_sigma_sq = np.full(N_ensemble, inv_sigma_sq[0])
                    log_norm = np.full(N_ensemble, log_norm[0])
                else:
                    raise ValueError(f"Pointwise requires sigma size ({sigma_vec.size}) to match ensemble size ({N_ensemble})")
            log_intensity = _numba_lorentzian(x, ensemble, inv_sigma_sq, log_norm, eff_dim_plus_1_over_2, is_balloon=False)
            return log_intensity.squeeze()
    
        elif estimator == 'balloon':
            # balloon: sigma_vec must match M_queries
            if sigma_vec.size != M_queries:
                if sigma_vec.size == 1:
                    inv_sigma_sq = np.full(M_queries, inv_sigma_sq[0])
                    log_norm = np.full(M_queries, log_norm[0])
                else:
                    raise ValueError(f"Balloon requires sigma size ({sigma_vec.size}) to match query size ({M_queries})")
            log_intensity = _numba_lorentzian(x, ensemble, inv_sigma_sq, log_norm, eff_dim_plus_1_over_2, is_balloon=True)
        log_intensity = log_intensity.squeeze()
    
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
                return log_intensity
            
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
                ax3.set_title('3D Topography')
                ax3.view_init(elev=35, azim=-60)
                ax3.set_zticks([])
                plt.show()
            
        return log_intensity
except:
    pass


def analyze(data, transform=False):
    '''Quick analysis of data matrix.'''

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    ### convert to dataframe
    df = pd.DataFrame(data).select_dtypes(include=[np.number])
    param_names = df.columns
    data_raw = df.values

    # arcsinh transform for orders of magnitude
    if transform:
        data = np.arcsinh(data_raw)
        print('Arcsinh transform applied.\n')
    else:
        data = data_raw

     # scaled data for PCA
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # n_dimensions
    n_dim = data.shape[1]
    if n_dim == 0:
        print('No numeric columns found.')
        return None

    ### PCA
    if n_dim > 2:
        pca_2d = PCA(n_components=2)
        try:
            data_reduced = pca_2d.fit_transform(data_scaled)
        except ValueError:
            data = np.arcsinh(data)
            data_scaled = scaler.fit_transform(data)
            data_reduced = pca_2d.fit_transform(data_scaled)
            print('Arcsinh transform applied (unstable Z-scaling).')
        
        # force scaling if PCA is null
        if np.isnan(data_reduced).any() or np.isinf(data_reduced).any():
            data = np.arcsinh(data_raw)
            data_scaled = scaler.fit_transform(data)
            data_reduced = pca_2d.fit_transform(data_scaled)
            print('Arcsinh transform applied (unstable covariance).')
        x, y = data_reduced[:,0], data_reduced[:,1]

        ### loadings
        loadings = pd.DataFrame(pca_2d.components_.T, columns=['PC1', 'PC2'], index=param_names)
        loadings['Magnitude'] = np.sqrt(loadings['PC1']**2 + loadings['PC2']**2)
        loadings = loadings.sort_values(by='Magnitude', ascending=False).head(15)
        
        # variance
        var_pc1 = pca_2d.explained_variance_ratio_[0]
        var_pc2 = pca_2d.explained_variance_ratio_[1]
        total_var = var_pc1 + var_pc2

        variance_row = pd.DataFrame(
            [[var_pc1, var_pc2, total_var]], 
            columns=['PC1', 'PC2', 'Magnitude'], 
            index=['Explained Var.']
            )
        loadings = pd.concat([loadings, variance_row])

    # 2d plots
    elif n_dim == 2:
        x, y = data[:,0], data[:,1]

    # 1d plot
    else:
        print('- Stats:')
        print(f'Mean: {np.mean(data):.3g}')
        print(f'Median: {np.median(data):.3g}')
        print(f'Stdev: {np.std(data):.3g}')
        
        sns.kdeplot(x=data.flatten(),alpha=0.75)
        plt.title('Probability Density')
        plt.xlabel('Value')
        plt.show()
        return None

    # remove nulls from data
    valid_indices = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[valid_indices], y[valid_indices]
    
    ### pairwise comparison
    try:
        ### normalized standard deviation (occupancy)
        N = len(data)
        span = data.max() - data.min()
        dim_sds = np.std(data, axis=0, ddof=1)
        norm_sds = dim_sds / span
        
        # standard error of the standard deviation (for a normal-ish distribution) is approx sigma / sqrt(2N)
        dim_se = dim_sds / np.sqrt(2 * N) 
        norm_se = dim_se / span
        total_occupancy = np.sqrt(np.mean(norm_sds**2))
        total_occupancy_se = np.mean(norm_se) # average uncertainty across the 11-D space

        ### normalized differential entropy
        efficiencies = []
        
        # iterate through dimensions
        col_spans = []
        for i in range(data.shape[1]):
            column_data = data[:, i]
            
            # actual differential entropy (in nats)
            h_actual = stats.differential_entropy(column_data)
            col_span = np.max(column_data) - np.min(column_data) # max possible entropy assumes uniform distribution over the span
            if col_span > 0:
                efficiencies.append(np.exp(h_actual) / col_span)
                col_spans.append(col_span)
            else:
                efficiencies.append(0.0)
        col_spans = np.array(col_spans)
        efficiencies = np.array(efficiencies)
        norm_entropy = np.mean(efficiencies)
        entropy_se = np.std(efficiencies, ddof=1) / np.sqrt(len(efficiencies)) # standard error across dimensions
        norm_span = np.linalg.norm(col_spans)
        print('Stats:')
        print(f'- Norm. Stdev: {total_occupancy:.1%} ± {total_occupancy_se:.1%}')
        print(f'- Entropy: {norm_entropy:.1%} ± {entropy_se:.1%}')
        print(f'- Span: {norm_span:.3g}\n')
        
        ### print metrics
        if n_dim > 2:
            print('Principal Axes:')
        
        # 2d metrics
        else:
            print('Axes:')
            ### calculate metrics
            # pearson correlation
            corr = stats.pearsonr(x, y)
    
            # wilcoxon signed-rank test
            t_stat, p_value = stats.wilcoxon(y,x)
            x_mean = np.mean(x)
            ratio = np.mean(y) / x_mean if x_mean != 0 else np.nan # crashes if denom is 0
    
            # linear regression
            lin_model = LinearRegression()
            lin_model.fit(x.reshape(-1,1),y.reshape(-1,1))
            r2 = lin_model.score(x.reshape(-1,1),y.reshape(-1,1))
            print(f'- Correlation: {corr[0]:.3f} (p={corr[1]:.3e})')
            print(f'- Regression: y = {lin_model.coef_[0][0]:.3g}x + {lin_model.intercept_[0]:.3g}  (r2={r2:.3f})')
            print(f'- Ratio: {ratio:.2g} (p={p_value:.3g})\n')
        print(loadings.rename_axis('Dimension')[:-1].round(3).to_markdown())
        pca_variance = loadings[-1:].copy()
        pca_variance.rename(columns={'Magnitude':'Total'},inplace=True)
        print(pca_variance.round(3).to_markdown())
        # print(loadings[-1:].round(3).to_markdown())
        print()
        
    except Exception as e:
        print(f'Bypassing metrics ({e})')

    ### plot
    fig, ax = plt.subplots(1,2,figsize=(11,5.5))

    # scatter
    ax[0].scatter(x=x,y=y,s=1)
    ax[0].set_xlabel('Axis 1')
    ax[0].set_ylabel('Axis 2')
    ax[0].set_title('Principal Components')

    # 2-d
    sns.kdeplot(x=x,alpha=0.75,label='Axis 1', ax=ax[1], common_norm=False)
    sns.kdeplot(x=y,alpha=0.75,label='Axis 2', ax=ax[1], common_norm=False)
    ax[1].set_title('Principal Components')
    ax[1].set_xlabel('')
    ax[1].set_ylabel('')
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    
    ### comparisons
    # correlation matrix
    df_transformed = pd.DataFrame(data, columns=param_names)
    corr_df = df_transformed.corr()
    
    # filter to top 10 correlations dimension
    if df.shape[1] > 10:
        overall_corr = corr_df.abs().mean().sort_values(ascending=False)
        top_vars = overall_corr.head(10).index
        corr_plot_data = corr_df.loc[top_vars, top_vars]
    else:
        corr_plot_data = corr_df
    corr_plot_data = corr_plot_data.rename(index=lambda x: str(x)[:15], columns=lambda x: str(x)[:10])
    
    # relative ratios
    top_params = loadings.iloc[:-1].head(15).index.tolist()
    df_top = df[top_params]
    n_top = len(top_params)
    means = df_top.mean().values # top means
    ratio_matrix = means[:, None] / means[None, :] # ratios
    
    # wilcoxon p-values
    p_matrix = np.zeros((n_top, n_top))
    for i in range(n_top):
        for j in range(n_top):
            if i == j:
                p_matrix[i, j] = 1.0
            else:
                _, p = stats.ttest_rel(df_top.iloc[:, i], df_top.iloc[:, j])
                p_matrix[i, j] = p

    # create ratio dataframe
    ratio_df = pd.DataFrame(ratio_matrix, index=top_params, columns=top_params)
    
    ### plot comparisons
    fig, ax = plt.subplots(1,2,figsize=(11.5,5))
    
    # correlations
    sns.heatmap(corr_plot_data, ax=ax[0], center=0)
    ax[0].set_title('Correlations')

    # ratios
    annot_matrix = []
    for i in range(len(top_params)):
        row_annot = []
        for j in range(len(top_params)):
            r = ratio_matrix[i, j]
            p = p_matrix[i, j]
            star = '*' if p < 0.05 else ''
            row_annot.append(f'{star}')
        annot_matrix.append(row_annot)
        
    # sns.heatmap(ratio_df, annot=annot_matrix, fmt='', center=1, ax=ax[1])
    sns.heatmap(ratio_df, annot=annot_matrix, fmt='', center=1, ax=ax[1], cbar_kws={'label': '[ * = p < 0.05 ]'})
    ax[1].set_title('Ratios')
    
    plt.tight_layout()
    plt.show()