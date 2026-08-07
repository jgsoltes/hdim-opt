import numpy as np
from scipy import stats
epsilon = np.finfo(float).tiny

### sensitivity analysis
try:
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
except:
    pass

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


### hyperslice of function
try:
    def hyperslice(func, bounds, slice_dims=(), n_samples=2**10, tol=1e-6, seed=None, verbose=True):
        '''
        Objective:
            - Generates an optimized hyperslice (0D to ND) of the objective's underlying solution manifold.
                - Uniformly samples the parameter space as a Sobol sequence.
                - Holds specified dimensions constant, locally optimizes all others at each Sobol grid coordinate.
    
        Inputs:
            - func: Objective function to slice.
            - bounds: Parameter bounds to evaluate.
            - slice_dims: Indices for the dimensions being held constant ([] for 0D, [0, 1, ...N] for N dimensions).
            - n_samples: Number of samples to distribute (preferred powers of 2).
            - tol: Objective value tolerance threshold for the local minimization.
            - seed: Random seed for reproducibility.
        
        Outputs:
            - slice_data: Hyperslice solution array.
            - slice_stats: Hyperslice statistics.
        '''
    
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.stats import qmc
        from scipy.optimize import minimize
        import matplotlib.tri as tri
        from sklearn.decomposition import PCA
        import time
    
        ### extract parameters
        n_samples = int(n_samples)
        n_dims = len(bounds)
        slice_dims = list(slice_dims) # convert input tuple to list
    
        # random seed
        if seed is None:
            seed = time.time()
        seed = int(seed)
        np.random.seed(seed)
            
        hidden_indices = [i for i in range(n_dims) if i not in slice_dims]
        hidden_bounds = [bounds[i] for i in hidden_indices]
        
        ### generate sobol sequence
        sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
        master_samples = sampler.random(n=n_samples)
        
        # scale samples to bounds
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        scaled_samples = qmc.scale(master_samples, lower_bounds, upper_bounds)
        
        # store results
        Z = np.zeros(n_samples)
    
        ### optimize sobol samples
        optimized_coords = np.zeros((n_samples, n_dims))
        for i in range(n_samples):
            current_sample = scaled_samples[i]
            
            # all dimensions fixed (essentially the QMC sequence)
            if len(hidden_indices) == 0:
                Z[i] = func(current_sample)
                optimized_coords[i] = current_sample
                
            # 0 dimensions fixed
            elif len(slice_dims) == 0:
                res = minimize(func, x0=current_sample, bounds=bounds, tol=tol)
                Z[i] = res.fun if res.success else func(current_sample)
                optimized_coords[i] = res.x if res.success else current_sample
                
            # multiple dimensions fixed
            else:
                locked_vis_vars = current_sample[slice_dims]
                
                def sub_objective(hidden_vars):
                    coord = np.zeros(n_dims)
                    coord[slice_dims] = locked_vis_vars
                    coord[hidden_indices] = hidden_vars
                    return func(coord)
                    
                starting_seed = current_sample[hidden_indices]
                res = minimize(sub_objective, x0=starting_seed, bounds=hidden_bounds, tol=tol)
                
                Z[i] = res.fun if res.success else sub_objective(starting_seed)
                
                # store actual coordinate
                full_coord = np.zeros(n_dims)
                full_coord[slice_dims] = locked_vis_vars
                full_coord[hidden_indices] = res.x if res.success else starting_seed
                optimized_coords[i] = full_coord
    
        ### statistics
        visible_volume = np.prod([bounds[i][1] - bounds[i][0] for i in slice_dims])
        dV = visible_volume / n_samples
        
        Z_pos = Z - np.min(Z) + 1e-16
        pdf = Z_pos / (np.sum(Z_pos) * dV)
        
        h_actual = -np.sum(pdf * np.log(pdf)) * dV
        h_max = np.log(visible_volume)
        norm_entropy = h_actual / h_max if h_max != 0 else np.nan
    
        slice_stats = {'diff_entropy': h_actual, 'norm_entropy': norm_entropy, 'min_obj': np.min(Z), 'max_obj': np.max(Z), 'stdev_obj': np.std(Z)}
    
        if verbose:
            ### print stats
            print(f'Hyperslice stats ({len(slice_dims)}D):')
            print(f'- Entropy: {norm_entropy:.2%}' if not np.isnan(norm_entropy) else '- Entropy: NaN (0D Volume)')
            print('- Objective Values:')
            print(f'   - Max: {np.max(Z):.3g}')
            print(f'   - Min: {np.min(Z):.3g}')
            print(f'   - Stdev: {np.std(Z):.3g}\n')
        
            ### plot
            plt.figure(figsize=(8, 6))
            
            if len(slice_dims) == 1:
                # sort for smooth 1D plotting
                X_vis = scaled_samples[:, slice_dims[0]]
                sort_idx = np.argsort(X_vis)
                plt.plot(X_vis[sort_idx], Z[sort_idx], color='indigo', linewidth=2)
                plt.fill_between(X_vis[sort_idx], Z[sort_idx], np.min(Z), color='indigo', alpha=0.1)
                plt.title('Manifold Hyperslice (1D)')
                plt.xlabel(f'Dimension {slice_dims[0]}')
                plt.ylabel('Objective Value')
                
            elif len(slice_dims) == 2:
                # triangulation surface over the Sobol points
                X_vis = scaled_samples[:, slice_dims[0]]
                Y_vis = scaled_samples[:, slice_dims[1]]
                triang = tri.Triangulation(X_vis, Y_vis)
                contour = plt.tricontourf(triang, Z, levels=30, cmap='inferno')
                plt.colorbar(contour, label='Objective Value')
                plt.title('Manifold Hyperslice (2D)')
                plt.xlabel(f'Dimension {slice_dims[0]}')
                plt.ylabel(f'Dimension {slice_dims[1]}')
                
            else:
                ### PCA for 3D+
                data_to_project = optimized_coords
                
                pca = PCA(n_components=2)
                projected = pca.fit_transform(data_to_project)
                
                scatter = plt.scatter(projected[:, 0], projected[:, 1], c=Z, cmap='inferno', alpha=0.8, edgecolor='k', s=40)
                plt.colorbar(scatter, label='Objective Value')
                
                title_prefix = fr'{len(slice_dims)}D Hyperslice ({n_dims} Space)'
                plt.title(f'{title_prefix}')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                
            plt.show()
    
        # store as dataframe
        data_matrix = np.column_stack((optimized_coords, Z))
        col_names = [f"x{i}" for i in range(n_dims)] + ['objective']
        slice_data = pd.DataFrame(data_matrix, columns=col_names)
        
        return slice_data, slice_stats
except:
    pass


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
            - log_intensity: Logarithmic intensity.
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


### analyze dataset
try:
    def analyze(data, transform=False):
        '''Quick analysis of data matrix.'''
    
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.linear_model import LinearRegression
        from matplotlib.collections import LineCollection
        from sklearn.metrics import r2_score
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from scipy import stats
        import numpy as np
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning) # suppress seaborn warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning) # suppress stats.entropy div0 warnings
    
        ### clean dataframe
        df = pd.DataFrame(data)
    
        # attempt to convert non-numeric to numeric
        obj_cols = df.select_dtypes(exclude=[np.number]).columns
        if not obj_cols.empty:
            df[obj_cols] = df[obj_cols].apply(pd.to_numeric, errors='ignore')
        
        # select numeric & non-null cols
        df = df.select_dtypes(include=[np.number])
        
        df = df.dropna(axis=1, how='all') # drop columns where all values are null
        df = df.dropna(axis=0, how='any') # drop rows where any value is null
        df = df.loc[:, df.var(ddof=0) > 0] # drop cols with zero variance
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
                data = np.arcsinh(data_raw)
                transform = True
                data_scaled = scaler.fit_transform(data)
                data_reduced = pca_2d.fit_transform(data_scaled)
                print('Arcsinh transform applied (unstable Z-scaling).')
            
            # force scaling if PCA is null
            if np.isnan(data_reduced).any() or np.isinf(data_reduced).any():
                data = np.arcsinh(data_raw)
                transform = True
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
                print(loadings.rename_axis('Dimension')[:-1].round(3).to_markdown())
                pca_variance = loadings[-1:].copy()
                pca_variance.rename(columns={'Magnitude':'Total'},inplace=True)
                print(pca_variance.round(3).to_markdown())
                print()
            
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
            
        except Exception as e:
            print(f'Bypassing metrics ({e})')
    
        
        ### plot
        try:
            fig, ax = plt.subplots(1,2,figsize=(12,5.5))
        
            # scatter plot
            ax[0].scatter(x=x,y=y,s=1)
            ax[0].set_xlabel('Axis 1')
            ax[0].set_ylabel('Axis 2')
            ax[0].set_title('Comparison')
            
            # density plot
            user_color = sns.color_palette()[0]
            bg_color = plt.rcParams['axes.facecolor'] 
            text_color = plt.rcParams['text.color']
        
            # colormap dark mode: black -> color -> white | light mode: white -> color -> black
            dynamic_cmap = sns.blend_palette([bg_color, user_color, text_color], as_cmap=True)
            sns.kdeplot(x=x, y=y, fill=True, cmap=dynamic_cmap, ax=ax[1], thresh=0.03)
            ax[1].set_title('Density')
            ax[1].set_xlabel('Axis 1')
            ax[1].set_ylabel('Axis 2')
            
            plt.tight_layout()
            plt.show()
        except:
            print(f'Skipped plot: {e}')
    
        
        ### correlations and ratios
        # correlation matrix
        df_transformed = pd.DataFrame(data, columns=param_names)
        corr_df = df_transformed.corr()

        try:
            # filter to top 10 correlations dimension
            if df.shape[1] > 10:
                overall_corr = corr_df.abs().mean().sort_values(ascending=False)
                top_vars = overall_corr.head(10).index
                corr_plot_data = corr_df.loc[top_vars, top_vars]
            else:
                corr_plot_data = corr_df
            corr_plot_data = corr_plot_data.rename(index=lambda x: str(x)[:15], columns=lambda x: str(x)[:10])
                
            ## correlations, etc
            if n_dim > 2:
                top_params = loadings.iloc[:-1].head(10).index.tolist()
                df_top = df_transformed[top_params]
                n_top = len(top_params)
                means = df_top.mean().values # top means
                ratio_matrix = means[:, None] / means[None, :] # ratios
                
                # rank-sum test
                top_vals = df_top.values
                n_top = len(top_params)
                p_matrix = np.ones((n_top, n_top))
                
                # upper triangle loop since p vals are symmetric
                for i in range(n_top):
                    for j in range(i + 1, n_top):
                        
                        # rank sum test
                        try:
                            _, p = stats.mannwhitneyu(top_vals[:, i], top_vals[:, j])
                        except ValueError:
                            # edge cases where all numbers are identical
                            p = 1.0 
                        
                        # assign symmetrically ( p(1,2) = p(2,1) )
                        p_matrix[i, j] = p
                        p_matrix[j, i] = p
            
                # create ratio dataframe
                ratio_df = pd.DataFrame(ratio_matrix, index=top_params, columns=top_params)
        
        
            ### plot feature distributions
            fig, ax = plt.subplots(1,2,figsize=(12,5.5))
            if n_dim > 2:
                sns.boxplot(data=df_top, ax=ax[0])
            else:
                sns.boxplot(data=df_transformed, ax=ax[0])
        
            boxplot_title = 'Feature Distributions (Arcsinh)' if transform else 'Feature Distributions'
            value_label = 'Value (Arcsinh)' if transform else 'Value'
            ax[0].set_title(boxplot_title)
            ax[0].set_ylabel(value_label)
        
            # plot overlaid feature KDE densities
            if n_dim > 2:
                sns.kdeplot(data=df_top, fill=True, common_norm=False, alpha=0.3, cut=0, ax=ax[1])
            else:
                sns.kdeplot(data=df_transformed, fill=True, common_norm=False, alpha=0.3, cut=0, ax=ax[1])
            feat_density_title = 'Feature Densities (Arcsinh)' if transform else 'Feature Densities'
            ax[1].set_title(feat_density_title)
            ax[1].set_xlabel(value_label)
            ax[1].set_ylabel('')
            
            plt.tight_layout()
            plt.show()
        
            ### plot correlations and ratios
            if n_dim > 2:
                fig, ax = plt.subplots(1,2,figsize=(12,5))
                
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
                    
                sns.heatmap(ratio_df, annot=annot_matrix, fmt='', center=1, ax=ax[1], cbar_kws={'label': '[ * = p < 0.05 ]'})
                ratio_title = 'Ratios (Arcsinh)' if transform else 'Ratios'
                ax[1].set_title(ratio_title)
                
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f'Skipped plot: {e}')
    

        try:
            ### plot parallel dimensions
            n_keep = 5
            variances = np.var(data, axis=0)
            top_var_indices = np.argsort(variances)[-n_keep:][::-1]
            
            var_data = data[:, top_var_indices]
            var_labels = param_names[top_var_indices]
            
            # pre-calculated PCA data
            pca_data = data_reduced
            n_components = 2
            pca_labels = [f'PC {i+1}' for i in range(n_components)]
            
            ### normalize data
            def normalize_for_plot(matrix):
                c_min = np.min(matrix, axis=0)
                c_max = np.max(matrix, axis=0)
                c_range = np.where(c_max > c_min, c_max - c_min, 1.0)
                return (matrix - c_min) / c_range
            norm_var_data = normalize_for_plot(var_data)
            norm_pca_data = normalize_for_plot(pca_data)
        
            ### plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            x_coords_var = np.arange(n_keep)
            x_coords_pca = np.arange(n_components)
            
            # tie dimensions to their corresponding principal component
            colors = plt.cm.plasma(norm_pca_data[:, 0])
        
            # render via LineCollection
            var_segments = np.zeros((data.shape[0], n_keep, 2))
            var_segments[:, :, 0] = x_coords_var
            var_segments[:, :, 1] = norm_var_data
            pca_segments = np.zeros((data.shape[0], n_components, 2))
            pca_segments[:, :, 0] = x_coords_pca
            pca_segments[:, :, 1] = norm_pca_data
            var_lc = LineCollection(var_segments, colors=colors, alpha=0.5, linewidths=1.5)
            pca_lc = LineCollection(pca_segments, colors=colors, alpha=0.5, linewidths=1.5)
            ax1.add_collection(var_lc)
            ax2.add_collection(pca_lc)
        
            # initialize scatter plot
            x_flat_var = np.tile(x_coords_var, data.shape[0])
            color_flat_var = np.repeat(colors, n_keep, axis=0)
            ax1.scatter(x_flat_var, norm_var_data.flatten(), color=color_flat_var, alpha=0.5, s=3.33, zorder=3)
        
            x_flat_pca = np.tile(x_coords_pca, data.shape[0])
            color_flat_pca = np.repeat(colors, n_components, axis=0)
            ax2.scatter(x_flat_pca, norm_pca_data.flatten(), color=color_flat_pca, alpha=0.5, s=3.33, zorder=3)
        
            # plot top variance dimensions
            ax1.set_xlim(x_coords_var[0], x_coords_var[-1])
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_xticks(x_coords_var)
            ax1.set_xticklabels(var_labels, rotation=45)
            ax1.set_ylabel('Normalized Value')
            ax1.set_title(f'Top Dimensions by Variance')
            ax1.grid(True, axis='x', linestyle='--', alpha=0.7)
            
            # plot PCA
            ax2.set_xlim(x_coords_pca[0], x_coords_pca[-1])
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_xticks(x_coords_pca)
            ax2.set_xticklabels(pca_labels, rotation=45)
            ax2.set_title(f'Principal Components')
            ax2.grid(True, axis='x', linestyle='--', alpha=0.7)
            ax2.set_yticks([])
            
            plt.tight_layout()
            plt.show()
        except:
            print(f'Skipped plot: {e}')

except:
    pass

### symbolic regression
try:
    def symbolic(X_or_func, y_or_bounds, 
                            feature_names=None, generations=10, population_size=2**10, tournament_size=None,
                            parsimony=0.005, init_depth=(2, 3), const_range=(-1, 1), function_set=None,
                            n_jobs=1, seed=None, verbose=True):
        '''
        - Performs symbolic regression (via gplearn) on the input data or function.
        - If input is a function, the landscape is first estimated by optimizing a uniform (Sobol) sample sequence.
        - This sequence becomes the data for the model fit.
        
        Inputs:
            - X_or_func: Feature (X) data, or function to estimate.
            - y_or_bounds: Target (y) data, or bounds to evaluate for function.
            - feature_names: List of feature names (auto-extracted if X is dataframe).
            - generations: Number of evolutionary generations.
            - population_size: Population size.
            - parsimony: Parsimony coefficient, to penalize longer expressions.
            - init_depth: Initial expression depth (number of terms).
            - const_range: Range to explore for constant values.
            - function_set: List of strings defining gplearn functions to test; i.e., ['add','sub','div']
                - Default: ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'inv')
            - n_jobs: Number of cores to utilize.
            - seed = Random seed for reproducibility.
            - verbose: Display progress and outputs.
            
        Outputs:
            - results_df: DataFrame containing all evaluated programs, refined formulas, and metrics.
            - best_eq: The top-performing analytical expression.
        '''
    
        import time
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.stats import qmc
        from gplearn.genetic import SymbolicRegressor
        from gplearn.functions import make_function
        from sklearn.metrics import r2_score, mean_absolute_error
        from sklearn.linear_model import LinearRegression
    
        # random seed
        if seed is None:
            seed = int(time.time())
        else:
            seed = int(seed)
        np.random.seed(seed)
    
        ### determine if input is function or dataset
        if callable(X_or_func):
            # if input is function
            if verbose:
                print('Estimating function landscape using Sobol sequence.\n')
            
            func = X_or_func
            bounds = y_or_bounds
            n_dims = len(bounds)
            
            # generate sobol sequence
            sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
            master_samples = sampler.random(n=population_size)
            
            lower_bounds = np.array([b[0] for b in bounds])
            upper_bounds = np.array([b[1] for b in bounds])
            X_vals = qmc.scale(master_samples, lower_bounds, upper_bounds)
            
            # evaluate objective
            y = np.array([func(x) for x in X_vals]).reshape(-1, 1)
            
            if feature_names is None:
                feature_names = [f'x{i}' for i in range(n_dims)]
                
        else:
            # if input is dataset
            if isinstance(X_or_func, pd.DataFrame):
                if feature_names is None:
                    feature_names = X_or_func.columns.tolist()
                X_vals = X_or_func.values
            else:
                X_vals = np.asarray(X_or_func)
                
            y = np.asarray(y_or_bounds).reshape(-1, 1)
            
            if feature_names is None:
                feature_names = [f"x{i}" for i in range(X_vals.shape[1])]
    
        y = np.asarray(y).reshape(-1, 1)
        
        # Z-scale target
        y_mean, y_std = y.mean(), y.std()
        y_scaled = (y - y_mean) / (y_std if y_std > 0 else 1.0)
    
        # define function set to test
        if function_set is None:
            function_set = ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'inv')
    
        if tournament_size is None:
            tournament_size = min(max(2, population_size // 8), 128)
        
        ### genetic programming architecture
        est_gp = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            init_depth=init_depth,
            parsimony_coefficient=parsimony,
            max_samples=1.0,
            const_range=const_range,
            function_set=function_set,
            feature_names=feature_names,
            verbose=verbose,
            n_jobs=n_jobs,
            random_state=seed
        )
    
        # fit model
        if verbose:
            print('Fitting symbolic regressor.')
        est_gp.fit(X_vals, y_scaled.ravel())
    
        ### OLS linear refinement
        all_programs = est_gp._programs[-1]
        refined_stats = []
        if verbose:
            print('\nRefining outputs.')
        for idx, program in enumerate(all_programs):
            if program is None:
                continue
                
            y_shape = program.execute(X_vals).reshape(-1, 1)
            
            if np.any(np.isnan(y_shape)) or np.any(np.isinf(y_shape)):
                continue
    
            # OLS linear refinement (intercept & scaling correction)
            refiner = LinearRegression().fit(y_shape, y)
            y_final = refiner.predict(y_shape)
            
            r2 = r2_score(y, y_final)
            mae = mean_absolute_error(y, y_final)
            
            raw_formula = str(program)
            intercept = refiner.intercept_[0]
            coefficient = refiner.coef_[0][0]
            refined_formula = f"{coefficient:.3g} * ({raw_formula}) + {intercept:.3g}"
    
            refined_stats.append({
                'index': idx,
                'program_obj': program,
                'refiner_model': refiner,
                'y_pred': y_final,
                'R2': r2,
                'MAE': mae,
                'Length': program.length_,
                'Formula': raw_formula,
                'Refined_Formula': refined_formula
            })
    
        # compile final results
        results_df = pd.DataFrame(refined_stats)
        if not results_df.empty:
            results_df = results_df.sort_values(by=['R2', 'MAE'], ascending=[False, True]).reset_index(drop=True)
        best_eq = results_df['program_obj'].iloc[0] if not results_df.empty else None
    
        ### display results
        if verbose and not results_df.empty:
            print('\nTop 10 Equations (by R^2):')
            pd.options.display.max_colwidth = 120
            print(results_df[['R2', 'MAE', 'Length', 'Refined_Formula']].head(5))
            print(f"\nBest Equation:\n{results_df['Refined_Formula'].iloc[0]}")
            print(f"- R^2: {results_df['R2'].iloc[0]:.4g}")
            print(f"- MAE: {results_df['MAE'].iloc[0]:.4g}")
    
            ### plot top result
            row = results_df.iloc[0]
            plt.figure(figsize=(6, 4))
            plt.scatter(y, row['y_pred'], alpha=0.3, s=2)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', alpha=0.5, label='Perfect Fit')
            plt.title(fr"Best Expression ($R^2$: {results_df['R2'].iloc[0]:.3f})")
            plt.xlabel('Actual Target')
            plt.ylabel('Symbolic Prediction')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return results_df, best_eq

except:
    pass