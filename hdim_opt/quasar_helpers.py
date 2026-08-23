# global imports
import numpy as np
epsilon = np.finfo(float).tiny
    
############## CONSTRAINTS ##############
def apply_penalty(fitnesses, solutions, constraints, constraint_penalty, vectorized):
    '''
    Objective:
        - Calculates the penalty for constraint violations and adds it to the fitness.
    Inputs:
        - fitnesses: Array of objective function values (without penalty).
        - solutions: Population or trial vectors (NxD or DxN array).
        - constraints: Dictionary of constraints.
        - vectorized: Boolean flag for population shape.
    Outputs:
        - penalized_fitnesses: Array of fitnesses + penalty.
    '''
    
    if not constraints:
        return fitnesses

    # define shape based on vectorized flag
    if vectorized:
        solutions = solutions.T # consistent with optimization calls

    if fitnesses.ndim > 1:
        # collapse extra dimensions
        fitnesses = fitnesses
    
    # initialize penalty array/matrix
    penalties = np.zeros_like(fitnesses)
    
    for con_name, con_spec in constraints.items():
        con_func = con_spec[0] # constraint function
        con_type = con_spec[1] # relational operator, i.e. '<='
        con_rhs = con_spec[2]  # logical reference value, i.e. 0

        # calculate constraint value
        if vectorized:
            # constraints assumed to accept NxD matrix and return N-D array
            con_values = con_func(solutions)
        else:
            # sequential calculations for non-vectorized constraints
            con_values = np.array([con_func(sol) for sol in solutions])

        # identify violation (g(x) - 0 for g(x) <= 0)
        violation = con_values - con_rhs

        if con_type in ('<=', '<'):
            # violation is max(0, g(x) - 0)
            violation[violation < 0] = 0
            penalties += violation * constraint_penalty
            
        elif con_type in ('>=', '>'):
            # violation is max(0, 0 - g(x))
            violation_reverse = con_rhs - con_values
            violation_reverse[violation_reverse < 0] = 0
            penalties += violation_reverse * constraint_penalty
        else:
            raise ValueError("""Constraint logical operators must be one of ['<=', '<', '>=', '>'].
            Additional logic should be incorporated into the constraint functions.""")
        
    return fitnesses + penalties


############## POLISHING ##############

### minimization wrapper
def minimize(func, bounds, 
             method='L-BFGS-B', init=None,
             constraints=None, tolerance=None,
             verbose=True, **kwargs):
    '''
    Objective:
        - Minimize the given function within the parameter bounds.
        - Effectively a wrapper for scipy.optimize.minimize.
    Inputs:
        - func: Objective function to minimize.
        - bounds: Parameter space bounds.
        - method: Solver; defaults to L-BFGS-B.
            - All scipy options are available that do not explicitly require a Jacobian.
        - x0: Initial guess.
        - constraints: Constraints, in scipy's input shape.
        - kwargs: Keyword arguments for scipy's minimize function.
    Outputs:
        - solution: Best solution found.
        - fitness: Objective function value.
    '''
    
    # import
    from scipy import optimize

    # convert to array
    bounds = np.array(bounds)
    
    # default initial guess to the median of bounds
    if init is None:
        init = np.mean(bounds, axis=1)

    # minimize via scipy
    results = optimize.minimize(func, x0=init, 
                                method=method, bounds=bounds, 
                                constraints=constraints, tol=tolerance,
                                **kwargs)
    solution, fitness = results.x, results.fun

    # if optimization fails
    if verbose:
        print('Results:')

        # print best fitness
        print(f'- f(x): {fitness:.2e}')

        # print best solution
        if len(solution)>3:
            formatted_display = ', '.join([f'{val:.2e}' for val in solution[:3]])
            print(f'- Solution: [{formatted_display}, ...]')
        else:
            formatted_display = ', '.join([f'{val:.2e}' for val in solution])
            print(f'- Solution: [{formatted_display}]')

        # print number of iterations
        print(f'- n_iter: {results.nfev}')

        # if minimization does not converge
        if not results.success:
            import warnings
            warnings.warn(f'Optimization failed: {results.message}')

    return solution, fitness

# QUASAR polishing
def polish_solution(
            func=None, best_solution=None, best_fitness=None, bounds=None, popsize=None, maxiter=None, 
            vectorized=None, constraints=None, args=None, kwargs=None, 
            polish_minimizer=None, verbose=None):
    try:
        from scipy import optimize
        # handles a single 1D vector input (x) and returns a scalar fitness value.
        def polish_obj_func(x):
            '''Wrapper function to handle vectorized and non-vectorized inputs for SciPy minimize.'''
            
            if vectorized:
                # reshape for vectorized objective functions, take first element for (N,)
                return func(x.reshape(1,-1), *args, **kwargs)[0]
            else:
                # if not vectorized, x is already in the correct shape
                return func(x, *args, **kwargs)

        # constraints
        scipy_constraints = []
        if constraints:
            # loop through constraints
            for con_name, con_spec in constraints.items():
                con_func = con_spec[0]
                con_type = con_spec[1]
                con_rhs = con_spec[2]

                # vectorized with constraints case
                if vectorized:
                    try: 
                        from .quasar_helpers import scalar_constraint_wrapper
                    except:
                        from quasar_helpers import scalar_constraint_wrapper
                    
                    # apply wrapper function to handle 1D input from minimize
                    constraint_fun = lambda x: scalar_constraint_wrapper(x, con_func)
                else:
                    constraint_fun = con_func

                if con_type == '<=':
                    # inequality: g(x) <= 0
                    scipy_constraints.append({
                        'type': 'ineq', 
                        'fun': lambda x, rhs=con_rhs: rhs - constraint_fun(x),
                        })
                elif con_type == '>=':
                    # inequality: g(x) >= 0 
                    scipy_constraints.append({
                        'type': 'ineq', 
                        'fun': lambda x, rhs=con_rhs: constraint_fun(x) - rhs,
                        }),
                elif con_type == '==':
                    # equality: g(x) = C [g(x) - C = 0]
                    scipy_constraints.append({
                        'type': 'eq', 
                        # fun: lambda x: g(x) - C
                        'fun': lambda x, rhs=con_rhs: constraint_fun(x) - rhs,
                        })
                    
            # use trust-constr when constraints are present
            if polish_minimizer is None:
                polish_minimizer = 'SLSQP' 
        else:
            # otherwise default to Powell
            if polish_minimizer is None:
                polish_minimizer = 'L-BFGS-B'

        polish_iterations = int(np.minimum(500,np.sqrt(popsize*maxiter)))

        # minimize
        if verbose:
            print(f'Polishing solution with {polish_minimizer}.')
        polish_result = optimize.minimize(
                                polish_obj_func, best_solution, method = polish_minimizer,
                                bounds=bounds, options={'maxiter': polish_iterations},
                                constraints = scipy_constraints if constraints else()
        )
    
        # update best solution
        if polish_result.success and polish_result.fun < best_fitness:
            best_fitness_polished = polish_result.fun
            best_solution_polished = polish_result.x
        else:
            best_fitness_polished = best_fitness
            best_solution_polished = best_solution

    except Exception as e:
        print(f'Polishing failed: {e}')
        
        return best_solution, best_fitness
        
    return best_solution_polished, best_fitness_polished

def scalar_constraint_wrapper(x, func):
    '''
    Wrapper function to convert the 1D input x into a 2D array,
    for vectorized constraint / objective functions in polishing minimization step.
    '''
    
    # convert 1D input to 2D array
    x_2d = np.atleast_2d(x)
    
    # # call vectorized function
    result_2d = func(x_2d)
    
    if result_2d.size == 1:
        return result_2d.item() # ensure scalar is extracted
    else:
        # flatten for 1d array of scalars
        return result_2d.flatten()
    
    # extract scalar(s)
    return result_2d.flatten()

    
############## PLOTTING ##############

def plot_mutations(n_points=100000):
    '''
    Plots the distribution of mutation factors for all mutation strategies.
    '''

    # ensure integer n_points
    n_points = int(n_points)

    # import matplotlib
    import matplotlib.pyplot as plt

    # mutation plot params
    dimensions = 1 # for plotting
    peak_loc = 0.5
    initial_std_loc = 0.25
    local_std = 0.33
    
    loc_signs = np.random.choice([-1.0, 1.0], size=(n_points, 1), p=[0.5, 0.5])
    locs = loc_signs * peak_loc
    base_mutations = np.random.normal(loc=0.0, scale=initial_std_loc, size=(n_points, dimensions))
    global_muts = base_mutations + locs
    global_muts_flat = global_muts.flatten()
    
    local_muts = np.random.normal(loc=0.0, scale=local_std, size=(n_points, dimensions))
    local_muts_flat = local_muts.flatten()


    try: # in case seaborn is not imported
        import seaborn as sns
        sns.histplot(x=global_muts_flat, bins=50, edgecolor='black',stat='density',kde=True,color='deepskyblue',alpha=0.85,label='Global')
        sns.histplot(x=local_muts_flat, bins=50, edgecolor='black',stat='density',kde=True,color='darkorange',alpha=0.85,label='Local')
        plt.title('Mutation Factor Distribution',fontsize=16)
        plt.xlabel('Mutation Factor',fontsize=15)
        plt.ylabel('Frequency',fontsize=15)
        plt.legend(fontsize=15)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        plt.hist(global_muts_flat, bins=50, edgecolor='black',density=True,color='deepskyblue',alpha=0.85,label='Global')
        plt.hist(local_muts_flat, bins=50, edgecolor='black',density=True,color='darkorange',alpha=0.85,label='Local')
        plt.title('Mutation Factor Distribution',fontsize=16)
        plt.xlabel('Mutation Factor',fontsize=15)
        plt.ylabel('Frequency',fontsize=15)
        plt.legend(fontsize=15)
        
        plt.tight_layout()
        plt.show()


def plot_trajectories(obj_function, pop_history, best_history, bounds, num_to_plot, plot_contour, args, kwargs, vectorized):
    '''
    Plots the solution position history.
    '''
    
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    original_dims = bounds.shape[0]
    
    # convert to arrays
    plot_pop_history = np.array(pop_history)
    plot_best_history = np.array(best_history)

    if original_dims == 1:
        plt.figure(figsize=(7, 5.5))
        plt.plot(best_history, color='r', marker='x')
        plt.xlabel('Generation')
        plt.ylabel('Parameter Value')
        plt.title('Best Solution Trajectory')
        plt.show()
        
        return None
    
    # ensure bounds array has more than 2 dimensions
    if original_dims > 2:
        if plot_pop_history.size > 0:
            pca = PCA(n_components=2)
            
            # reshape data to fit PCA
            all_data = plot_pop_history.reshape(-1, original_dims)
            
            # fit PCA on population history
            pca.fit(all_data)

            # reshape data
            num_generations = plot_pop_history.shape[0]
            popsize = plot_pop_history.shape[1]
            
            # transform and reshape data
            plot_pop_history = pca.transform(all_data).reshape(num_generations, popsize, 2)
            plot_best_history = pca.transform(plot_best_history)
            
            # adjust bounds
            combined_transformed_data = np.concatenate([plot_pop_history.reshape(-1, 2), plot_best_history], axis=0)
            
            # ensure combined data is not empty
            if combined_transformed_data.size > 0:
                min_vals = np.min(combined_transformed_data, axis=0)
                max_vals = np.max(combined_transformed_data, axis=0)
                x_min, x_max = min_vals[0], max_vals[0]
                y_min, y_max = min_vals[1], max_vals[1]
            else:
                # fallback if all data is empty
                x_min, x_max = -1, 1
                y_min, y_max = -1, 1

        else:
            # case where only best_history exists (num_to_plot = 0)
            if plot_best_history.shape[0] > 1:
                pca = PCA(n_components=2)
                pca.fit(plot_best_history) # fit PCA on best_history
                plot_best_history = pca.transform(plot_best_history) # transform best_history
                plot_pop_history = None
                
                # adjust bounds based on history
                min_vals = np.min(plot_best_history, axis=0)
                max_vals = np.max(plot_best_history, axis=0)
                x_min, x_max = min_vals[0], max_vals[0]
                y_min, y_max = min_vals[1], max_vals[1]
            else:
                 # if no history available
                plot_best_history = None
                plot_pop_history = None
                x_min, x_max = -1, 1
                y_min, y_max = -1, 1
    
    plt.figure(figsize=(7, 5.5))
    if original_dims == 2:
        plt.xlabel('Dimension 0')
        plt.ylabel('Dimension 1')
    else:
        plt.xlabel('Principal Component 0')
        plt.ylabel('Principal Component 1')
    plt.title('Solution Trajectories')
    
    # plot contour
    if original_dims == 2:
        x_min, x_max = bounds[0, 0], bounds[0, 1]
        y_min, y_max = bounds[1, 0], bounds[1, 1]
        
        if plot_contour:
            try:
                import time
                grid_res = 100
                
                # grid size check for non-vectorized functions
                if not vectorized:
                    # time a single function evaluation
                    t0 = time.time()
                    _ = obj_function(np.array([x_min, y_min]), *args, **kwargs)
                    eval_time = max(time.time() - t0, 1e-9) # prevent div by zero
                    
                    # if 10,000 evals takes longer than 2 seconds, scale it down
                    if eval_time * 10000 > 2.0:
                        # calculate maximum grid that finishes in 2 seconds
                        max_evals = int(2.0 / eval_time)
                        grid_res = int(np.sqrt(max_evals))
                        
                        # if 20x20 grid takes too long, skip contour
                        if grid_res < 20:
                            plot_contour = False

                # proceed if didn't abort
                if plot_contour:
                    # objective function contour plot
                    x = np.linspace(x_min, x_max, grid_res)
                    y = np.linspace(y_min, y_max, grid_res)
                    X, Y = np.meshgrid(x, y)
                    xy_coords = np.vstack([X.ravel(), Y.ravel()]).T
                    
                    if vectorized:
                        Z = obj_function(xy_coords, *args, **kwargs).reshape(X.shape)
                    else:
                        fitness_list = [obj_function(coords, *args, **kwargs) for coords in xy_coords]
                        Z = np.array(fitness_list).reshape(X.shape)
                    
                    # evaluate objective function over 2D grid, log scale in case large orders of magnitude 
                    Z = np.log10(Z + epsilon)
                    Z[~np.isfinite(Z)] = np.nanmax(Z[np.isfinite(Z)]) * 1.1 if np.any(np.isfinite(Z)) else 0
                    
                    # remove 5% worst outliers and clip for visualization
                    z_min_clip = np.percentile(Z.flatten(), 5)
                    z_max_clip = np.percentile(Z.flatten(), 95)
                    Z_clipped = np.clip(Z, z_min_clip, z_max_clip)
                    
                    plt.contourf(X, Y, Z_clipped, levels=50, cmap='viridis', alpha=0.5, zorder=0) 
                    plt.colorbar(label='Objective Value')
                    
            except Exception as e:
                print(f'Contour failed: {e}')

    # plot solutions
    if (plot_pop_history is not None) and (num_to_plot > 0):
        indices_to_plot = np.random.choice(plot_pop_history.shape[1], min(num_to_plot, plot_pop_history.shape[1]), replace=False)
        
        for i in indices_to_plot:
            x_coords = plot_pop_history[:, i, 0]
            y_coords = plot_pop_history[:, i, 1]
            plt.plot(x_coords, y_coords, linestyle='-', marker='o', markersize=3, alpha=0.67, zorder=1)

    # plot path of best solution
    if plot_best_history is not None:
        x_coords = plot_best_history[:, 0]
        y_coords = plot_best_history[:, 1]
        plt.plot(x_coords, y_coords, linestyle='-', marker='x', markersize=8, color='red', label='Best Solution Trajectory', 
                 alpha=0.85, zorder=2)
        plt.scatter(x_coords[0], y_coords[0], color='deepskyblue', marker='d', s=150, label='Initial Best Solution', alpha=0.9, zorder=5)
        plt.scatter(x_coords[-1], y_coords[-1], color='deepskyblue', marker='X', s=150, label='Final Best Solution', alpha=0.9, zorder=5)
    
    plt.legend(fontsize=9,markerscale=0.67)
    plt.tight_layout()
    plt.show()