# global imports
import numpy as np
from scipy import stats
epsilon = 1e-12 # small epsilon to prevent zero-point errors


### test functions, for local testing ###

def rastrigin(x, vectorized=False):
    '''Rastrigin test function, for local testing.'''
    
    A = 10 # rastrigin coefficient
    
    # check if x is a matrix (2D) or a single vector (1D)
    matrix_flag = x.ndim > 1
    if matrix_flag:
        # for x of (popsize, dimensions)
        n = x.shape[1]
        rastrigin_value = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)
    else:
        # for single solution vector x
        n = x.shape[0]
        rastrigin_value = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    return rastrigin_value

def ackley(x, vectorized=False):
    '''Ackley test function, for local testing.'''
    
    # check if x is a matrix (2D) or a single vector (1D)
    matrix_flag = x.ndim > 1
    if matrix_flag:
        # for x of (popsize, dimensions)
        n = x.shape[1]
        arg1 = -0.2 * np.sqrt(1/n * np.sum(x**2, axis=1))
        arg2 = 1/n * np.sum(np.cos(2 * np.pi * x), axis=1)
    else:
        # for single solution vector x
        n = x.shape[0]
        arg1 = -0.2 * np.sqrt(1/n * np.sum(x**2))
        arg2 = 1/n * np.sum(np.cos(2 * np.pi * x))
    
    ackley_val = -20 * np.exp(arg1) - np.exp(arg2) + 20 + np.exp(1)
    
    return ackley_val

def sphere(x, vectorized=False):
    '''Sphere test function, for local testing.'''
    
    # check if x is a matrix (2D) or a single vector (1D)
    matrix_flag = x.ndim > 1
    if matrix_flag:
        sphere_val = np.sum(x**2, axis=1)
    else:
        sphere_val = np.sum(x**2)
    
    return sphere_val

def shifted_function(func, shift_vector, vectorized=False):
    '''
    Objective:
        - Shifts the global optimum of the given test function.
    
    Inputs:
        - func: Original test function.
        - shift_vector: 1D array of the new optimum.
    
    Outputs:
        - shifted_func: New function with the shifted optimum.
    '''
    
    def shifted_func(x, vectorized=False):
        return func(x - shift_vector, vectorized=vectorized)
    return shifted_func


### verbose plotting functions ###

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

    with plt.style.context('dark_background'):
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


def plot_trajectories(obj_function, pop_history, best_history, bounds, num_to_plot):
    '''
    Plots the solution position history.
    '''
    
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # visualization params
    with plt.style.context('dark_background'):
    
        original_dims = bounds.shape[0]
        
        # convert to arrays
        plot_pop_history = np.array(pop_history)
        plot_best_history = np.array(best_history)
        
        # ensure bounds array has more than 2 dimensions
        if original_dims > 2:
            pca = PCA(n_components=2)
            
            # reshape data to fit PCA
            all_data = plot_pop_history.reshape(-1, original_dims)
            
            # fit PCA on population history
            pca.fit(all_data)
    
            # reshape data
            num_generations = plot_pop_history.shape[0]
            popsize = plot_pop_history.shape[1]
            
            # transform and reshape dadta
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
                x_min, x_max = -1, 1
                y_min, y_max = -1, 1
        
        else:
            x_min, x_max = bounds[0, 0], bounds[0, 1]
            y_min, y_max = bounds[1, 0], bounds[1, 1]
        
        plt.figure(figsize=(9, 7))
        if original_dims == 2:
            plt.xlabel('Dimension 0')
            plt.ylabel('Dimension 1')
        else:
            plt.xlabel('Principal Component 0')
            plt.ylabel('Principal Component 1')
        plt.title('Solution Trajectories')

        if original_dims == 2:
            # objective function contour plot
            x = np.linspace(x_min, x_max, 100)
            y = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x, y)
            xy_coords = np.vstack([X.ravel(), Y.ravel()]).T
            
            # evaluate objective function over 2D grid
            Z = obj_function(xy_coords).reshape(X.shape)
            Z = np.log10(Z + 1e-6)
            plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.5, zorder=0) 
            plt.colorbar(label='Objective Value')
        
        if plot_pop_history is not None:
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
            plt.scatter(x_coords[0], y_coords[0], color='cyan', marker='d', s=300, label='Initial Best Solution', alpha=0.8, zorder=5)
            plt.scatter(x_coords[-1], y_coords[-1], color='cyan', marker='X', s=350, label='Final Best Solution', alpha=0.8, zorder=5)
        
        plt.legend(fontsize=10,markerscale=0.67)
        plt.grid(False)
        plt.show()


### evolution algorithm ###

def initialize_population(popsize, bounds, init, hds_weights, seed, verbose):
    '''
    Objective:
        - Initializes a population using Sobol, Adaptive Hyperellipsoid, or a custom population.
    '''

    # misc extracts
    n_dimensions = bounds.shape[0]
    
    # if input is not a string assume it is the initial population
    if isinstance(init, str):
        init = init.lower() # ensure lowercase string
        
        # generate adaptive hypersphere sequence
        if init == 'hds':
            # import hds
            try:
                from . import hyperellipsoid_sampling as hds
            except ImportError:
                import hyperellipsoid_sampling as hds
    
            # generate samples
            if verbose:
                print('Initializing Hyperellipsoid vectors.')
            initial_population = hds.sample(popsize, bounds, weights=hds_weights, 
                                            seed=seed, verbose=False)
    
        # generate sobol sequence
        elif init == 'sobol':
            if verbose:
                print('Initializing Sobol vectors.')
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning) # ignore power-of-2 warning
            sobol_sampler = stats.qmc.Sobol(d=n_dimensions, seed=seed)
            sobol_samples_unit = sobol_sampler.random(n=popsize)
            initial_population = stats.qmc.scale(sobol_samples_unit, bounds[:, 0], bounds[:, 1])
    
        elif init == 'lhs':
            if verbose:
                print('Initializing Latin Hypercube vectors.')
            lhs_sampler = stats.qmc.LatinHypercube(d=n_dimensions, seed=seed)
            lhs_samples_unit = lhs_sampler.random(n=popsize)
            initial_population = stats.qmc.scale(lhs_samples_unit, bounds[:, 0], bounds[:, 1])
    
        elif init == 'random':
            if verbose:
                print('Initializing random vectors.')
            initial_population = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(popsize, n_dimensions))
    else:
        initial_population = init
        if verbose:
            print('Initializing custom population.')

    return initial_population

def evolve_generation(obj_function, population, fitnesses, best_solution, 
                      bounds, entangle_rate, generation, maxiter, 
                      vectorized, *args):
    '''
    Objective:
        - Evolves the population for the current generation.
            - Dynamic crossover rate:
                - Worst solution CR = 1.0, best solution CR = 0.33.
            - Local & global mutation factor distributions.
                - Can be displayed using the 'plot_mutations' function.
            - Greedy selection:
                - New solution vector is chosen as the better between of trial and current vectors.
            - Covariance reinitialization is handled externally.
    '''

    # crossover parameters
    base_crossover_proba = 1.0
    min_crossover_proba = 0.33
    
    if vectorized:
        dimensions, popsize = population.shape
        
        # select random entangled partners for all solutions
        random_indices = np.random.randint(0, popsize, size=popsize)
        entangled_partners = population[:, random_indices]

        # global mutation factor:
        global_std = 0.25
        global_peaks = 0.5
        is_positive_peak = np.random.choice([True, False], p=[0.5, 0.5])
        global_mutation = np.random.normal(loc=global_peaks, scale=global_std, 
                                            size=(dimensions,1)) if is_positive_peak else np.random.normal(loc=-global_peaks, 
                                            scale=global_std, size=(dimensions,1))

        # local mutation factor:
        local_mutation_std = 0.33
        local_mutation = np.random.normal(0.0, local_mutation_std, size=(dimensions,1))
        
        # best solution as a column vector (dimensions, 1)
        best_solution_broadcast = best_solution[:, np.newaxis]

        # identify solution indices to use Spooky-Best strategy
        local_indices = np.random.rand(1, popsize) < entangle_rate

        # global mutations
        mutant_vectors_current = population + global_mutation * (best_solution_broadcast - entangled_partners)
        mutant_vectors_random = entangled_partners + global_mutation * (population - entangled_partners)
        
        # 50% chance of using Spooky-Random, otherwise Spooky-Current
        entangled_random_indices = np.random.rand(1, popsize) < 0.5
        
        # select between the two global mutations
        global_mutants = np.where(entangled_random_indices, mutant_vectors_random, mutant_vectors_current)
        
        # select between local and global mutations
        mutant_vectors = np.where(local_indices, best_solution_broadcast + local_mutation * (population - entangled_partners), 
                                  global_mutants)
        
        # rank solutions by fitness
        sorted_indices = np.argsort(fitnesses)
        ranks = np.zeros_like(fitnesses)
        ranks[sorted_indices] = np.arange(popsize)
        max_rank = popsize - 1

        # calculate adaptive crossover rates
        relative_fitness = (max_rank - ranks) / max_rank
        adaptive_crossover_proba_raw = (1 - base_crossover_proba) + base_crossover_proba * relative_fitness
        adaptive_crossover_proba = np.maximum(adaptive_crossover_proba_raw, min_crossover_proba)

        # apply crossover to create trial vectors
        crossover_indices = np.random.rand(dimensions, popsize) < adaptive_crossover_proba
        trial_vectors = np.where(crossover_indices, mutant_vectors, population)

        # clip trial vectors to bounds
        trial_vectors = np.clip(trial_vectors, bounds[:, np.newaxis, 0], bounds[:, np.newaxis, 1])
        
        # steady-state elitism selection
        trial_fitnesses = obj_function(trial_vectors.T, *args)
        selection_indices = trial_fitnesses < fitnesses
        
        new_population = np.where(selection_indices[np.newaxis, :], trial_vectors, population)
        new_fitnesses = np.where(selection_indices, trial_fitnesses, fitnesses)
        
        return new_population, new_fitnesses
        
    else:
        # extract shape
        popsize, dimensions = population.shape

        # initialize arrays
        new_population = np.zeros_like(population)
        new_fitnesses = np.zeros_like(fitnesses)

        # global mutation factor
        global_std = 0.25
        global_peaks = 0.5
        is_positive_peak = np.random.choice([True, False], p=[0.5, 0.5])
        global_mutation = np.random.normal(loc=global_peaks, scale=global_std, 
                                           size=dimensions) if is_positive_peak else np.random.normal(loc=-global_peaks, 
                                                                                      scale=global_std, size=dimensions)
        
        # local mutation factor
        local_mutation_std = 0.33
        local_mutation = np.random.normal(0.0, local_mutation_std, size=dimensions)

        # adaptive crossover calculations
        sorted_indices = np.argsort(fitnesses)
        ranks = np.zeros_like(fitnesses)
        ranks[sorted_indices] = np.arange(popsize)
        max_rank = popsize - 1
        relative_fitness = (max_rank - ranks) / max_rank
        adaptive_crossover_proba_raw = (1 - base_crossover_proba) + base_crossover_proba * relative_fitness
        adaptive_crossover_proba = np.maximum(adaptive_crossover_proba_raw, min_crossover_proba)
        
        # loop through each solution in population
        for i in range(popsize):
            solution = population[i]
            current_fitness = fitnesses[i]

            # select random 'entangled' partner indices
            random_index = np.random.randint(0, popsize)
            entangled_partner = population[random_index]

            # apply mutations
            if np.random.rand() < entangle_rate:
                mutant_vector = best_solution + local_mutation * (solution - entangled_partner) # original
            else:
                # 50% chance of moving around current location
                mutant_vector = solution + global_mutation * (best_solution - entangled_partner) # original
                # 50% chance of moving to entangled partner
                if np.random.rand() < 0.5:
                    mutant_vector = entangled_partner + global_mutation * (solution - entangled_partner) # original
            
            crossover_indices = np.random.rand(dimensions) < adaptive_crossover_proba[i]
            trial_vector = np.where(crossover_indices, mutant_vector, solution)

            # clip trial vectors to bounds
            trial_vector = np.clip(trial_vector, bounds[:, 0], bounds[:, 1])

            # steady-state elitism selection
            trial_fitness = obj_function(trial_vector, *args)
            
            if trial_fitness < current_fitness:
                new_population[i] = trial_vector
                new_fitnesses[i] = trial_fitness
            else:
                new_population[i] = solution
                new_fitnesses[i] = current_fitness
        
        return new_population, new_fitnesses

def covariance_reinit(population, current_fitnesses, bounds, vectorized):
    '''
    Objective:
        - Reinitializes the worst 33% solutions in the population.
        - Locations are determined based on a Gaussian distribution from the covariance matrix of 25% best solutions.
            - Noise is added to enhance diversity.
        - Probability decays to 33% at the 33% generation.
        - Conceptualizes particles tunneling to a more stable location.
    '''

    # reshape depending on vectorized input
    if vectorized:
        dimensions, popsize = population.shape
    else:
        popsize, dimensions = population.shape

    # handle case where not enough points for covariance matrix
    if popsize < dimensions + 1:
        return population

    # keep 25% of best solutions
    num_to_keep_factor = 0.25
    num_to_keep = int(popsize * num_to_keep_factor)
    if num_to_keep <= dimensions:
        num_to_keep = dimensions + 1 # minimum sample size scaled by dimensions
    
    # identify best solutions to calculate covariance gaussian model
    sorted_indices = np.argsort(current_fitnesses)
    best_indices = sorted_indices[:num_to_keep]
    if vectorized:
        best_solutions = population[:, best_indices]
    else:
        best_solutions = population[best_indices]
    
    # learn full-covariance matrix
    if vectorized:
        mean_vector = np.mean(best_solutions, axis=1)
        cov_matrix = np.cov(best_solutions)
    else:
        mean_vector = np.mean(best_solutions, axis=0)
        cov_matrix = np.cov(best_solutions, rowvar=False)

    # add epsilon to the diagonal to prevent singular matrix issues
    cov_matrix += np.eye(dimensions) * epsilon

    # identify solutions to be reset
    reset_population = 0.33
    num_to_replace = int(popsize * reset_population)
    worst_indices = sorted_indices[-num_to_replace:]

    # new solutions sampled from multivariate normal distribution
    if vectorized:
        new_solutions_sampled = np.random.multivariate_normal(mean=mean_vector, cov=cov_matrix, size=num_to_replace).T
    else:
        new_solutions_sampled = np.random.multivariate_normal(mean=mean_vector, cov=cov_matrix, size=num_to_replace)
    
    # add noise for exploration
    noise_scale = (bounds[:, 1] - bounds[:, 0]) / 20.0
    if vectorized:
        noise = np.random.normal(0, noise_scale[:, np.newaxis], size=new_solutions_sampled.shape)
        new_solutions = new_solutions_sampled + noise
    else:
        noise = np.random.normal(0, noise_scale, size=new_solutions_sampled.shape)
        new_solutions = new_solutions_sampled + noise

    # clip new solutions to bounds
    if vectorized:
        population[:, worst_indices] = np.clip(new_solutions, bounds[:, np.newaxis, 0], bounds[:, np.newaxis, 1])
    else:
        population[worst_indices] = np.clip(new_solutions, bounds[:, 0], bounds[:, 1])

    return population

def optimize(func, bounds, args=(), 
              init='sobol', popsize=None, maxiter=100,
              entangle_rate=0.33, polish=True,
              patience=np.inf, vectorized=False, 
              verbose=True, num_to_plot=10,
              hds_weights=None,
              seed=None):
    '''
    Objective:
        - Searches for the optimal solution to minimize the objective function.
        - Conceptualizes quantum stellar particles searching for the most stable energy state.
        - Mutation factors can be visualized with plot_mutations().
    Inputs:
        - func: Objective function to minimize.
        - bounds: Parameter bounds.
        - *args: Input arguments for objective function.
        - init: Initial population sampling method. 
            - Options are 'sobol', 'random', 'hds', 'lhs', or a custom sample sequence.
                - Defaults to 'sobol'.
                - 'hds' to accelerate convergence with slower initialization.
        - popsize: Number of solution vectors to evolve. 
            - Defaults to 10 * n_dimensions.
        - maxiter: Number of generations to evolve.
        - entangle_rate: Probability of solutions using the local Spooky-Best mutation strategy.
            - Defaults to 0.33. This causes to the three mutation strategies to be applied equally.
            - Decreasing leads to higher exploration.
        - patience: Number of generations without improvement before early convergence.
        - polish: Final polishing step using 'L-BFGS-B' minimization.
        - vectorized: Boolean to accept vectorized objective functions.
        - verbose: Displays prints and plots.
        - num_to_plot: Number of solution trajectories to display in the verbose plot.
    Outputs:
        - (best_solution, best_fitness) tuple:
            - best_solution: Best solution found.
            - best_fitness: Fitness of the optimal solution.
    '''

    # set random seed
    if seed == None:
        np.random.seed()
    else:
        np.random.seed(seed)

    # handle case where time is not imported
    if verbose:
        try:
            import time
            start_time = time.time()
        except:
            pass

    # lowercase initialization
    if type(init) == str:
        init = init.lower()
        
    # raise errors for invalid inputs:
    # entangle rate error
    if not 0.0 <= entangle_rate <= 1.0:
        raise ValueError('Entanglement rate must be between [0,1].')

    # initialization error
    if (type(init) == str) and init not in ['sobol','hds','random','lhs']:
        raise ValueError("Initial sampler must be one of ['sobol','random','ahs','lhs'], or a custom population.")
    
    # patience error
    if patience <= 1:
        raise ValueError('Patience must be > 1 generation.')
    
    # initialize histories
    pop_history = []
    best_history = []

    # ensure bounds is array; shape (n_dimensions,2)
    bounds = np.array(bounds)
    n_dimensions = bounds.shape[0]
    
    # if init is not a string, assume it is a custom population
    if not isinstance(init, str):
        popsize = init.shape[0]
        
    # default popsize to 10*n_dimensions
    if popsize == None:
        popsize = 10*n_dimensions

    # ensure integers
    popsize, maxiter = int(popsize), int(maxiter)

    # generate initial population
    initial_population = initialize_population(popsize, bounds, init, hds_weights, seed, verbose)
    if verbose:
        print('\nEvolving population:')

    # match differential evolution conventions
    if vectorized:
        initial_population = initial_population.T
        initial_fitnesses = func(initial_population.T, *args)
    else:
        initial_fitnesses = np.array([func(sol, *args) for sol in initial_population])

    # calculate initial best fitness
    min_fitness_idx = np.argmin(initial_fitnesses)
    initial_best_fitness = initial_fitnesses[min_fitness_idx]

    # identify initial best solution
    if vectorized:
        initial_best_solution = initial_population[:, min_fitness_idx].copy()
    else:
        initial_best_solution = initial_population[min_fitness_idx].copy()

    # initialze population and fitnesses
    population = initial_population
    current_fitnesses = initial_fitnesses

    # add initial population to population history
    if verbose:
        if vectorized:
            pop_history.append(population.T.copy())
        else:
            pop_history.append(population.copy())

    # add initial best solution to best solution history
    best_history.append(initial_best_solution.copy())
    
    # initialize best solution and fitness
    best_solution = initial_best_solution
    best_fitness = initial_best_fitness
    
    # iterate through generations
    last_improvement_gen = 0
    for generation in range(maxiter):
        
        # evolve population
        if vectorized:
            population, current_fitnesses = evolve_generation(func, population, current_fitnesses, best_solution, bounds, 
                                                             entangle_rate, generation, maxiter, vectorized, *args)
        else:
            population, current_fitnesses = evolve_generation(func, population, current_fitnesses, best_solution, bounds, 
                                                             entangle_rate, generation, maxiter, vectorized, *args)
        
        # update best solution found
        min_fitness_idx = np.argmin(current_fitnesses)
        current_best_fitness = current_fitnesses[min_fitness_idx]

        # identify current best solution
        if vectorized:
            current_best_solution = population[:, min_fitness_idx].copy()
        else:
            current_best_solution = population[min_fitness_idx].copy()

        # update best known solution
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution
            last_improvement_gen = 0
        else:
            last_improvement_gen += 1

        # apply asymptotic covariance reinitialization to population
        final_proba = 0.33
        decay_generation = 0.33
        reinit_proba = np.e**((np.log(final_proba)/(decay_generation*maxiter))*generation)
        if np.random.rand() < reinit_proba:
            population = covariance_reinit(population, current_fitnesses, bounds, vectorized=vectorized)

        # clip population to bounds
        if vectorized:
            population = np.clip(population, bounds[:, np.newaxis, 0], bounds[:, np.newaxis, 1])
        else:
            population = np.clip(population, bounds[:, 0], bounds[:, 1])

        # add to population history
        if verbose:
            if vectorized:
                pop_history.append(population.T.copy())
            else:
                pop_history.append(population.copy())
            best_history.append(best_solution.copy())

        # print generation status
        if verbose:
            stdev = np.std(current_fitnesses)
            print(f' Gen. {generation+1}/{maxiter} | f(x)={best_fitness:.2e} | stdev={stdev:.2e} | reinit={reinit_proba:.2f}')

        # patience for early convergence
        if (generation - last_improvement_gen) > patience:
            if verbose:
                print(f'\nEarly convergence: generations without improvement exceeds patience ({patience}).')
            break

    # polish final solution step via L-BFGS-B
    if polish:
        if verbose:
            print('Polishing solution.')
        try:
            from scipy.optimize import minimize
            if vectorized:
                res = minimize(func, best_solution, args=args, bounds=bounds, method='L-BFGS-B', tol=0)
            else:
                res = minimize(func, best_solution, args=args, bounds=bounds, method='L-BFGS-B', tol=0)
            if res.success:
                best_solution = res.x
                best_fitness = res.fun
        except Exception as e:
            print(f'Polishing failed: {e}')

    # final solution prints
    if verbose:
        print('\nResults:')

        # print best fitness
        print(f'- f(x): {best_fitness:.2e}')
        
        #print best solution
        if len(best_solution)>3:
            formatted_display = ', '.join([f'{val:.2e}' for val in best_solution[:3]])
            print(f'- Solution: [{formatted_display}, ...]')
        else:
            formatted_display = ', '.join([f'{val:.2e}' for val in best_solution])
            print(f'- Solution: [{formatted_display}]')

        # print optimization time
        try:
            print(f'- Elapsed: {(time.time() - start_time):.3f}s')
        except Exception as e:
            print(f'- Elapsed: null') # case where time isn't imported
        
        # plotting
        print()
        try:
            plot_trajectories(func, pop_history, best_history, bounds, num_to_plot)

        except Exception as e:
            print(f'Failed to generate plots: {e}')
    
    return (best_solution, best_fitness)