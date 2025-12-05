# global imports
import numpy as np
from scipy import stats
epsilon = 1e-12 # small epsilon to prevent zero-point errors

def initialize_population(popsize, bounds, init, hds_weights, seed, verbose):
    '''
    Objective:
        - Initializes the population using Sobol, Hyperellipsoid Density, Latin Hypercube, Random, or a custom population.
    Inputs:
        - popsize: Population size to generate.
        - bounds: Parameter space bounds.
        - init: Sampling method; ['sobol','hds','lhs','random'], or a custom array.
    Outputs:
        - initial_population: Initial population to optimize.
    '''

    # misc extracts
    n_dimensions = bounds.shape[0]
    
    # if input is not a string assume it is the initial population
    if isinstance(init, str):
        init = init.lower() # ensure lowercase string
        
        # generate hyperellipsoid density sequence
        if init == 'hds':
            # import hds
            try:
                from . import hyperellipsoid_sampling as hds
            except ImportError:
                import hyperellipsoid_sampling as hds
    
            # generate samples
            if verbose:
                print(f'Initializing: Hyperellipsoid population (N={popsize}, D={n_dimensions}).')
            initial_population = hds.sample(popsize, bounds, weights=hds_weights, 
                                            seed=seed, verbose=False)
    
        # generate sobol sequence
        elif init == 'sobol':
            if verbose:
                print(f'Initializing: Sobol population (N={popsize}, D={n_dimensions}).')
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning) # ignore power-of-2 warning
            sobol_sampler = stats.qmc.Sobol(d=n_dimensions, seed=seed)
            sobol_samples_unit = sobol_sampler.random(n=popsize)
            initial_population = stats.qmc.scale(sobol_samples_unit, bounds[:, 0], bounds[:, 1])
    
        elif (init == 'lhs') or (init == 'latinhypercube'):
            if verbose:
                print(f'Initializing: Latin Hypercube population (N={popsize}, D={n_dimensions}).')
            lhs_sampler = stats.qmc.LatinHypercube(d=n_dimensions, seed=seed)
            lhs_samples_unit = lhs_sampler.random(n=popsize)
            initial_population = stats.qmc.scale(lhs_samples_unit, bounds[:, 0], bounds[:, 1])
    
        elif init == 'random':
            if verbose:
                print(f'Initializing: Random population (N={popsize}, D={n_dimensions}).')
            initial_population = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(popsize, n_dimensions))
    else:
        initial_population = init
        if verbose:
            custom_popsize, custom_n_dimensions = init.shape
            print(f'Initializing: Custom population (N={custom_popsize}, D={custom_n_dimensions}).')

    return initial_population

def evolve_generation(obj_function, population, fitnesses, best_solution, 
                      bounds, entangle_rate, generation, maxiter, 
                      vectorized, n_workers, constraints, constraint_penalty, *args, **kwargs):
    '''
    Objective:
        - Evolves the population for the current generation.
            - Rank-based crossover rate:
                - Worst solution CR = 1.0, best solution CR = 0.33.
            - Local & global mutation factor distributions.
                - Can be displayed using 'plot_mutations()' function.
            - Greedy selection:
                - New solution vector is chosen as the better between of trial and current vectors.
            - Covariance reinitialization is handled externally.
            - Constraints are applied as additional objective penalties.
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
        
        # calculate trial fitnesses
        trial_fitnesses = obj_function(trial_vectors.T, *args, **kwargs)

        # penalize constraints
        if constraints:
            try:
                from .quasar_helpers import apply_penalty 
            except ImportError:
                from quasar_helpers import apply_penalty
            trial_fitnesses = apply_penalty(
                                        trial_fitnesses, 
                                        trial_vectors, 
                                        constraints,
                                        constraint_penalty,
                                        vectorized
                                        )
            
        # greedy elitism selection
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
        trial_vectors = []
        for i in range(popsize):
            solution = population[i]
            current_fitness = fitnesses[i]

            # select random 'entangled' partner indices
            random_index = np.random.randint(0, popsize)
            entangled_partner = population[random_index]

            # apply mutations
            if np.random.rand() < entangle_rate:
                mutant_vector = best_solution + local_mutation * (solution - entangled_partner)
            else:
                # 50% chance of moving around current location
                mutant_vector = solution + global_mutation * (best_solution - entangled_partner)
                # 50% chance of moving to entangled partner
                if np.random.rand() < 0.5:
                    mutant_vector = entangled_partner + global_mutation * (solution - entangled_partner)

            # calculate index values to crossover/recombine
            crossover_indices = np.random.rand(dimensions) < adaptive_crossover_proba[i]
            trial_vector = np.where(crossover_indices, mutant_vector, solution)

            # clip trial vectors to bounds
            trial_vector = np.clip(trial_vector, bounds[:, 0], bounds[:, 1])
            trial_vectors.append(trial_vector)

        # create array of trial vectors
        trial_vectors = np.array(trial_vectors)
        if n_workers > 1:
            try:
                import functools
                from concurrent.futures import ProcessPoolExecutor
            except:
                raise ImportError('Failed to import parallelization packages: functools, concurrent.futures.ProcessPoolExecutor.')
            
            func_with_args = functools.partial(obj_function, *args, **kwargs)
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # map() distributes the trial vectors to workers
                trial_fitnesses = np.array(list(executor.map(func_with_args, trial_vectors)))
        else:
            # standard sequential calculations 
            trial_fitnesses = np.array([obj_function(sol, *args, **kwargs) for sol in trial_vectors])

        # penalize constraints
        if constraints:
            try:
                from .quasar_helpers import apply_penalty 
            except ImportError:
                from quasar_helpers import apply_penalty
            trial_fitnesses = apply_penalty(trial_fitnesses, trial_vectors, constraints, constraint_penalty, vectorized)
        
        # greedy selection (now vectorized)
        selection_indices = trial_fitnesses < fitnesses

        # if selection index: trial_fitnesses[i], else: fitnesses[i]
        new_fitnesses = np.where(selection_indices, trial_fitnesses, fitnesses)

        # apply selection indices row-wise to population
        new_population = np.where(selection_indices.reshape(-1,1), trial_vectors, population)
        
    return new_population, new_fitnesses

def asym_reinit(population, current_fitnesses, bounds, reinit_method, seed, vectorized):
    '''
    Objective:
        - Reinitializes the worst 33% solutions in the population.
        - Locations are determined based on either:
            - 'covariance' (default): Gaussian distribution from the covariance of 25% best solutions (exploitation).
            - 'sobol': Uniformly Sobol distributed within the bounds (exploration).
    '''

    # reshape depending on vectorized input
    if vectorized:
        dimensions, popsize = population.shape
    else:
        popsize, dimensions = population.shape

    # handle case where not enough points for reliable covariance matrix
    if popsize < dimensions + 1:
        return population
    
    # identify solutions to be reset
    reset_population = 0.33
    num_to_replace = int(popsize * reset_population)
    if num_to_replace == 0:
        return population
        
    sorted_indices = np.argsort(current_fitnesses)
    worst_indices = sorted_indices[-num_to_replace:]
    
    # initializing new solutions
    new_solutions = None
    
    # covariance reinitalization; exploitation
    if reinit_method == 'covariance':
        
        # keep 25% of best solutions
        num_to_keep_factor = 0.25
        num_to_keep = int(popsize * num_to_keep_factor)
        if num_to_keep <= dimensions:
            num_to_keep = dimensions + 1 # minimum sample size scaled by dimensions
        
        # identify best solutions to calculate covariance gaussian model
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

        # new solutions sampled from multivariate normal distribution
        new_solutions_sampled = np.random.multivariate_normal(mean=mean_vector, cov=cov_matrix, size=num_to_replace)
        
        # add noise for exploration
        noise_scale = (bounds[:, 1] - bounds[:, 0]) / 20.0

        # reshape
        if vectorized:
            new_solutions_sampled = new_solutions_sampled.T 
            noise = np.random.normal(0, noise_scale[:, np.newaxis], size=new_solutions_sampled.shape)
            new_solutions = new_solutions_sampled + noise
        else:
            noise = np.random.normal(0, noise_scale, size=new_solutions_sampled.shape)
            new_solutions = new_solutions_sampled + noise

    # sobol reinitialization (high exploration)
    elif reinit_method == 'sobol':
        
        # generate sobol samples
        sobol_sampler = stats.qmc.Sobol(d=dimensions, seed=seed) 
        sobol_samples_unit = sobol_sampler.random(n=num_to_replace)
        
        bounds_low = bounds[:, 0]
        bounds_high = bounds[:, 1]
        scaled_samples = stats.qmc.scale(sobol_samples_unit, bounds_low, bounds_high) 

        # reshape
        if vectorized:
            new_solutions = scaled_samples.T
        else:
            new_solutions = scaled_samples
        

    # update the selected worst indices population
    if new_solutions is not None:
        if vectorized:
            population[:, worst_indices] = np.clip(new_solutions, 
                                                   bounds[:, np.newaxis, 0], 
                                                   bounds[:, np.newaxis, 1])
        else:
            population[worst_indices] = np.clip(new_solutions, bounds[:, 0], bounds[:, 1])

    return population
    

# main optimize function

def optimize(func, bounds, args=(),
              init='sobol', popsize=None, maxiter=100,
              entangle_rate=0.33, polish=True, polish_minimizer=None,
              patience=np.inf, vectorized=False,
              hds_weights=None, kwargs={},
              constraints=None, constraint_penalty=1e9,
              reinitialization=True, reinitialization_method='covariance',
              verbose=True, plot_solutions=True, num_to_plot=10, plot_contour=True,
              workers=1, seed=None
              ):
    '''
    Objective:
        - Finds the optimal solution for a given objective function.
        - Designed for non-differentiable, high-dimensional problems.
        - Test functions available for local testing, called as hdim_opt.test_functions.function_name.
            - Existing test functions: [rastrigin, ackley, sinusoid, sphere, shubert].
            
    Inputs:
        - func: Objective function to minimize.
        - bounds: Parameter bounds.
        - args: Tuple/list of positional arguments for the objective function.
        - kwargs: Dictionary of keyword arguments for the objective function.

        - init: Initial population sampling method. 
            - Defaults to 'sobol'. (Recommended power-of-2 population sizes for maximum uniformity).
            - Existing options are:
                - 'sobol': Sobol (highly uniform QMC; powers of 2 population sizes recommended).
                - 'hds': Hyperellipsoid Density (non-uniform; density weights 'hds_weights' recommended). 
                - 'lhs': Latin Hypercube (uniform QMC).
                - 'random': Random sampling (uniform).
                -  custom population (N x D matrix).
        - popsize: Number of solution vectors to evolve (default 10 * n_dimensions).
            - Recommended to be a power of 2 for Sobol initialization.
        - maxiter: Number of generations to evolve (default 100).
            
        - entangle_rate: Probability of solutions using the local Spooky-Best mutation strategy.
            - Defaults to 0.33. This causes to the three mutation strategies to be applied equally. 
            - Higher implies more exploitation.
        - polish: Boolean to implement final polishing step, using SciPy.optimize.minimize.
        - polish_minimizer: Minimization function to use for polishing step (using SciPy naming conventions).
            - Defaults to 'Powell' minimization, or 'SLSQP' if 'constraints' parameter is provided.
            - Recommended to place constraints in objective function to use Powell.
        
        - patience: Number of generations without improvement before early convergence.
        - vectorized: Boolean to accept vectorized (N,D) objective functions 
            - Extremely efficient, highly recommended whenever possible.
        
        - hds_weights: Optional weights for hyperellipsoid density sampling initialization.
            - {0 : {'center' : center, 'std': stdev}, 1: {...} }
        - kwargs: Dictionary of keyword arguments for the objective function.

        - constraints: Dictionary of constraints to penalize.
            - If possible, it is highly recommended to implement constraints as 
                high penalties into user's objective function instead. The same logic is used here, but
                
            - Example for x[1] - x[0]**2 <= 100:
                def test_constraint(x):
                    return x[1] - x[0]**2  # non-vectorized, or
                    return x[:,1] - x[:,0]**2  # vectorized
                custom_constraints = {
                                    'heat_capacity': (test_constraint, '<=', 100) 
                                        }
        - constraint_penalty: Penalty applied to each constraint violated, defaults to 1e12.

        - reinitialization: Boolean to disable covariance reinitialization if needed.
            - For cases where the population size is computationally prohibitive.
            - Disabled by default for 1D problems.
        - reinitialization_method: Type of re-sampling to use in the asymptotic reinitialization.
            - Options are ['covariance', 'sobol'].
            - 'covariance' (exploitative) is default for most problems.
            - 'sobol' (explorative) is optional, for high exploration and faster computation.
        
        - verbose: Displays prints and plots.
            - Mutation factor distribution shown with hdim_opt.test_functions.plot_mutations()
        - num_to_plot: Number of solutions to display in the verbose plot.
        - plot_contour: Display 2D contour, for 2D problems only.
        
        - workers: Number of workers / jobs / cores to use.
            - Default is 1. Set to -1 to use all available.
            - If workers != 1, constraint & objective functions must be imported from external module, for pickling.
        - seed: Random seed for deterministic & reproducible results.
        
    Outputs:
        - (best_solution, best_fitness) tuple:
            - best_solution: Best solution found.
            - best_fitness: Fitness of the optimal solution.
    '''

    ################################# INITIALIZE PARAMETERS #################################
    
    # set random seed
    import numpy as np
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
    
    # lowercase initialization string
    if type(init) == str:
        init = init.lower()
    
    # identify number of workers to use if input is -1
    if workers == -1:
        try:
            import os
        except:
            raise ImportError('Failed to import parallelization package: os.')
            
        # use all available CPU cores
        n_workers = os.cpu_count()
        
        # leave one core free for the main process while ensuring at least 1 active worker
        n_workers = max(1, n_workers - 1)
    else:
        # use specified number of workers
        n_workers = int(workers)
    
    # initialize histories
    pop_history = []
    best_history = []

    # ensure bounds is array; shape (n_dimensions,2)
    bounds = np.array(bounds)
    n_dimensions = bounds.shape[0]

    if n_dimensions == 1:
        reinitialization = False
        
    # if init is not a string, assume it is a custom population
    if not isinstance(init, str):
        popsize = init.shape[0]

    # default popsize to 10*n_dimensions
    if popsize == None:
        popsize = min(2**7,10*n_dimensions)
        
    # ensure integers
    popsize, maxiter = int(popsize), int(maxiter)

    
    ################################# INPUT ERRORS #################################
    
    # entangle rate error
    if not 0.0 <= entangle_rate <= 1.0:
        raise ValueError('Entanglement rate must be between [0,1].')

    # initialization error
    if (type(init) == str) and init not in ['sobol','hds','random','lhs','latinhypercube']:
        raise ValueError("Initial sampler must be one of ['sobol','random','hds','lhs'], or a custom population.")
    
    # patience error
    if patience <= 1:
        raise ValueError('Patience must be > 1 generation.')

    
    ################################# INITIAL POPULATION #################################
    
    # generate initial population
    initial_population = initialize_population(popsize, bounds, init, hds_weights, seed, verbose)
    if verbose:
        print('\nEvolving population:')

    # match differential evolution conventions    
    if vectorized:
        initial_population = initial_population.T
        initial_fitnesses = func(initial_population.T, *args, **kwargs)
    
    # non-vectorized parallel execution
    elif (n_workers > 1) and not vectorized:
        try:
            import functools
            from concurrent.futures import ProcessPoolExecutor
        except:
            raise ImportError("Failed to import parallelization packages: 'functools', 'concurrent.futures.ProcessPoolExecutor'.")
            
        if verbose:
            print(f'Parallel execution with {n_workers} workers.')
        
        func_with_args = functools.partial(func, *args, **kwargs) # functools.partial to fix the *args for objective function
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # map() distributes the population elements to workers
            initial_fitnesses = np.array(list(executor.map(func_with_args, initial_population)))

    # non-vectorized
    else:
        initial_fitnesses = np.array([func(sol, *args, **kwargs) for sol in initial_population])

    # apply constraint penalty to initial fitnesses
    if constraints:
        try:
            from .quasar_helpers import apply_penalty 
        except ImportError:
            from quasar_helpers import apply_penalty
        initial_fitnesses = apply_penalty(initial_fitnesses, initial_population, constraints, 
                                          constraint_penalty, vectorized)
    
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

    # add initial population to population history for plotting
    if plot_solutions:
        # determine which solutions to sample
        if popsize <= num_to_plot:
            # if popsize is small, take the whole population
            indices_to_sample = np.arange(popsize)
        else:
            # otherwise, randomly sample
            indices_to_sample = np.random.choice(popsize, num_to_plot, replace=False)
    
        if vectorized:
            sampled_population = population[:, indices_to_sample].T.copy()
        else:
            sampled_population = population[indices_to_sample].copy()
        pop_history.append(sampled_population)
        best_history.append(initial_best_solution.copy())
    
    # initialize best solution and fitness
    best_solution = initial_best_solution
    best_fitness = initial_best_fitness

    
    ################################# EVOLVE GENERATIONS #################################
    
    last_improvement_gen = 0
    for generation in range(maxiter):
        # evolve population
        if vectorized:
            population, current_fitnesses = evolve_generation(func, population, current_fitnesses, best_solution, bounds, 
                                                             entangle_rate, generation, maxiter, vectorized, 
                                                              n_workers, constraints, constraint_penalty, *args, **kwargs)
        else:
            population, current_fitnesses = evolve_generation(func, population, current_fitnesses, best_solution, bounds, 
                                                             entangle_rate, generation, maxiter, vectorized, 
                                                              n_workers, constraints, constraint_penalty, *args, **kwargs)
        
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
        if reinitialization:
            reinit_proba = np.e**((np.log(final_proba)/(decay_generation*maxiter))*generation)
        else:
            reinit_proba = 0.0
        if np.random.rand() < reinit_proba:
            population = asym_reinit(population, current_fitnesses, bounds, reinitialization_method, seed, vectorized=vectorized)

        # clip population to bounds
        if vectorized:
            population = np.clip(population, bounds[:, np.newaxis, 0], bounds[:, np.newaxis, 1])
        else:
            population = np.clip(population, bounds[:, 0], bounds[:, 1])

        # add to population history
        if verbose:
          # determine which solutions to sample
          if popsize <= num_to_plot:
              indices_to_sample = np.arange(popsize)
          else:
              indices_to_sample = np.random.choice(popsize, num_to_plot, replace=False)
        
          if vectorized:
              sampled_population = population[:, indices_to_sample].T.copy()
          else:
              sampled_population = population[indices_to_sample].copy()
              
          pop_history.append(sampled_population)
          best_history.append(best_solution.copy())

        # print generation status
        if verbose:
            stdev = np.std(current_fitnesses)
            print(f' Gen. {generation+1}/{maxiter} | f(x)={best_fitness:.2e} | stdev={stdev:.2e} | reinit={reinit_proba:.2f}')

        # patience for early convergence
        if (generation - last_improvement_gen) > patience:
            if verbose:
                print(f'\nEarly convergence: number of generations without improvement exceeds patience ({patience}).')
            break

    
    ################################# POLISH #################################
    
    # polish final solution
    if polish:
        try:
            from .quasar_helpers import polish_solution
        except:
            from quasar_helpers import polish_solution
        best_solution, best_fitness = polish_solution(
                                        func=func, best_solution=best_solution, best_fitness=best_fitness, 
                                        bounds=bounds, popsize=popsize, 
                                        maxiter=maxiter, vectorized=vectorized, constraints=constraints, 
                                        args=args, kwargs=kwargs, 
                                        polish_minimizer=polish_minimizer, verbose=verbose
    )

    
    ################################# VERBOSE #################################
    
    # final solution prints
    if verbose:
        print('\nResults:')

        # print best fitness
        print(f'- f(x): {best_fitness:.2e}')
        
        # print best solution
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

    if plot_solutions and verbose:
        print()
        try:
            try:
                from .quasar_helpers import plot_trajectories
            except ImportError:
                from quasar_helpers import plot_trajectories
            plot_trajectories(func, pop_history, best_history, bounds, num_to_plot, plot_contour, args, kwargs, vectorized)

        except Exception as e:
            print(f'Failed to generate plots: {e}')
    
    return (best_solution, best_fitness)