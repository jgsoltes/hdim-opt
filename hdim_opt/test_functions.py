import numpy as np

def shifted_function(func, shift_vector):
    '''
    Objective:
        - Shifts the global optimum of the given test function.
        - Not compatible with workers > 1.
    
    Inputs:
        - func: Original test function.
        - shift_vector: 1D array of the new optimum.
    
    Outputs:
        - shifted_func: New function with the shifted optimum.
    '''
    
    def shifted_func(x):
        return func(x - shift_vector)
    return shifted_func

def rastrigin(x):
    '''
    Rastrigin test function, for local testing.
    '''
    
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

def ackley(x):
    '''
    Ackley test function, for local testing.
    '''
    
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

def sinusoid(x):
    '''
    Sinusoidal test function, for local testing.
    '''
    
    # check if x is a matrix (2D) or a single vector (1D)
    matrix_flag = x.ndim > 1
    if matrix_flag:
        sinusoid_val = np.sum(np.sin(x), axis=1)
    else:
        sinusoid_val = np.sum(np.sin(x))
    
    return sinusoid_val

def sphere(x):
    '''
    Sphere test function, for local testing.
    '''
    
    # check if x is a matrix (2D) or a single vector (1D)
    matrix_flag = x.ndim > 1
    if matrix_flag:
        sphere_val = np.sum(x**2, axis=1)
    else:
        sphere_val = np.sum(x**2)
    
    return sphere_val

def shubert(x):
    '''
    Shubert test function, for local testing.
    '''

    # check if x is matrix or single vector
    matrix_flag = x.ndim > 1
    
    j_values = np.arange(1, 6) # [1, 2, 3, 4, 5]
    x_reshaped = np.expand_dims(x, axis=-1)
    arg = (j_values + 1) * x_reshaped + j_values
    term = j_values * np.cos(arg)
    g_x = np.sum(term, axis=-1)

    if matrix_flag:
        shubert_value = np.prod(g_x, axis=1)
    else:
        shubert_value = np.prod(g_x)
    
    return shubert_value