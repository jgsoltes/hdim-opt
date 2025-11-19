def sens_analysis(func, bounds, n_samples=2**10, param_names=None, verbose=True, log_scale=True):
    try:
        import numpy as np
        from SALib.sample import sobol as sobol_sample
        from SALib.analyze import sobol as sobol_analyze
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            f"Sensitivity analysis requires dependencies: (SALib, pandas)."
        ) from e
    
    # define input parameters and their ranges
    bounds = np.array(bounds)
    num_params = bounds.shape[0]
    if param_names == None:
        param_names = range(0,num_params)

    # define problem
    problem = {
        'num_vars': num_params,
        'names': param_names,
        'bounds' : bounds
        }

    # generate samples
    if verbose:
        print(f'Generating {n_samples:,.0f} Sobol samples for sensitivity analysis.')
    param_values = sobol_sample.sample(problem, n_samples)

    # evaluate the samples
    values = func(param_values) 
    
    # running sensitivity analysis
    print('Running sensitivity analysis.')
    Si = sobol_analyze.analyze(problem, values, calc_second_order=True, print_to_console=False)

    if verbose:
        # import
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError as e:
            raise ImportError(
                f"Plotting requires dependencies: (matplotlib, seaborn)."
                ) from e
        
        # visualization parameters
        with plt.style.context('dark_background'):
            
            # plot 1: first-order (S1) and total-order (ST) indices
            sens_plot, axs = plt.subplots(2,1,figsize=(9, 13)) 
            
            # define bar width and positions
            bar_width = 0.35
            index = np.arange(num_params)
            
            # plot S1 (first order) sensitivities
            axs[0].barh(index - bar_width/2, Si['S1'], bar_width,
                   xerr=Si['S1_conf'], 
                   label='First-order ($S_1$)',
                   color='cornflowerblue',
                    alpha=1,
                   ecolor='lightgray',
                   capsize=2.5,
                   edgecolor='black')
            
            # plot ST (total order) sensitivities
            axs[0].barh(index + bar_width/2, Si['ST'], bar_width,
                   xerr=Si['ST_conf'], 
                   label='Total-order ($S_T$)',
                   color='violet', 
                   ecolor='white',
                   alpha=0.75, 
                   capsize=2.5,
                   edgecolor='lightgray')
            axs[0].set_title('Sensitivity Indices ($S_1$, $S_T$)')
            if log_scale:
                axs[0].set_xscale('log')
            axs[0].set_xlabel('Sensitivity Index')
            axs[0].set_ylabel('Parameter')
            axs[0].legend(loc='upper right')
            axs[0].grid(False)
            axs[0].set_yticks(index)
            axs[0].set_yticklabels(param_names, ha='right')
            
            # convert S2 indices to dataframe to process easier
            S2_matrix = Si['S2']
            S2_df = pd.DataFrame(S2_matrix, index=param_names, columns=param_names)
            S2_df = S2_df.fillna(S2_df.T)
            mask = np.tril(np.ones_like(S2_df, dtype=bool))
            
            # heatmap of second order indices
            sns.heatmap(data=S2_df, mask=mask, cmap='berlin', cbar_kws={'label': 'Second-order Index ($S_2$)'},ax=axs[1]) # magma
            axs[1].set_title('Second-order Interactions ($S_2$)')
            axs[1].invert_yaxis()
            sens_plot.tight_layout() 
            plt.show()
    
    return Si, S2_matrix