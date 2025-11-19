# global epslion
epsilon = 1e-12

### misc helper functions 

def sample_hypersphere(n_dimensions, radius, n_samples_in_sphere, radius_qmc_sequence):
    '''
    Objective:
        - Samples unit hyperspheres using Marsaglia polar vectors scaled by a QMC sequence.
    '''
    import numpy as np
    
    # generate normal distribution (for angular direction)
    samples = np.random.normal(size=(n_samples_in_sphere, n_dimensions))
    
    # normalize vectors to get points on the surface of a unit sphere (direction)
    squared_norms = np.sum(samples**2, axis=1)
    inv_norms = 1.0 / (np.sqrt(squared_norms) + epsilon)
    
    # efficiently apply directions (broadcasting faster than division)
    samples = samples * inv_norms[:, np.newaxis]
    
    # use input QMC sequence for radius scaling (r^(1/n))
    random_radii_qmc = np.power(radius_qmc_sequence, (1.0 / n_dimensions))
    
    # scale and apply final radius
    samples = samples * (random_radii_qmc * radius)[:, np.newaxis]
    
    return samples

def sample_hyperellipsoid(n_dimensions, n_samples_in_ellipsoid, origin, pca_components, pca_variances, scaling_factor, radius_qmc_sequence=None):
    '''
    Objective:
        - Generates samples inside the hyperellipsoid.
            - Calls the function to sample unit hyperspheres.
            - Transforms the hyperspherical samples to the ellipsoid axes defined using the PCA variances.
    '''
    import numpy as np
    
    # generate samples in unit hypersphere
    unit_sphere_samples = sample_hypersphere(n_dimensions, 1.0, n_samples_in_ellipsoid, radius_qmc_sequence)
    
    # axis lengths: L = sqrt(variance + epsilon) * scaling_factor
    axis_lengths = np.sqrt(pca_variances + epsilon) * scaling_factor
    
    # scale samples in PCA space and rotates back to original parameter space
    scaled_and_rotated = (unit_sphere_samples * axis_lengths) @ pca_components
    
    # translate samples to cluster origin
    ellipsoid_samples = scaled_and_rotated + origin
    
    return ellipsoid_samples

def sample_in_voids(existing_samples, n_to_fill, bounds_min, bounds_max,
                    k_neighbors=5, spread_factor=0.5,
                    n_query_max=1000, n_tree_max=10000): 
    '''
    Objective:
        - Identify & fill voids in the sample space, using the out-of-bounds sample set.
        - Uses BallTree K-NearestNeighbors to identify voids.
    '''
    from sklearn.neighbors import BallTree
    from sklearn.random_projection import GaussianRandomProjection
    import numpy as np
    from scipy import stats
    import time
    
    # extract shape
    n_existing, n_dimensions = existing_samples.shape

    # if no samples to replace
    if n_to_fill <= 0:
        return np.zeros((0, n_dimensions))
        
    # if number of neighbors exceeds number of existing samples
    if n_existing < k_neighbors + 1:
        return stats.uniform.rvs(loc=bounds_min, scale=bounds_max - bounds_min, size=(n_to_fill, n_dimensions))

    # for number of points > 100,000, reduce size to 10,000 for speed
    if n_existing > n_tree_max:
        tree_indices = np.random.choice(n_existing, size=n_tree_max, replace=False)
        tree_samples = existing_samples[tree_indices]
    else:
        tree_indices = np.arange(n_existing)
        tree_samples = existing_samples
        
    # recalculate n_existing for reduced set
    n_existing_for_tree = tree_samples.shape[0]

    # reduce dimensionality for speed
    n_rp_components = min(max(10, int(2*np.log2(n_dimensions))), n_dimensions)
    if n_rp_components < n_dimensions:
        rp = GaussianRandomProjection(n_components=n_rp_components)
        existing_samples_reduced = rp.fit_transform(tree_samples)
    else:
        existing_samples_reduced = tree_samples
    
    # build BallTree
    start_kdtree_build = time.time()
    tree = BallTree(existing_samples_reduced) 
    
    # BallTree query on subset
    k_to_query = k_neighbors + 1 # k-th neighbor distance

    # select random subset of centers to calculate void probability for (query set)
    # query set sampled from the reduced set (existing_samples_reduced)
    if n_existing_for_tree > n_query_max:
        query_subset_indices = np.random.choice(n_existing_for_tree, size=n_query_max, replace=False)
        existing_samples_query = existing_samples_reduced[query_subset_indices]
    else:
        query_subset_indices = np.arange(n_existing_for_tree)
        existing_samples_query = existing_samples_reduced
    
    # query the subset against the reduced-sample tree
    # returns distances first, then indices.
    kth_nn_distances_subset, _ = tree.query(existing_samples_query, k=k_to_query,)
    
    # extract distance to the k-th nearest neighbor
    kth_nn_distances_subset = kth_nn_distances_subset[:, k_neighbors]
    
    # identify void centers
    probabilities = kth_nn_distances_subset / (kth_nn_distances_subset.sum() + epsilon)
    void_center_query_indices = np.random.choice(
        len(query_subset_indices),
        size=n_to_fill,
        p=probabilities,
        replace=True
        )
    
    # map back to indices in the tree_samples set
    void_center_tree_indices = query_subset_indices[void_center_query_indices]

    # map the indices back to the original full-sized existing_samples array
    void_center_full_indices = tree_indices[void_center_tree_indices]

    # extract centers and spreads from original, full-D data
    chosen_centers = existing_samples[void_center_full_indices]
    # kth_nn_distances_subset is still correct as it corresponds to the query points
    chosen_kth_distances = kth_nn_distances_subset[void_center_query_indices] 
    
    # calculate spreads and generate samples (in full-D)
    spreads = chosen_kth_distances[:, np.newaxis] * spread_factor + epsilon
    a_params = (bounds_min - chosen_centers) / spreads
    b_params = (bounds_max - chosen_centers) / spreads
    new_samples = stats.truncnorm.rvs(a=a_params, b=b_params, loc=chosen_centers, scale=spreads)
    
    return new_samples

def fit_pca_for_cluster(cluster_samples, current_origin, initial_samples_std, n_dimensions):
    '''
    Performs PCA on a single cluster's samples or returns a default, 
    called in parallel.
    '''

    from sklearn.decomposition import PCA
    import numpy as np
    
    # extract shape
    n_cluster_samples = len(cluster_samples)
    
    if n_cluster_samples > n_dimensions * 2 and n_cluster_samples > 0:
        pca = PCA(n_components=n_dimensions)
        pca.fit(cluster_samples)
        
        return {'origin': current_origin, 
                'components': pca.components_.T, 
                'variances': pca.explained_variance_}
    else:
        # handle empty/too small clusters
        fixed_variance = np.ones(n_dimensions) * initial_samples_std

        return {'origin': current_origin, 
                'components': np.eye(n_dimensions), 
                'variances': fixed_variance}


# main sample function

def sample(n_samples, bounds,
           weights=None, normalize=False,
           n_ellipsoids=None, n_initial_clusters=None, n_initial_qmc=None,
           seed=None, plot_dendrogram=False, verbose=False):
    '''
    Objective:
        - Generates a Hyperellipsoid Density sample sequence over the specified parameter range.
    Inputs:
        - n_samples: Number of samples to generate.
        - bounds: Bounds of the parameter range.
        - weights: Gaussian weights to influence clusters and final sample locations.
        - normalize: Boolean to scale samples to the original parameter space, or leave normalized from [0,1].
        - n_ellipsoids: Number of hyperellipsoids to sample.
            - Replaces and skips the Agglomerative Hierarchical Clustering (AHC) step.
        - n_initial_clusters: Number of initial clusters to use in calculating number of hyperellipsoids.
            - Redunant if n_ellipsoids is specified.
        - n_initial_qmc: Number of initial QMC samples to use for cluster analysis.
        - seed: Random seed.
        - plot_dendrogram: Boolean to display dendrogram used for ellipsoid determination.
        - verbose: Boolean to display stats and plots.
    Outputs:
        - hds_samples: Hyperellipsoid Density sample sequence.
    '''

    # imports
    try:
        import numpy as np
        import pandas as pd
        from scipy import stats
        from joblib import Parallel, delayed
        from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
        from sklearn.random_projection import GaussianRandomProjection
        from sklearn.decomposition import PCA
        import scipy.cluster.hierarchy as shc
        import time
        import warnings
    except ImportError as e:
        raise ImportError(
            f"HDS requires additional dependencies: (pandas, sklearn), if verbose: (matplotlib)."
        ) from e
        
    warnings.filterwarnings('ignore', category=UserWarning)

    # initialize misc parameters:
    start_time = time.time()
    if seed is None:
        seed = time.time()
    seed = int(round(seed))
    np.random.seed(seed)

    # initialize sampling parameters:
    n_samples = int(n_samples)
    bounds = np.array(bounds)
    n_dimensions = bounds.shape[0]

    # n_initial_clusters scaling for D > 100
    if n_initial_clusters is None:
        if n_dimensions <= 500:
            n_initial_clusters = 100 # 100 clusters for <= 500D
        elif n_dimensions < 1000:
            n_initial_clusters = 50 # 50 clusters sees same stddev as 100 for > 500D
        else:
            n_initial_clusters = 25 # 25 clusters sees same stddev as 100 for > 1000D
    n_initial_clusters = int(n_initial_clusters)
    
    # number of qmc samples scaling
    n_initial_qmc_max = 2**15
    if n_initial_qmc is None:
        min_qmc_dimensions = int(2**np.ceil(np.log2(n_dimensions*200)))
        n_initial_qmc = min(min_qmc_dimensions, n_initial_qmc_max)
    
    # keep data normalized (0 to 1) for clustering
    bounds_min_orig = bounds[:, 0]
    bounds_max_orig = bounds[:, 1]
    working_bounds_min = np.zeros(n_dimensions)
    working_bounds_max = np.ones(n_dimensions)
    
    # generate initial QMC samples:
    qmc_start_time = time.time()
    initial_sobol_sampler = stats.qmc.Sobol(d=n_dimensions, seed=np.random.randint(0, 1000))
    initial_samples_unit = initial_sobol_sampler.random(n=n_initial_qmc)
    initial_samples = initial_samples_unit
    
    # calculate sample weights based on input
    sample_weights = np.ones(initial_samples.shape[0])
    if weights:
        initial_samples_denorm = stats.qmc.scale(initial_samples, bounds_min_orig, bounds_max_orig)
        for dim, info in weights.items():
            center = info['center']
            std = info['std']
            if not std > 0:
                raise ValueError(f'Gaussian weight stddevs must be > 0.')
                return None
            dim_values = initial_samples_denorm[:, dim]
            gaussian_weights = stats.norm.pdf(dim_values, loc=center, scale=std)
            
            gaussian_weights += epsilon
            sample_weights *= gaussian_weights

    if verbose and n_ellipsoids is None:
        print('Calculating ellipsoid density.')
        
    # determine number of ellipsoids via agglomerative clustering
    # KMeans to get stable sub-cluster centers
    kmeans_initial = MiniBatchKMeans(n_clusters=n_initial_clusters, random_state=seed, n_init=6)
    kmeans_initial.fit(initial_samples, sample_weight=sample_weights)
    initial_centroids = kmeans_initial.cluster_centers_
    
    # skip agglomerative clustering if number of initial clusters is provided
    linkage_matrix = None
    optimal_distance = None
    if n_ellipsoids is not None and n_ellipsoids >= 1:
        n_hyperellipsoids = n_ellipsoids
    else:
        # hierarchical clustering on the centroids to find natural grouping
        linkage_matrix = shc.linkage(initial_centroids, method='ward')
        
        # find optimal cut-off distance (d) based on largest jump
        distances = linkage_matrix[:, 2]
        optimal_distance = 0
        if len(distances) > 2:
            diffs = distances[1:] - distances[:-1]
            max_diff_index = np.argmax(diffs)
            optimal_distance = distances[max_diff_index + 1] 
            
            # in case initial sample clusters are uniform
            if optimal_distance < 0.1: 
                optimal_distance = 0.5 
                
            # agglomerative clustering to determine k at the optimal distance
            agg_model = AgglomerativeClustering(n_clusters=None, distance_threshold=optimal_distance, linkage='ward')
            agg_model.fit(initial_centroids)
            n_hyperellipsoids = agg_model.n_clusters_
            
        else:
            n_hyperellipsoids = 1
    n_hyperellipsoids = max(1, n_hyperellipsoids)
    
    # K-Means again with optimal n_hyperellipsoids to find final centers
    kmeans = MiniBatchKMeans(n_clusters=n_hyperellipsoids, random_state=seed, n_init=6)
    kmeans.fit(initial_samples, sample_weight=sample_weights)
    origins = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    
    # pre-calculate sample stddev
    initial_samples_std = np.std(initial_samples)
    
    # prepare inputs for the parallel loop
    cluster_data_inputs = []
    for i in range(n_hyperellipsoids):
        cluster_samples = initial_samples[cluster_labels == i]
        current_origin = origins[i]
        
        # recenter origin if the cluster is empty
        if len(cluster_samples) == 0:
            current_origin = initial_samples[np.random.randint(len(initial_samples))]
        
        cluster_data_inputs.append((cluster_samples, current_origin, initial_samples_std, n_dimensions))

    # calculate hyperellipsoid shapes via PCA (parallelized)
    if verbose:
        print(f'Orienting axes.')
    ellipsoid_params = Parallel(n_jobs=-1)(
        delayed(fit_pca_for_cluster)(*args) for args in cluster_data_inputs
        )
    
    # recalculate cluster counts from labels, for proportional sampling
    cluster_sample_counts = np.array([np.sum(cluster_labels == i) for i in range(n_hyperellipsoids)])

    # distribute samples proportionally to cluster size
    total_cluster_count = np.sum(cluster_sample_counts)
    if total_cluster_count == 0:
        n_samples_per_ellipsoid = np.ones(n_hyperellipsoids, dtype=int)
    else:
        n_samples_per_ellipsoid = np.round(n_samples * (cluster_sample_counts / total_cluster_count)).astype(int)
    n_samples_per_ellipsoid[-1] += n_samples - np.sum(n_samples_per_ellipsoid)
    n_samples_per_ellipsoid = np.maximum(0, n_samples_per_ellipsoid)
    
    # generate hyperellipsoid samples (sobol distributed radius):
    hds_samples_normalized = np.zeros((0, n_dimensions))

    # radius scaling factor; scales with dimension
    confidence_level = 0.9999 # captures 99.99% of cluster's samples

    # critical value (the statistical radius squared)
    chi2_critical_value = stats.chi2.ppf(confidence_level, df=n_dimensions)
    baseline_factor = 0.55 - 0.01*np.log(n_dimensions) # empirically derived to resample out-of-bounds points
    
    # square root as the scaling factor (Mahalanobis distance)
    ellipsoid_scaling_factor = baseline_factor * np.sqrt(chi2_critical_value)

    # QMC sequence for radius scaling
    radius_qmc_sampler = stats.qmc.Sobol(d=1, seed=seed+1) # offset seed from initial qmc
    radius_qmc_sequence_base = radius_qmc_sampler.random(n=int(n_samples * 2.5)) # generate extra samples
    radius_start_idx = 0
    
    # sequentially generate samples from each ellipsoid
    collected_samples = []
    for i, params in enumerate(ellipsoid_params):
        n_to_generate = n_samples_per_ellipsoid[i] * 2
        
        # select next chunk of QMC radius sequence
        radius_end_idx = radius_start_idx + n_to_generate
        
        # prevent index out of bounds
        if radius_end_idx > len(radius_qmc_sequence_base):
            # use the remainder
            radius_qmc_chunk = radius_qmc_sequence_base[radius_start_idx:].flatten()
        else:
            radius_qmc_chunk = radius_qmc_sequence_base[radius_start_idx:radius_end_idx].flatten()
        
        radius_start_idx = radius_end_idx

        # prevent ValueError from empty array
        if n_to_generate > 0 and radius_qmc_chunk.size == 0:
            continue # skip ellipsoid if this chunk is empty

        # generate samples inside current ellipsoid
        ellipsoid_samples = sample_hyperellipsoid(n_dimensions, 
                                                n_to_generate, 
                                                params['origin'], 
                                                params['components'],
                                                params['variances'],
                                                scaling_factor=ellipsoid_scaling_factor,
                                                radius_qmc_sequence=radius_qmc_chunk
                                                )
        
        # identify points outside boundaries ([0,1] hypercube)
        in_bounds_mask = np.all(ellipsoid_samples >= 0, axis=1) & np.all(ellipsoid_samples <= 1, axis=1)
        valid_samples = ellipsoid_samples[in_bounds_mask]
        
        # add required number of valid samples
        num_to_add = min(n_samples_per_ellipsoid[i], len(valid_samples))
        collected_samples.append(valid_samples[:num_to_add])
        
    # vstack
    if collected_samples:
        hds_samples_normalized = np.vstack(collected_samples)
    else:
        hds_samples_normalized = np.zeros((0, n_dimensions))

    # identify number of points to resample
    n_to_fill = n_samples - len(hds_samples_normalized) 
    if n_to_fill > 0:
        if verbose:
            print(f'Geometric void filling {n_to_fill} samples.')

        # use the existing collected hds samples to find the voids
        k_void_neighbors = min(max(5, int(n_dimensions / 10)), 10)
        void_resamples = sample_in_voids(
            existing_samples=hds_samples_normalized, 
            n_to_fill=n_to_fill, 
            bounds_min=working_bounds_min,
            bounds_max=working_bounds_max,
            k_neighbors=k_void_neighbors, # k scales with dimension
            spread_factor=0.25 # stay local
            )
        
        # combine original hds samples with new void samples
        hds_samples_normalized = np.vstack([hds_samples_normalized, void_resamples])
    
    hds_samples_normalized = hds_samples_normalized[:n_samples]
    
    if normalize:
        # leave in [0,1] space
        hds_sequence = hds_samples_normalized
    else:
        # scale samples to original bounds
        hds_sequence = hds_samples_normalized * (bounds_max_orig - bounds_min_orig) + bounds_min_orig
    
    ### print & plot results:
    if verbose:
        end_time = time.time()
        sample_generation_time = end_time - start_time
        
        # visualization imports
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.patches import Circle, Rectangle
            sns.set_style('dark')
        except ImportError as e:
            raise ImportError(
                f'Plotting requires dependencies: (matplotlib, seaborn).'
            ) from e
        
        # print results
        print('\nresults:')
        print('    - number of samples:', len(hds_sequence))
        print(f'    - sample generation time: {sample_generation_time:.2f}')
        print(f'    - number of hyperellipsoids: {n_hyperellipsoids}')
        print(f'    - number of initial QMC: {n_initial_qmc}')
        print(f'    - number of initial clusters: {n_initial_clusters}')
        if weights:
            print(f'    - gaussian weights: {weights}')
        
        # generate a sobol sequence for comparison
        sobol_sampler = stats.qmc.Sobol(d=n_dimensions, seed=seed+2) # offset seed to be different from initial qmc
        sobol_samples_unit = sobol_sampler.random(n=n_samples)
        if normalize:
            sobol_samples = sobol_samples_unit
        else:
            sobol_samples = stats.qmc.scale(sobol_samples_unit, bounds_min_orig, bounds_max_orig)

        # samples stats:
        hds_mean = np.mean(hds_sequence)
        sobol_mean = np.mean(sobol_samples)
        hds_std = np.std(hds_sequence)
        sobol_std = np.std(sobol_samples)

        print('\nstats:')
        print(f'    - mean HDS: {hds_mean:.2f}')
        print(f'    - mean comparison QMC: {sobol_mean:.2f}')
        print(f'    - stdev HDS: {hds_std:.2f}')
        print(f'    - stdev comparison QMC: {sobol_std:.2f}\n')

        # dendrogram of centroids
        if plot_dendrogram:
            if linkage_matrix is not None:
                plt.figure(figsize=(8, 6))
                plt.title(f'Dendrogram of Initial Centroids: {n_dimensions}D')
                
                # using pre-calculated linkage matrix
                shc.dendrogram(linkage_matrix, color_threshold=optimal_distance, above_threshold_color='gray')
                plt.axhline(y=optimal_distance, color='r', linestyle='--', label=f'Optimal Cutoff (k={n_hyperellipsoids})')
                plt.ylabel('Dissimilarity Distance')
                plt.xticks([])
                
                plt.legend(loc='upper right')
                plt.show()
        
        # plot for 1d samples
        if n_dimensions == 1:
            plt.figure(figsize=(6,5))
            plt.hist(hds_sequence, bins=30, color='deepskyblue', alpha=0.9, label='HDS Samples')
            plt.hist(sobol_samples, bins=30, color='red', alpha=0.5, label='Sobol Samples')
            plt.title('HDS Sample Distribution')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
            return hds_sequence

        
        ### PCA for n_dim > 2:
        
        if normalize:
            data_to_plot_raw = initial_samples
        else:
            data_to_plot_raw = stats.qmc.scale(initial_samples, bounds_min_orig, bounds_max_orig)

        if n_dimensions > 2:
            pca = PCA(n_components=2)
            data_to_plot = pca.fit_transform(data_to_plot_raw)
            hds_sequence_plot = pca.transform(hds_sequence)
            sobol_samples_plot = pca.transform(sobol_samples)
            origins_plot = pca.transform(origins)
            title_str = f'Parameter Space (PCA): n={n_samples}, D={n_dimensions}'
            xlabel_str = f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})'
            ylabel_str = f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})'
        else:
            data_to_plot = data_to_plot_raw
            hds_sequence_plot = hds_sequence
            sobol_samples_plot = sobol_samples
            origins_plot = origins
            title_str = f'Parameter Space: n={n_samples}, D={n_dimensions}'
            xlabel_str = f'Dimension 0'
            ylabel_str = f'Dimension 1'

        # dark visualization parameters for better sample visuals
        with plt.style.context('dark_background'):
            # samples
            fig, ax = plt.subplots(1,2,figsize=(11,5.5))
            
            ax[0].scatter(hds_sequence_plot[:, 0], hds_sequence_plot[:, 1], color='deepskyblue', s=0.67, zorder=5, label='HDS Samples')
            
            # data hypercube boundary
            min_plot = np.min(data_to_plot, axis=0)
            max_plot = np.max(data_to_plot, axis=0)
            width = max_plot[0] - min_plot[0]
            height = max_plot[1] - min_plot[1]
            hypercube_boundary = Rectangle((min_plot[0], min_plot[1]), width, height, fill=False, alpha=0.75, color='cornflowerblue', linewidth=1, 
                                           linestyle='--', zorder=6)
            
            ax[0].add_patch(hypercube_boundary)
            ax[0].set_title(title_str, fontweight='bold', fontsize=14)
            ax[0].set_xlabel(xlabel_str)
            ax[0].set_ylabel(ylabel_str)
            ax[0].axis(False)
            ax[0].legend(loc=(0.7,0.87), fontsize=8)
    
            # plot histograms
            ax[1].hist(hds_sequence.flatten(), bins=30, label='HDS Samples', edgecolor='black',color='deepskyblue', alpha=0.75)
            ax[1].set_title('HDS Distribution')
            ax[1].set_ylabel('')
            ax[1].set_yticks([])
            ax[1].set_xticks([])
            ax[1].set_xlabel('')
    
            plt.tight_layout()
            plt.show()

    return hds_sequence