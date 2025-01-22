from sklearn.manifold import MDS

def mds_optimization(D, n_components=3, max_iter=500, eps=1e-3, n_init=5):
    """
    Perform Multidimensional Scaling (MDS) to recover 3D coordinates from a distance matrix.

    Args:
        D (np.ndarray): (n, n) distance matrix.
        n_components (int): Number of dimensions for the output space.
        max_iter (int): Maximum number of iterations for the optimization.
        eps (float): Relative tolerance with respect to stress to declare convergence.
        n_init (int): Number of times the algorithm will be run with different initializations.

    Returns:
        np.ndarray: Recovered (n, n_components) coordinates.
    """
    mds = MDS(n_components=n_components, max_iter=max_iter, eps=eps, n_init=n_init, dissimilarity='precomputed', random_state=42)
    coordinates = mds.fit_transform(D)
    
    return coordinates