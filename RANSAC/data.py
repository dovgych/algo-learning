def noisy_line_data():
    """
    :return:
        X - data points (2d np array)
        model - dict of gt model params: 'm', 'n'. s.t. y = mx + n
    """
    import numpy as np
    # --
    n_pts = 500 # total number of points
    p_line_pts = 0.2 # portion of model-originated points
    model_noise = 0.001 # noise of model-originated points
    n_line_pts = int(n_pts * p_line_pts)
    n_noise_pts = n_pts - n_line_pts
    # --
    X = np.zeros((n_pts, 2), float)
    # random model:
    model = {'n': np.random.random(), 'm': np.random.random()}
    # generate model points:
    X[:n_line_pts, 0] = np.random.random(n_line_pts)
    X[:n_line_pts, 1] = model['m'] * X[:n_line_pts, 0] + model['n']
    X[:n_line_pts, 1] += model_noise * np.random.randn(n_line_pts)
    # generate noise points:
    mn = np.min(X[:n_line_pts]) - 10 * model_noise
    mx = np.max(X[:n_line_pts]) + 10 * model_noise
    X[n_line_pts:, :] = np.random.random((n_noise_pts, 2)) * (mx - mn) - mn
    np.random.shuffle(X)
    return X, model