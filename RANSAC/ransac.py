import matplotlib.pyplot as plt
import numpy as np
import random
import math

PI = math.pi
LINE = 0
CIRCLE = 1
MODEL_TYPES = [LINE, CIRCLE]
ALLOWED_NOISE = 0.002


def noisy_line_data(n_pts, p_line_pts, model_noise):
    """
    :return:
        X - data points (2d np array)
        model - dict of gt model params: 'm', 'n'. s.t. y = mx + n
    """
    # --
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


def noisy_circle_data(n_pts, p_circle_pts, model_noise):
    """
    :return:
        X - data points (2d np array)
        model - dict of gt model params: 'm', 'n'. s.t. y = mx + n
    """
    # --
    n_circle_pts = int(n_pts * p_circle_pts)
    n_noise_pts = n_pts - n_circle_pts
    # --
    X = np.zeros((n_pts, 2), float)
    # random model:
    model = {'r': np.random.random(), 'cx': np.random.random(), 'cy': np.random.random()}
    # generate model points:
    X[:n_circle_pts] = np.array([[math.cos( 2 * PI / n_circle_pts * x) * model['r'] + model_noise * np.random.random(), \
                  math.sin( 2 * PI/ n_circle_pts * x) * model['r'] + model_noise * np.random.random()] \
                  for x in range(0, n_circle_pts)])

    # generate noise points:
    mn = np.min(X[:n_circle_pts]) - 10 * model_noise + ((-1)**(round(np.random.random()*100)%2)*50*np.random.random())*model_noise
    mx = np.max(X[:n_circle_pts]) + 10 * model_noise + ((-1)**(round(np.random.random()*100)%2)*50*np.random.random())*model_noise
    X[n_circle_pts:, :] = np.random.random((n_noise_pts, 2)) * (mx - mn) + mn
    np.random.shuffle(X)
    return X, model


def distance(pt_0, pt_1, pt_x):
    return np.linalg.norm(np.cross(pt_1 - pt_0, pt_0 - pt_x)) / np.linalg.norm(pt_1 - pt_0)


def fit_line_to_data(data, n_iter, max_noise, sufficient_ratio):

    best_ratio = 0
    best_params = [0,0]
    iteration = 0

    while iteration < n_iter and best_ratio < sufficient_ratio:
        # pick two random points
        indexes = random.sample(range(len(data)), 2)
        pt_0 = data[int(indexes[0])]
        pt_1 = data[int(indexes[1])]
        m = (pt_0[1] - pt_1[1]) / (pt_0[0] - pt_1[0])
        n = pt_0[1] - m * pt_0[0]

        inliners = []

        for pt in data:
            if distance(pt_0, pt_1, pt) < max_noise:
                inliners.append(pt)

        inliners_ratio = len(inliners) / len(data)
        if inliners_ratio > best_ratio:
            best_ratio = inliners_ratio
            best_params = [m, n]

        # calc_model_error(data, LINE, [m, n])
        iteration += 1


    print('Number of iterations: ' + str(iteration))
    print('Inliners ratio: ' + str(best_ratio))
    print("Model params: m = {}, n = {}".format(best_params[0], best_params[1]))
    return best_params


def fit_circle_to_three_points(b, c, d):
    temp = c[0]**2 + c[1]**2
    bc = (b[0]**2 + b[1]**2 - temp) / 2
    cd = (temp - d[0]**2 - d[1]**2) / 2
    det = (b[0] - c[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (b[1] - c[1])

    if abs(det) < 1.0e-10:
        return None

    # Center of circle
    cx = (bc * (c[1] - d[1]) - cd * (b[1] - c[1])) / det
    cy = ((b[0] - c[0]) * cd - (c[0] - d[0]) * bc) / det

    radius = ((cx - b[0])**2 + (cy - b[1])**2)**.5

    return [radius, [cx, cy]]


def fit_circle_to_data(data, n_iter, max_noise, sufficient_ratio):

    best_ratio = 0
    best_params = []
    iteration = 0

    while iteration < n_iter and best_ratio < sufficient_ratio:
        # pick three random points
        indexes = random.sample(range(len(data)), 3)
        pt_0, pt_1, pt_2 = data[indexes]

        radius, center = fit_circle_to_three_points(pt_0, pt_1, pt_2)

        inliners = []

        for pt in data:
            dist = np.linalg.norm(pt - center)
            if abs(dist - radius) < max_noise:
                inliners.append(pt)

        inliners_ratio = len(inliners) / len(data)
        if inliners_ratio > best_ratio:
            best_ratio = inliners_ratio
            best_params = [radius, center]

        # calc_model_error(data, LINE, [m, n])
        iteration += 1


    print('Number of iterations: ' + str(iteration))
    print('Inliners ratio: ' + str(best_ratio))
    print("Model params: radius = {}, center = {}".format(best_params[0], best_params[1]))
    return best_params


def fit_model_to_data(data, model_type, n_iter, max_noise, sufficient_ratio):
    assert(model_type in MODEL_TYPES)

    params = []
    if model_type == LINE:
        params = fit_line_to_data(data, n_iter, max_noise, sufficient_ratio)
    elif model_type == CIRCLE:
        params = fit_circle_to_data(data, n_iter, max_noise, sufficient_ratio)

    return params


if __name__ == '__main__':

    pi = math.pi

    MODEL_RATIO = 0.2
    X, model = noisy_line_data(500, MODEL_RATIO, ALLOWED_NOISE)
    print('Actual model: m = {}, n = {}'.format(model['m'], model['n']))
    params = fit_model_to_data(data = X, model_type = LINE, n_iter = 40,
                               max_noise=ALLOWED_NOISE*2, sufficient_ratio = MODEL_RATIO*0.8)
    min_x = np.min(X[:,0])
    min_y = min_x * params[0] + params[1]
    max_x = np.max(X[:,0])
    max_y = max_x * params[0] + params[1]
    plt.plot([min_x, max_x], [min_y, max_y], 'r')

    min_gt_y = min_x * model['m'] + model['n']
    max_gt_y = max_x * model['m'] + model['n']
    plt.plot([min_x, max_x], [min_gt_y, max_gt_y], 'g')

    plt.plot(X[:,0], X[:,1], 'bo')
    plt.show()

    X, model = noisy_circle_data(500, MODEL_RATIO, ALLOWED_NOISE)

    params = fit_model_to_data(data = X, model_type = CIRCLE, n_iter = 100,
                               max_noise=ALLOWED_NOISE*2, sufficient_ratio = MODEL_RATIO*0.8)

    print("Actual model: r = {}, cx = {}, cy = {}".format(model['r'], model['cx'], model['cy']))
    plt.plot(X[:,0], X[:,1], 'bo')
    circle = plt.Circle((params[1][0], params[1][1]), params[0], color='r', fill=False)
    ax = plt.gca()
    ax.add_artist(circle)

    plt.show()

