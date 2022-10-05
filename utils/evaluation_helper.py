import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial import cKDTree
from scipy.stats import entropy

try:
    # available with 'pip install lapsolver'
    from lapsolver import solve_dense as linear_sum_assignment
except:
    print('failed to import lapsolver')
    from scipy.optimize import linear_sum_assignment


def distance(x, y):
    #return tf.metrics.mean_squared_error(x, y).numpy()
    return np.linalg.norm(x - y, axis=-1)


def optimal_assignment_distance(x, y):
    dist = distance_matrix(x, y)
    row_idx, col_idx = linear_sum_assignment(dist)
    return dist[row_idx, col_idx]


def chamfer_distance(pred, gt):
    tree = cKDTree(pred)
    dist, _ = tree.query(gt)
    return dist


def compute_stats(x):
    return {
        'mean': np.mean(x),
        'mse': np.mean(x**2),
        'var': np.var(x),
        'min': np.min(x),
        'max': np.max(x),
        'median': np.median(x),
        'num_particles': x.shape[0],
    }


def compare_dist(x, y, bin_size=25):
    assert x.shape == y.shape
    cnt = x.shape[0]
    dim = x.shape[-1]
    bin_cnt = cnt // bin_size
    bin_cnt_per_dim = int(bin_cnt**(1 / dim))

    min_v = np.percentile(np.concatenate((x, y), axis=0), 5, axis=0)
    max_v = np.percentile(np.concatenate((x, y), axis=0), 95, axis=0)

    bin_w = (max_v - min_v + 1e-6) / bin_cnt_per_dim

    #bins = np.zeros((bin_cnt_per_dim + 1, ) * dim)
    binsx = np.zeros((bin_cnt_per_dim + 1, ) * dim) + 1e-5
    binsy = np.zeros((bin_cnt_per_dim + 1, ) * dim) + 1e-5

    def to_idx(val):
        return tuple(
            np.clip(((val - min_v) / bin_w).astype("int32"), 0,
                    bin_cnt_per_dim))

    for v in x:
        binsx[to_idx(np.array(v))] += 1
        #bins[to_idx(np.array(v))] += 1
    for v in y:
        binsy[to_idx(np.array(v))] += 1
        #bins[to_idx(np.array(v))] -= 1

    return entropy(np.reshape(binsx, -1), np.reshape(binsy, -1))
    #return np.sum(np.abs(bins)) / cnt


def merge_dicts(dicts, op, start_val=0):
    output = {}
    for d in dicts:
        for k, v in d.items():
            if k not in output:
                output[k] = start_val
            output[k] = op(output[k], v)
    return output
