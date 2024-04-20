from gplearn.fitness import _Fitness
from joblib import wrap_non_picklable_objects
from scipy import stats
import numpy as np

def make_fitness(*, function, greater_is_better, wrap=True):
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.' % function.__code__.co_argcount)
    if wrap:
        return _Fitness(function=wrap_non_picklable_objects(function),
                        greater_is_better=greater_is_better)
    return _Fitness(function=function,
                    greater_is_better=greater_is_better)


def compute_average_kendall(y, y_pred, w=None):
    y_0 = np.array([_y.split('+')[0] for _y in y]).astype(float)
    y_1 = np.array([_y.split('+')[1] for _y in y])
    unique_label = np.unique(y_1)
    corr_scores = []
    for label in unique_label:
        idx = np.argwhere(y_1 == label).reshape(-1)
        _y_pred = y_pred[idx]
        _y = y_0[idx]
        cor = stats.kendalltau(_y_pred, _y, nan_policy='omit')[0]
        if np.isnan(cor):
            cor = -1.0
        corr_scores.append(np.round(cor, 6))
    return corr_scores
    # mean_score = np.round(np.mean(corr_scores), 6)
    # return mean_score

def compute_kendall(y, y_pred, w=None):
    cor = stats.kendalltau(y, y_pred, nan_policy='omit')[0]
    if np.isnan(cor):
        cor = -1.0
    return cor

fitness_function_single = make_fitness(function=compute_kendall, greater_is_better=True)
fitness_function_multiple = make_fitness(function=compute_average_kendall, greater_is_better=True)
