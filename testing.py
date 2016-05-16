from itertools import izip
from copy import deepcopy
from scipy.stats import norm, chi2_contingency
import numpy as np
from sklearn.cross_validation import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score, f1_score
from data_keeper import get_data_keeper
from common import RANDOM_STATE


CONFUSION_MATRIX = 'confusion_matrix'
ACCURACY = 'accuracy'
FEATURES = 'features'
RAW_PREDICTIONS = 'raw_predictions'
OBJECTS = 'objects'
TRUE_VALUES = "true_values"
F1 = 'f1_score'

ALL_METRICS = [
    CONFUSION_MATRIX,
    ACCURACY,
    FEATURES,
    RAW_PREDICTIONS,
    OBJECTS,
    TRUE_VALUES,
    F1,
]


def test_model_with_drug(model, drug, metrics, as_indexes, n_folds=10):
    X, y = get_data_keeper().get_train_data(drug, as_indexes=as_indexes)
    return get_testing_metrics(model, X, y, metrics, as_indexes, n_folds)


def get_testing_metrics(model, X, y, metrics, as_indexes, n_folds):
    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=StratifiedKFold(
            y,
            n_folds=n_folds,
            shuffle=True,
            random_state=RANDOM_STATE
        )
    )
    print "y_pred", y_pred
    result = dict()
    if TRUE_VALUES in metrics:
        result[TRUE_VALUES] = y
    if RAW_PREDICTIONS in metrics:
        result[RAW_PREDICTIONS] = y_pred
    if CONFUSION_MATRIX in metrics:
        result[CONFUSION_MATRIX] = confusion_matrix(y, y_pred)
    if ACCURACY in metrics:
        result[ACCURACY] = accuracy_score(y, y_pred)
    if FEATURES in metrics:
        result[FEATURES] = model.fit(X, y).get_support(indices=True)
    if OBJECTS in metrics:
        if as_indexes:
            result[OBJECTS] = [get_data_keeper().get_object_name_by_index(index) for (index,) in X]
        else:
            result[OBJECTS] = list(X.index)
    if F1 in metrics:
        result[F1] = f1_score(y, y_pred)
    return result


def test_models_with_drugs(models, drugs, metrics=ALL_METRICS, as_indexes=False):
    result = dict()
    for model_name, model in models:
        for drug_name in drugs:
            result[(model_name, drug_name)] = test_model_with_drug(
                model,
                drug_name,
                metrics,
                as_indexes,
            )
    return result


class MetricsGetter:
    def __init__(self, metrics, as_indexes, loss_func, n_folds):
        self._metrics = metrics
        self._as_indexes = as_indexes
        self._loss_func = loss_func
        self._n_folds = n_folds

    def set_folds_count(self, n_folds):
        self._n_folds = n_folds

    def __call__(self, model, X, y):
        model = deepcopy(model)
        metrics = get_testing_metrics(
            model,
            X,
            y,
            self._metrics,
            self._as_indexes,
            self._n_folds,
        )
        loss = self._loss_func(metrics)
        return metrics, loss


def results_differ_p_value(y_true, y1, y2):
    y1 = (np.array(y1) == np.array(y_true)).astype(np.float64)
    y2 = (np.array(y2) == np.array(y_true)).astype(np.float64)
    diff = y1 - y2
    norm_stat = diff.mean() / diff.std() * np.sqrt(diff.shape[0])
    quantile = norm.cdf(norm_stat)
    return min(quantile, 1.0 - quantile)


def test_features_combimation(combination, X, y):
    combination_feature = and_arrays(X[:,combination].T)
    matr = np.zeros((2, 2), dtype=np.int32)
    for y_true, y_pred in izip(y, combination_feature):
        matr[y_true, y_pred] += 1
    return chi2_contingency(matr)[1]
