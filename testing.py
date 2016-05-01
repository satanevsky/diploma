from scipy.stats import norm
import numpy as np
from sklearn.cross_validation import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score
from data_keeper import get_data_keeper


CONFUSION_MATRIX = 'confusion_matrix'
ACCURACY = 'accuracy'
FEATURES = 'features'
RAW_PREDICTIONS = 'raw_predictions'
OBJECTS = 'objects'
TRUE_VALUES = "true_values"

ALL_METRICS = [CONFUSION_MATRIX,
               ACCURACY,
               FEATURES,
               RAW_PREDICTIONS,
               OBJECTS,
               TRUE_VALUES]


def test_model_with_drug(model, drug, metrics, as_indexes):
    print metrics
    X, y = get_data_keeper().get_train_data(drug, as_indexes=as_indexes)
    y_pred = cross_val_predict(model, X, y, cv=StratifiedKFold(y, n_folds=10, shuffle=True, random_state=42))
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
    return result


def test_models_with_drugs(models, drugs, metrics=ALL_METRICS, as_indexes=False):
    result = dict()
    for model_name, model in models:
        for drug_name in drugs:
            result[(model_name, drug_name)] = test_model_with_drug(model, drug_name, metrics, as_indexes)
    return result


def results_differ_p_value(y_true, y1, y2):
    y1 = (np.array(y1) == np.array(y_true)).astype(np.float64)
    y2 = (np.array(y2) == np.array(y_true)).astype(np.float64)
    diff = y1 - y2
    norm_stat = diff.mean() / diff.std() * np.sqrt(diff.shape[0])
    quantile = norm.cdf(norm_stat)
    return min(quantile, 1.0 - quantile)
