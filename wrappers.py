from copy import copy
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
from xgboost import XGBClassifier
from joblib import Memory
from boruta import BorutaPy
from common import forward_out
from generate_subsets import SubsetGenerator


mem_xgb = Memory(cachedir='cache/xgboost')


#@forward_out("logs/xgb.log")
@mem_xgb.cache
def fit_xgboost(params, X, y):
    clf = XGBClassifier(**params)
    clf.fit(X, y)
    return clf


class GridSearchCVWrapper(GridSearchCV):
    def get_support(self, *args, **kwargs):
        return self.best_estimator_.get_support(*args, **kwargs)


class ModelBasedFeatureImportanceGetter(BaseEstimator):
    def __init__(self, inner_model):
        self.inner_model = inner_model

    def get_feature_importances(self, X, y):
        return self.inner_model.fit(X, y).feature_importances_

    def get_support(self, *args, **kwargs):
        return self.inner_model.get_support(*args, **kwargs)


class XGBoostClassifierFeatureImportances(XGBClassifier):
    @property
    def feature_importances_(self):
        importances_dict = self.booster().get_fscore()
        print importances_dict
        result = np.array([importances_dict.get('f{}'.format(i), 0) for i in xrange(self._features_count)], dtype=np.float64)
        return result

    def fit(self, X, y):
        self._features_count = X.shape[1]
        if X.shape[1] == 0:
            X = np.zeros((X.shape[0], 1))
        super(XGBoostClassifierFeatureImportances, self).fit(X, y)
        self.__dict__.update(fit_xgboost(self.get_params(), X, y).__dict__)
        return self

    def get_support(self, indices=False):
        if indices:
            return np.arange(self._features_count)
        else:
            return np.ones(self._features_count, dtype=np.bool)

    def predict(self, X):
        if X.shape[1] == 0:
            X = np.zeros((X.shape[0], 1))
        return super(XGBoostClassifierFeatureImportances, self).predict(X)


class MatrixCleaningWrapper(BaseEstimator):
    def __init__(self, inner_model):
        self.inner_model = inner_model

    def _drop(self, X):
        X = X.drop(self._to_drop, axis=1, inplace=False)
        return X.as_matrix()

    def get_support(self, indices=False):
        if indices == False:
            raise KeyError("indices should be true")
        support = self.inner_model.get_support(indices=True)
        return np.array([self._features_invert_index[el] for el in support])

    def _set_dropped(self, X, to_drop):
        self._to_drop = to_drop
        self._features_invert_index = list()
        to_drop_set = set(to_drop)
        for el in X.columns.values:
            if el not in to_drop_set:
                self._features_invert_index.append(el)

    def fit(self, X, y):
        X = X.copy()
        X[X != 1] = 0

        ones_count = X.sum(axis=0)
        to_drop = ones_count[(ones_count <= 2) |
                  (ones_count >= (X.shape[0] / 3))].index
        self._set_dropped(X, to_drop)
        X = self._drop(X)
        print "cleaner fit", X.shape
        self.inner_model.fit(X, y)
        #self.feature_importances_ = self.inner_model.feature_importances_
        return self

    def predict(self, X):
        X = X.copy()
        X[X != 1] = 0
        X = self._drop(X)
        print "cleaner predict:", X.shape
        return self.inner_model.predict(X)

class SparseWrapper(BaseEstimator):
    def __init__(self, inner_model):
        self.inner_model = inner_model

    def _to_sparse(self, X):
        return csr_matrix(np.array(X))

    def get_support(self, *args, **kwargs):
        return self.inner_model.get_support(*args, **kwargs)

    def set_params(self, n_estimators=None, **params):
        super(SparseWrapper, self).set_params(**params)
        if n_estimators is not None:
            self.inner_model.set_params(n_estimators=n_estimators)

    def fit(self, X, y):
        X = self._to_sparse(X)
        #print "sparser", X.shape
        self.inner_model.fit(X, y)
        self.feature_importances_ = self.inner_model.feature_importances_
        return self

    def predict(self, X):
        X = self._to_sparse(X)
        return self.inner_model.predict(X)


def get_support_for_feature_selection_wrapper(
    feature_selector_indices,
    inner_indices,
    indices,
    ):
    result_support_indices = feature_selector_indices[inner_indices]
    if indices:
        return result_support_indices
    else:
        raise KyeError("indices should be true")


class ModelFeatureSelectionWrapper(BaseEstimator):
    def __init__(self, estimator, inner_model, feature_selection_threshold_coef=3):
        self.estimator=estimator
        self.inner_model = inner_model
        self.feature_selector = None
        self.feature_selection_threshold_coef = feature_selection_threshold_coef

    def _get_feature_selector(self):
        if self.feature_selector is None:
            self.feature_selector = SelectFromModel(self.estimator,
                                                    threshold='{}*mean'.format(float(self.feature_selection_threshold_coef)))
        return self.feature_selector

    def get_support(self, indices=False):
        feature_selector_support = self.feature_selector.get_support(indices=True)
        inner_support = self.inner_model.get_support(indices=True)
        return get_support_for_feature_selection_wrapper(
            feature_selector_support,
            inner_support,
            indices,
        )


    def fit(self, X, y):
        print X, X.shape
        X = self._get_feature_selector().fit(X.copy(), y.copy()).transform(X.copy())
        self.inner_model.fit(X.copy(), y)
        return self

    def predict(self, X):
        X = self._get_feature_selector().transform(X.copy())
        return self.inner_model.predict(X.copy())


class ModelBasedFeatureImportanceGetter(BaseEstimator):
    def __init__(self, inner_model):
        self.inner_model = inner_model

    def get_feature_importances(self, X, y):
        return self.inner_model.fit(X, y).feature_importances_

    def get_support(self, indices=False):
        return self._inner_model.get_support(indices=indices)


class SubsetGeneratorWrapper:
    def __init__(self, gen_getter):
        self._gen_getter = gen_getter

    def __getattr__(self, attr):
        return self._gen_getter().__getattribute__(attr)

    def __getinitargs__(self):
        return [self._gen_getter]


class AsMatrixWrapper(BaseEstimator):
    def __init__(self, inner_model):
        self.inner_model = inner_model

    def fit(self, X, y):
        self._feature_names = np.array(list(X.columns.values))
        self.inner_model.fit(X.as_matrix(), y)
        return self

    def predict(self, X):
        return self.inner_model.predict(X.as_matrix())

    def get_support(self, indices=False):
        if indices == False:
            raise KeyError("indices should be true")
        return self._feature_names[self.inner_model.get_support(indices=True)]

class LogisticRegressionWrapper(BaseEstimator):
    def __init__(self, lr):
        self.lr = lr

    def fit(self, X, y):
        self._features_count = X.shape[1]
        if self._features_count == 0:
            X = np.zeros((X.shape[0], 1), dtype=X.dtype)
        self.lr.fit(X, y)
        print "LR: ", X.shape,
        self.feature_importances_ = self.lr.coef_.ravel()

        print "LR: ", X.shape, self.feature_importances_.shape
        return self

    def predict(self, X):
        if self._features_count == 0:
            X = np.zeros((X.shape[0], 1), dtype=X.dtype)
        return self.lr.predict(X)

    def get_support(self, indices=False):
        if indices:
            return np.arange(self._features_count)
        else:
            return np.ones(self._features_count, dtype=np.bool)


class BorutaWrapper(BaseEstimator):
    def __init__(self, estimator, inner_model):
        self.estimator = estimator
        self.inner_model = inner_model

    def fit(self, X, y):
        self._feature_selector = BorutaPy(estimator=self.estimator)
        if len(X.shape) == 2 and X.shape[1] >= 1:
            print X, y
            X = self._feature_selector.fit_transform(X, y)
        else:
            X = np.ones((X.shape[0], 1), dtype=np.int64)
        self.inner_model.fit(X, y)
        return self

    def predict(self, X):
        if len(X.shape) == 2 and X.shape[1] >= 1:
            X = self._feature_selector.transform(X)
        else:
            X = np.ones((X.shape[0], 1), dtype=np.int64)
        return self.inner_model.predict(X)

    def get_support(self, indices=False):
        feature_selector_mask = self._feature_selector.support_
        feature_selector_indices = feature_selector_mask.nonzero()[0]
        inner_model_indices = self.inner_model.get_support(indices=True)
        return get_support_for_feature_selection_wrapper(
            feature_selector_indices,
            inner_model_indices,
            indices,
        )


class SelectKBestWrapper(BaseEstimator):
    def __init__(self, inner_model, k_best):
        self.inner_model = inner_model
        self.k_best = k_best

    def _fix_params(self, X, y):
        self.k_best.set_params(k=min(self.k_best.get_params()['k'], X.shape[1]))

    def fit(self, X, y):
        self._fix_params(X, y)
        X = self.k_best.fit_transform(X, y)
        self.inner_model.fit(X, y)
        return self

    def predict(self, X):
        X = self.k_best.transform(X)
        return self.inner_model.predict(X)

    def get_support(self, indices=False):
        feature_selector_support = self.k_best.get_support(indices=True)
        inner_model_support = self.inner_model.get_support(indices=True)
        return get_support_for_feature_selection_wrapper(
            feature_selector_support,
            inner_model_support,
            indices,
        )
