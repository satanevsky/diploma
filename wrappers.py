from copy import copy
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel
from scipy.sparse import csr_matrix
from xgboost import XGBClassifier
from joblib import Memory
from common import forward_out
from generate_subsets import SubsetGenerator


mem_xgb = Memory(cachedir='cache/xgboost')


@forward_out("logs/xgb.log")
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
        result = np.array([importances_dict.get('f{}'.format(i), 0) for i in xrange(self.__features_count)], dtype=np.float64)
        return result

    def fit(self, X, y):
        if X.shape[1] == 0:
            X = np.zeros((X.shape[0], 1))
        super(XGBoostClassifierFeatureImportances, self).fit(X, y)
        self.__dict__.update(fit_xgboost(self.get_params(), X, y).__dict__)
        self.__features_count = X.shape[1]
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
        support = self._inner_model.get_support(indices=True)
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
        print "cleaner", X.shape
        self.inner_model.fit(X, y)
        #self.feature_importances_ = self.inner_model.feature_importances_
        return self

    def predict(self, X):
        X = X.copy()
        X = self._drop(X)
        return self.inner_model.predict(X)

class SparseWrapper(BaseEstimator):
    def __init__(self, inner_model):
        self.inner_model = inner_model

    def _to_sparse(self, X):
        return csr_matrix(np.array(X))

    def get_support(self, *args, **kwargs):
        return self.inner_model.get_support(*args, **kwargs)

    @property
    def feature_importances_(self):
        return self.inner_model.feature_importances_

    def fit(self, X, y):
        X = self._to_sparse(X)
        #print "sparser", X.shape
        self.inner_model.fit(X, y)
        #self.feature_importances_ = self.inner_model.feature_importances_
        return self

    def predict(self, X):
        X = self._to_sparse(X)
        return self.inner_model.predict(X)


class ModelFeatureSelectionWrapper(BaseEstimator):
    def __init__(self, inner_model, feature_selection_threshold_coef=3):
        self.inner_model = inner_model
        self.feature_selector = None
        self.feature_selection_threshold_coef = feature_selection_threshold_coef

    def _get_feature_selector(self):
        if self.feature_selector is None:
            self.feature_selector = SelectFromModel(XGBoostClassifierFeatureImportances(n_estimators=40),
                                                    threshold='{}*mean'.format(float(self.feature_selection_threshold_coef)))
        return self.feature_selector

    def get_support(self, indices=False):
        current_support = self.feature_selector.get_support(indices=True)
        inner_support = self.inner_model.get_support(indices=False)
        result_support_indices = current_support[inner_support]
        if indices:
            return result_support_indices
        else:
            result = np.zeros(current_support.shape, dtype=np.bool)
            result[result_support_indices] = True
            return result


    def fit(self, X, y):
        X = self._get_feature_selector().fit_transform(X, y)
        self.inner_model.fit(X, y)
        return self

    def predict(self, X):
        X = self._get_feature_selector().transform(X)
        return self.inner_model.predict(X)


class ModelBasedFeatureImportanceGetter(BaseEstimator):
    def __init__(self, inner_model):
        self.inner_model = inner_model

    def get_feature_importances(self, X, y):
        return self.inner_model.fit(X, y).feature_importances_

    def get_support(self, indices=False):
        return self._inner_model.get_support(indices=indices)


class SubsetGeneratorWrapper(SubsetGenerator):
    def __deepcopy__(self, memo):
        return self
