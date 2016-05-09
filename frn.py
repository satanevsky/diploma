import numpy as np
from sklearn.base import BaseEstimator
from joblib import Memory
from pygco import pygco
from common import forward_out
#from numba import jit


mem_frn = Memory(cachedir='cache/frn')

@forward_out("logs/frn.log")
@mem_frn.cache
def get_rel_feat_frn(params, X, feature_importances, importance_threshold):
    frn = FeatureRelevanceNetwork(**params)
    return frn.get_relevant_features(X, feature_importances, importance_threshold)


#@jit(nopython=True)
def do_params_estimation_heavy_job(
    correlation_coefficient,
    min_correlation,
    edges,
    edge_weights,
    ):
    edges_added = 0
    for i in xrange(correlation_coefficient.shape[0]):
        for j in xrange(i + 1, correlation_coefficient.shape[1]):
            if correlation_coefficient[i, j] >= min_correlation:
                edges[edges_added, 0] = i
                edges[edges_added, 1] = j
                edge_weights[edges_added] = correlation_coefficient[i, j]
                edges_added += 1
    return edges_added


class FeatureRelevanceNetwork(BaseEstimator):
    def __init__(self,
                 min_correlation=0.8,
                 feature_importance_priority=1000000000000.0,
                 feature_selection_threshold_coef=1.0):
        self.min_correlation = min_correlation
        self.feature_importance_priority = feature_importance_priority
        self.feature_selection_threshold_coef = feature_selection_threshold_coef

    def _get_model_parameters(self, X, feature_importances, importance_threshold):
        real_importances = (np.array(feature_importances) - np.array(importance_threshold)).astype(np.float64)
        real_importances /= np.sqrt((real_importances ** 2).mean())
        #print "real importances", (real_importances > 0).sum()
        unary_cost = np.zeros((X.shape[1], 2), dtype=np.float64)
        unary_cost[:,0] = real_importances * self.feature_importance_priority
        pairwise_cost = np.array([[0.0, 1.0],
                                  [1.0, 0.0]])
        correlation_coefficient = np.corrcoef(np.array(X.T, dtype=np.float64))
        edges_count = ((correlation_coefficient >= self.min_correlation - 0.1).sum() - correlation_coefficient.shape[0]) / 2
        edges = np.zeros((edges_count + 10, 2), dtype=np.uint32)
        edge_weights = np.zeros(edges_count + 10, dtype=np.float64)
        edges_added = do_params_estimation_heavy_job(
            correlation_coefficient,
            self.min_correlation,
            edges,
            edge_weights,
        )
        edges = edges[:edges_added]
        edge_weights = edge_weights[:edges_added]
        return edges, edge_weights, unary_cost, pairwise_cost

    def get_relevant_features(self, X, feature_importances, importance_threshold):
        print "get_relevant_features_call"
        edges, edge_weights, unary_cost, pairwise_cost = self._get_model_parameters(X, feature_importances, importance_threshold)
        result = pygco.cut_general_graph(edges, edge_weights, unary_cost, pairwise_cost, n_iter=1, algorithm="expansion")
        print "features count:", result.sum()
        print result, result.shape
        return result



class FeatureRelevanceNetworkWrapper(BaseEstimator):
    def __init__(self,
                 inner_model,
                 feature_importances_getter,
                 min_correlation=0.8,
                 feature_importance_priority=1000000000000.0,
                 feature_selection_threshold_coef=1.0):
        self.inner_model = inner_model
        self.feature_importances_getter = feature_importances_getter
        self.min_correlation = min_correlation
        self.feature_importance_priority = feature_importance_priority
        self.feature_selection_threshold_coef = feature_selection_threshold_coef

    def _get_reduced_params(self):
        result = self.get_params(deep=False).copy()
        del result['inner_model']
        del result['feature_importances_getter']
        return result

    def _filter_X(self, X):
        result = np.array(X)[:,self._relevant_feature_mask.astype(np.bool)]
        return result

    def _set_relevant_feature_mask(self, X, y):
        feature_importances = self.feature_importances_getter.get_feature_importances(X, y)
        importance_threshold = feature_importances.mean() * self.feature_selection_threshold_coef
        self._relevant_feature_mask = get_rel_feat_frn(self._get_reduced_params(), X, feature_importances, importance_threshold)

    def get_support(self, indices=False):
        if not indices:
            return self._relevant_feature_mask
        else:
            return np.array([i for i in xrange(len(self._relevant_feature_mask)) if self._relevant_feature_mask[i]])

    def fit(self, X, y):
        self._set_relevant_feature_mask(X, y)
        X = self._filter_X(X)
        self.inner_model.fit(X, y)
        return self

    def predict(self, X):
        X = self._filter_X(X)
        return self.inner_model.predict(X)
