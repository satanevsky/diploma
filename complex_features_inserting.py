import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln
from sklearn.base import BaseEstimator


def and_arrays(arrays):
    arrays_sum = arrays.sum(axis=0)
    return (arrays_sum == len(arrays)).astype(arrays.dtype)


class ExtenderStrategy(object):
    def __init__(self,
                 max_features,
                 priority_getter,
                 pre_filter,
                 generator,
                 simple_features_indexes_getter,
                 max_index=150):
        self._max_features = max_features
        self._generator = generator
        self._max_features_sets_storing = 1000000
        assert self._max_features_sets_storing > 2 * max_features
        self._best_feature_sets = list()
        self._max_index = max_index
        self._priority_getter = priority_getter
        self._pre_filter_strategy = pre_filter
        self._simple_features_indexes_getter = simple_features_indexes_getter

    def set_generator(self, generator):
        self._generator = generator

    def _generate_candidates(self):
        print self._generator.get_sets_count()
        for i in xrange(self._generator.get_sets_count()):
            yield self._get_translated_indexes(self._generator.get_set(i))

    def _pre_filter(self, candidate):
        return self._pre_filter_strategy.pre_filter(candidate)

    def _estimate_parameters(self, simple_features, y, indexes):
        self._priority_getter.estimate_parameters(self._generate_candidates(), simple_features, y, self._generator, indexes)

    def _set_indexes(self, indexes):
        self._index_translation = -1 * np.ones(self._max_index, dtype=np.int32)
        for i, el in enumerate(indexes):
            self._index_translation[el] = i
        self._indexes = indexes

    def _get_translated_indexes(self, candidate):
        translated_indexes_with_garbage = self._index_translation[candidate]
        return translated_indexes_with_garbage[translated_indexes_with_garbage >= 0]

    def _get_simple_feature_indexes(self, simple_features, candidate):
        return self._simple_features_indexes_getter.get_features_indexes(simple_features, candidate, self._indexes)

    def _set_result_feature_sets(self, feature_sets):
        self._result_feature_sets = feature_sets

    def _get_top_priority_feature_sets(self):
        return sorted(self._best_feature_sets, reverse=True, key=lambda x: x[0])[:self._max_features]

    def _clean_storing_feature_sets(self):
        self._best_feature_sets = self._get_top_priority_feature_sets()

    def _get_candidate_priority(self, candidate, y_values, simple_feature_indexes, candidate_index):
        return self._priority_getter.get_candidate_priority(candidate, y_values, simple_feature_indexes, candidate_index)

    def fit(self, simple_features, y, indexes):
        if self._generator is None:
            raise KeyError("generator should be setted")
        indexes = indexes.ravel()
        print indexes
        self._set_indexes(indexes)
        self._estimate_parameters(simple_features, y, indexes)
        self._simple_features_count = simple_features.shape[1]
        with self._priority_getter:
            for candidate_index, candidate in enumerate(self._generate_candidates()):
                #print candidate_index,
                if self._pre_filter(candidate):
                    candidate_feature_rows = simple_features[candidate]
                    y_values = y[candidate]
                    simple_feature_indexes = self._get_simple_feature_indexes(simple_features,
                                                                              candidate)
                    if simple_feature_indexes is None:
                        continue
                    priority = self._get_candidate_priority(candidate,
                                                            y_values,
                                                            simple_feature_indexes,
                                                            candidate_index)
                    if priority > 0:
                        self._best_feature_sets.append((priority, simple_feature_indexes))
                        if len(self._best_feature_sets) > self._max_features_sets_storing:
                            self._clean_storing_feature_sets()
        self._set_result_feature_sets([el[1] for el in self._get_top_priority_feature_sets()])
        return self

    def get_support(self, indices=False):
        if indices == False:
            raise KeyError("indices should be True")
        return [[el] for el in xrange(self._simple_features_count)] + self._result_feature_sets

    def transform(self, simple_features):
        to_add = np.zeros((simple_features.shape[0], len(self._result_feature_sets)), dtype=np.int32)
        for i, simple_features_indexes in enumerate(self._result_feature_sets):
            simple_features_values = simple_features[:,simple_features_indexes]
            result = and_arrays(simple_features_values.T)
            to_add[:,i] = result
        return np.concatenate((simple_features, to_add), axis=1)


class AndBasedSimpleFeaturesIndexesGetter:
    def get_features_indexes(self, simple_features, candidate, indexes):
        return and_arrays(simple_features[candidate]).nonzero()[0]


class MinSimpleFeaturesIndexGetter:
    def __init__(self, generator, max_check):
        self._generator = generator
        self._max_check = max_check

    def get_features_indexes(self, simple_features, candidate, indexes):
        raw_indexes = indexes[candidate]
        result = self._generator.get_probable_features_indexes(raw_indexes, self._max_check)
        if result[0] < 1000:
            return result
        else:
            return None


class MinSizePreFilter(object):
    def __init__(self, min_size):
        self._min_size = min_size

    def pre_filter(self, candidate):
        return len(candidate) >= self._min_size


class SimplePriorityGetter(object):
    def estimate_parameters(self, candidates_iterator, simple_features, y, generator, indexes):
        true_y_indexes = list()
        for i in xrange(len(indexes)):
            if y[i]:
                true_y_indexes.append(indexes[i])
        true_y_indexes = np.array(true_y_indexes)
        print true_y_indexes
        self._true_y_indexes = true_y_indexes
        self._generator = generator

    def get_candidate_priority(self, candidate, y_values, simple_feature_indexes, candidate_index):
        if y_values.sum() < len(y_values) or len(y_values) < 3:
            result = -1
        else:
            result = len(y_values)
        return result

    def __enter__(self):
        self._generator.set_filtered_have_ones_in_positions(self._true_y_indexes)

    def __exit__(self, type, value, tb):
        self._generator.restore()


class BayesBasedPriorityGetter(object):
    class ValidityCalculator(object):
        def __init__(self, all_true_statistics):
            self._all_true_statistics = all_true_statistics

        def __call__(self, (a, b)):
            ans = 0.0
            for i, row in enumerate(self._all_true_statistics):
                for j, val in enumerate(row):
                    if val != 0:
                        ans += (betaln(a + j, b + i - j) - betaln(a, b)) * val
            return -ans

    def __init__(self, max_features, reg_param=None):
        self._reg_param = reg_param
        self._max_features = max_features

    def estimate_parameters(self, candidates_iterator, simple_features, y, generator, indexes):
        self._generator = generator
        self._y = y.astype(np.int32)
        self._indexes = indexes.astype(np.int32)
        if self._reg_param is None:
            all_true_statistics = generator.get_count_and_y_true_statistics(self._y, self._indexes)
            validity_calculator = BayesBasedPriorityGetter.ValidityCalculator(all_true_statistics)
            ret = minimize(validity_calculator, [1, 1], bounds=[(-1, None), (-1, None)])
            self._reg_param = ret.x

    def __enter__(self):
        alpha_reg, beta_reg = self._reg_param
        self._generator.set_filtered_best_beta_binomial(alpha_reg, beta_reg, self._y, self._indexes, self._max_features)

    def __exit__(self, type, value, tb):
        self._generator.restore()

    def get_candidate_priority(self, candidate, y_values, simple_feature_indexes, candidate_index):
        return 1e9 - candidate_index


class ComplexFeaturesAdderWrapper(BaseEstimator):
    def __init__(self, inner_model, matrix_before_generating, features_names, extender_strategy):
        self.inner_model = inner_model
        self.matrix_before_generating = matrix_before_generating
        self.features_names = features_names
        self.extender_strategy = extender_strategy

    def _get_simple_features(self, indexes):
        return self.matrix_before_generating[indexes]

    def _fit_feature_extender(self, simple_features, y, indexes):
        self.extender_strategy.fit(simple_features, y, indexes)

    def _extend_features(self, simple_features):
        return self.extender_strategy.transform(simple_features)

    def fit(self, indexes=None, y=None, simple_features=None):
        indexes = indexes.ravel()
        if y is None:
            raise KeyError("y should be setted")
        if simple_features is None:
            simple_features = self._get_simple_features(indexes)
        else:
            simple_features = simple_features[self.features_names]
        self._fit_feature_extender(simple_features, y, indexes)
        extended_features = self._extend_features(simple_features)
        self.inner_model.fit(extended_features, y)
        return self

    def predict(self, indexes=None, simple_features=None):
        if simple_features is None:
            if indexes is None:
                raise KeyError("either simple_features or indexes should be setted")
            indexes = indexes.ravel()
            simple_features = self._get_simple_features(indexes)
        extended_features = self._extend_features(simple_features)
        return self.inner_model.predict(extended_features)

    def get_support(self, indices=False):
        if indices == False:
            raise KeyError("indices should be true")
        extender_support = self.extender_strategy.get_support(indices=True)
        return [extender_support[i] for i in self.inner_model.get_support(indices=True)]
