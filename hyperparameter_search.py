from collections import defaultdict
import traceback
import time
import cPickle as pickle
from hyperopt import hp, STATUS_OK, STATUS_FAIL, Trials, fmin, tpe
from hyperopt.pyll import scope
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.linear_model import LogisticRegression
from wrappers import XGBoostClassifierFeatureImportances as XGB
from wrappers import ModelFeatureSelectionWrapper
from wrappers import BorutaWrapper
from wrappers import SelectKBestWrapper
from wrappers import AsMatrixWrapper
from wrappers import LogisticRegressionWrapper
from wrappers import MatrixCleaningWrapper
from complex_features_inserting import SimplePriorityGetter
from complex_features_inserting import BayesBasedPriorityGetter
from complex_features_inserting import MinSimpleFeaturesIndexGetter
from complex_features_inserting import AndBasedSimpleFeaturesIndexGetter
from complex_features_inserting import ExtenderStrategy
from complex_features_inserting import NothingDoingExtenderStrategy
from complex_features_inserting import MinSizePreFilter
from complex_features_inserting import ComplexFeaturesAdderWrapper
from frn import FeatureRelevanceNetworkWrapper
from common import RANDOM_STATE
from generate_subsets_for_common_x import get_ready_generator


def get_full_name(model_name, local_name):
    return "{}__{}".format(model_name, local_name)


@scope.define
def get_as_matrix_wrapper(*args, **kwargs):
    return AsMatrixWrapper(*args, **kwargs)


def get_as_matrix_wrapper_params(inner_model_params):
    return scope.get_as_matrix_wrapper(inner_model_params)

@scope.define
def get_lr_model(*args, **kwargs):
    return LogisticRegressionWrapper(
        lr=LogisticRegression(*args, **kwargs)
    )


def get_linear_model_params(name="linear_common"):
    return scope.get_lr_model(
        C=hp.loguniform(get_full_name(name, 'C'), -15, 15),
        penalty=hp.choice(get_full_name(name, 'penalty'), ('l1', 'l2')),
        class_weight=hp.choice(get_full_name(name, 'class_weight'), (defaultdict(lambda: 1.0), 'balanced')),
        fit_intercept=hp.choice(get_full_name(name, 'fit_intercept'), (False, True)),
        random_state=RANDOM_STATE,
        solver='liblinear',
    )


@scope.define
def get_xgb_model(*args, **kwargs):
    return XGB(*args, **kwargs)

#@scope.define
#def int(val):
#    return int(val)


def get_xgboost_params(name="xgboost_common"):
    return scope.get_xgb_model(
        n_estimators=scope.int(
            hp.quniform(
                get_full_name(name, "n_estimators"),
                1, 200, 1,
            ),
        ),
        max_depth=scope.int(
            hp.quniform(
                get_full_name(name, 'max_depth'),
                1, 13, 1,
            ),
        ),
        min_child_weight=scope.int(
            hp.quniform(
                get_full_name(name, 'min_child_weight'),
                1, 6, 1,
            ),
        ),
        subsample=scope.int(
            hp.uniform(
                get_full_name(name, 'subsample'),
                0.5, 1,
            ),
        ),
        gamma=hp.uniform(
            get_full_name(name, 'gamma'),
            0.5, 1,
        ),
        nthread=1,
        seed=RANDOM_STATE,
    )


@scope.define
def get_matrix_cleaning_wrapper(*args, **kwargs):
    return MatrixCleaningWrapper(*args, **kwargs)


def get_matrix_cleaning_wrapper_params(inner_model_params):
    return scope.get_matrix_cleaning_wrapper(inner_model_params)


@scope.define
def get_rf_model(*args, **kwargs):
    return RandomForestClassifier(*args, **kwargs)


def get_rf_model_params(name='rf'):
    return scope.get_rf_model(n_estimators=100, n_jobs=1)


@scope.define
def get_model_based_feature_selection_model(*args, **kwargs):
    return ModelFeatureSelectionWrapper(*args, **kwargs)


def get_model_based_feature_selector_params(
        inner_model_params,
        name='model_based_feature_selector',
        estimator=None,
    ):
    if estimator is None:
        estimator = get_feature_selector_estimator_params()
    return scope.get_model_based_feature_selection_model(
        estimator=estimator,
        inner_model=inner_model_params,
        feature_selection_threshold_coef=hp.loguniform(get_full_name(name, "threshold"), -5, 5),
    )


@scope.define
def get_boruta_feature_selector(*args, **kwargs):
    return BorutaWrapper(*args, **kwargs)


def get_boruta_feature_selector_params(
        inner_model_params,
        name='boruta_common',
        estimator=None,
    ):
    if estimator is None:
        estimator=get_rf_model_params(
            name=get_full_name(name, 'estimator'),
        )
    return scope.get_boruta_feature_selector(
        estimator=estimator,
        inner_model=inner_model_params,
    )


@scope.define
def get_k_best_wrapper(*args, **kwargs):
    return SelectKBestWrapper(*args, **kwargs)


@scope.define
def get_k_best(*args, **kwargs):
    return SelectKBest(*args, **kwargs)


def get_k_best_params(inner_model_params, name='k_best_selector'):
    return scope.get_k_best_wrapper(
        inner_model=inner_model_params,
        k_best=scope.get_k_best(
            k=hp.qloguniform(get_full_name(name, 'k'), 0, 5, 1),
            score_func=hp.choice(get_full_name(name, 'score'), (chi2, f_classif)),
        ),
    )


def get_feature_selector_estimator_params(name='feature_selector_estimator'):
    return hp.choice(
        name, (
            get_rf_model_params(name=get_full_name(name, 'rf')),
            get_linear_model_params(name=get_full_name(name, 'lr')),
        )
    )


def get_feature_selector_params(inner_model_params, name='feature_selector_params'):
    model_based_estimator = get_feature_selector_estimator_params()
    return hp.choice(
            name, (
                get_k_best_params(
                    inner_model_params=inner_model_params,
                    name=get_full_name(name, 'k_best'),
                ),
                #get_boruta_feature_selector_params(
                #    name=get_full_name(name, 'boruta'),
                #    inner_model_params=inner_model_params,
                #),
                get_model_based_feature_selector_params(
                    name=get_full_name(name, 'model_based_selector'),
                    estimator=model_based_estimator,
                    inner_model_params=inner_model_params,
                ),
            ),
        )


def get_model_params(name="model_common", xgb_name=None, lr_name=None):
    xgb_result_name = xgb_name if xgb_name is not None else get_full_name(name, 'xgb')
    lr_result_name = lr_name if lr_name is not None else get_full_name(name, 'lr')
    return hp.choice(name, (
                    get_xgboost_params(xgb_result_name),
                    get_linear_model_params(lr_result_name),
                    get_rf_model_params('rf'),
                )
           )


@scope.define
def get_bayes_based_priority_getter(*args, **kwargs):
    return BayesBasedPriorityGetter(*args, **kwargs)


def get_bayes_based_priority_getter_params(name='bayes_based_priority_getter_common'):
    return scope.get_bayes_based_priority_getter(
        max_features=hp.quniform(
            get_full_name(name, 'max_features'),
            0, 3000, 1,
        ),
    )


@scope.define
def get_min_size_prefilter(*args, **kwargs):
    return MinSizePreFilter(*args, **kwargs)


def get_min_size_prefilter_params(name="min_size_prefilter_common"):
    return scope.get_min_size_prefilter(
        min_size=hp.quniform(
            get_full_name(name, 'min_size'),
            0, 10, 1,
        ),
    )


@scope.define
def get_and_based_index_getter(*args, **kwargs):
    kwargs['all_features'] = get_ready_generator()[1].as_matrix()
    return AndBasedSimpleFeaturesIndexGetter(*args, **kwargs)


def get_and_based_index_getter_params(name="and_based_index_getter_common"):
    return scope.get_and_based_index_getter(
        use_raw_candidate=hp.choice(
            get_full_name(name, 'use_raw_candidate'),
            (False, True),
        ),
    )


@scope.define
def get_min_simple_features_index_getter(*args, **kwargs):
    kwargs['generator'] = get_ready_generator()[0]
    return MinSimpleFeaturesIndexGetter(*args, **kwargs)


def get_min_simple_features_index_getter_params(
    name="min_simple_features_index_getter_common"
    ):
    return scope.get_min_simple_features_index_getter(
        use_raw_candidate=hp.choice(
            get_full_name(name, 'use_raw_candidate'),
            (False, True),
        ),
        max_check=1000,
    )


def get_index_getter_params(name='features_index_getter_common'):
    return hp.choice(
        name,
        (
            get_and_based_index_getter_params(),
            get_min_simple_features_index_getter_params(),
        ),
    )


@scope.define
def get_extender_strategy(*args, **kwargs):
    kwargs['generator'] = get_ready_generator()[0]
    return ExtenderStrategy(*args, **kwargs)


@scope.define
def get_simple_priority_getter():
    return SimplePriorityGetter()


def get_priority_getter_params(name='priority_getter_common'):
    return hp.choice(
        name,
        (
            scope.get_simple_priority_getter(),
            get_bayes_based_priority_getter_params(
                get_full_name(name, 'bayes_priority_getter'),
            ),
        ),
    )


@scope.define
def get_complex_features_adder_wrapper(*args, **kwargs):
    matrix_before_generating = get_ready_generator()[1]
    kwargs['matrix_before_generating'] = matrix_before_generating.as_matrix()
    kwargs['features_names'] = list(matrix_before_generating.columns.values)
    return ComplexFeaturesAdderWrapper(*args, **kwargs)


@scope.define
def get_frn(*args, **kwargs):
    return FeatureRelevanceNetworkWrapper(*args, **kwargs)


def get_frn_params(
    inner_model,
    feature_importances_getter=None,
    name='frn_common'
    ):
    if feature_importances_getter is None:
        feature_importances_getter = get_feature_selector_estimator_params()

    return scope.get_frn(
        inner_model=inner_model,
        feature_importances_getter=feature_importances_getter,
        lmbda=hp.loguniform(
            get_full_name(name, 'lmbda'),
            -5, 15,
        ),
        feature_selection_threshold_coef=hp.uniform(
            get_full_name(name, 'threshold_coef'),
            0.0, 1.0,
        ),
    )


@scope.define
def get_nothing_doing_extender_strategy():
    return NothingDoingExtenderStrategy()


def get_simple_feature_adder_wrapper_params(
        inner_model_params,
        max_features=None,
        pre_filter=None,
        features_indexes_getter=None,
        priority_getter=None,
        name='feature_adder_common'
    ):
    priority_getter = priority_getter if priority_getter is not None \
        else get_priority_getter_params(get_full_name(name, 'priority_getter'))
    pre_filter = pre_filter if pre_filter is not None \
        else get_min_size_prefilter_params(get_full_name(name, 'pre_filter'))
    features_indexes_getter = features_indexes_getter if features_indexes_getter is not None \
        else get_index_getter_params(get_full_name(name, 'indexes_getter'))
    max_features = max_features if max_features is not None \
        else hp.qloguniform(
                get_full_name(name, 'max_features'),
                -1, 10, 1,
            )
    extender_strategy = hp.choice(
        get_full_name(name, 'extender_strategy'),
        (
            scope.get_extender_strategy(
                max_features=max_features,
                priority_getter=priority_getter,
                pre_filter=pre_filter,
                simple_features_indexes_getter=features_indexes_getter,
            ),
            scope.get_nothing_doing_extender_strategy(),
        ),
    )
    return scope.get_complex_features_adder_wrapper(
        inner_model=inner_model_params,
        extender_strategy=extender_strategy,
    )


def get_objective_function(X, y, X_test, metrics_getter, results_dumper, callback=None):
    def objective(model):
        print "!OBJECTIVE"
        start_time = time.time()
        try:
            metrics, loss = metrics_getter(model, X, y, X_test)
            result = {
                'status': STATUS_OK,
                'loss': loss,
                'full_metrics': metrics,
                'time_calculated': time.time(),
                'time_spent': time.time() - start_time,
                'model': model,
            }
        except:
            with open(results_dumper.get_errors_log_filename(), "a") as f:
                f.write(traceback.format_exc())
                f.write("\n")
                f.write(repr(model))
                f.write("\n")
            result = {
                'status': STATUS_FAIL,
                'traceback': traceback.format_exc(),
                'time_calculated': time.time(),
                'time_spent': time.time() - start_time,
                'model': model,
            }
        if callback is not None:
            callback(result)
        return result
    return objective


class HyperParameterSearcher(BaseEstimator):
    def __init__(self, params, results_dumper, metrics_getter, max_evals=100):
        self.params = params
        self.results_dumper = results_dumper
        self.metrics_getter = metrics_getter
        self.max_evals = max_evals

    def fit(self, X, y, X_test=None):
        try:
            fmin(
                get_objective_function(
                    X,
                    y,
                    X_test,
                    self.metrics_getter,
                    self.results_dumper,
                    callback=lambda result: self.results_dumper.add_result(result),
                ),
                space=self.params,
                algo=tpe.suggest,
                max_evals=self.max_evals,
            )
            min_loss = 10.0
            for el in trials.results:
                if el['status'] == STATUS_OK:
                    if el['loss'] < min_loss:
                        self._result_model = el['model']
                        min_loss = el['loss']
            print self._result_model
        finally:
            self.results_dumper.flush()
        self._result_model.fit(X, y)
        return self

    def predict(self, X):
        return self._result_model.predict(X)

    def get_support(self, indices=False):
        return self._result_model.get_support(indices=indices)
