from collections import defaultdict
import traceback
from hyperopt import hp, STATUS_OK, STATUS_FAIL, Trials
from hyperopt.pyll import scope
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from wrappers import SparseWrapper
from wrappers import XGBoostClassifierFeatureImportances as XGB
from wrappers import ModelFeatureSelectionWrapper
from wrappers import BorutaWrapper
from wrappers import SelectKBestWrapper
from complex_features_insertion import SimplePriorityGetter
from complex_features_insertion import BayesBasedPriorityGetter
from complex_features_insertion import MinSimpleFeaturesIndexGetter
from complex_features_insertion import AndBasedSimpleFeaturesIndexGetter
from complex_features_insertion import ExtenderStrategy
from trials_keeper import TrialsFactory

RANDOM_STATE = 42


def get_full_name(model_name, local_name):
    return "{}__{}".format(model_name, local_name)


@scope.define
def get_lr_model(*args, **kwargs):
    return SparseWrapper(LogisticRegression(*args, **kwargs))


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
    return SparseWrapper(XGB(*args, **kwargs))


def get_xgboost_params(name="xgboost_common"):
    return scope.get_xgb_model(
        n_estimators=hp.quniform(get_full_name(name, "n_estimators"), 1, 200, 1),
        max_depth=hp.quniform(get_full_name(name, 'max_depth'), 1, 13, 1),
        min_child_weight=hp.quniform(get_full_name(name, 'min_child_weight'), 1, 6, 1),
        subsample=hp.uniform(get_full_name(name, 'subsample'), 0.5, 1),
        gamma=hp.uniform(get_full_name(name, 'gamma'), 0.5, 1),
        colsample_bytree=hp.uniform(get_full_name(name, 'colsample_bytree'), 0.05, 1.0),
        nthread=-1,
        seed=RANDOM_STATE,
    )


@scope.define
def get_rf_model(*args, **kwargs):
    return RandomForestClassifier(*args, **kwargs)


def get_rf_model_params(name='rf'):
    return scope.get_rf_model(n_estimators=100)


@scope.define
def get_model_based_feature_selection_model(*args, **kwargs):
    return ModelFeatureSelectionWrapper(*args, **kwargs)


def get_model_based_feature_selector_params(
        inner_model_params,
        name='model_based_feature_selector'
    ):
    return scope.get_model_based_feature_selection_model(
        inner_model=inner_model_params,
        feature_selection_threshold_coef=hp.loguniform(get_full_name(name, "threshold"), -5, 5),
    )


@scope.define
def get_boruta_feature_selector(*args, **kwargs):
    raise BorutaWrapper(*args, **kwargs)


def get_boruta_feature_selector_params(
        inner_model_params,
        name='boruta_common'
    ):
    return scope.get_boruta_feature_selector(
        estimator=estimator,
        inner_model=inner_model_params
    )


@scope.define
def get_k_best(*args, **kwargs):
    return SelectKBestWrapper(SelectKBest(*args, **kwargs))


def get_k_best_params(name='k_best_selector'):
    return scope.get_k_best(
        k=hp.qloguniform(get_full_name(name, 'k'), 0, 5, 1),
        score_func=hp.choice(get_full_name(name, 'score'), (chi2, f_classif)),
    )


def get_feature_selector_estimator_params(name='feature_selector_estimator'):
    return hp.choice(
        name, (
            get_rf_model_params(name=get_full_name(name, 'rf')),
            get_linear_model_params(name=get_full_name(name, 'lr')),
        )
    )


def get_feature_selector_params(name='feature_selector_params'):
    return hp.choice(
            name, (
                get_k_best_params(name=get_full_name(name, 'k_best')),
                get_boruta_feature_selector_params(
                    name=get_full_name(name, 'boruta'),
                    inner_model_params=get_feature_selector_estimator_params(
                        name=get_full_name(name, 'boruta_estimator'),
                    ),
                ),
                get_model_based_feature_selector_params(
                    name=get_full_name(name, 'model_based_selector'),
                    inner_model_params=get_feature_selector_estimator_params(
                        name=get_full_name(name, 'model_based_estimator'),
                    )
                ),
            ),
        )


def get_model_params(name="model_common", xgb_name=None, lr_name=None):
    xgb_result_name = xgb_name if xgb_name is not None else get_full_name(name, 'xgb')
    lr_result_name = lr_name if lr_name is not None else get_full_name(name, 'lr')
    return hp.choice(name, (
                    get_xgboost_params(xgb_result_name),
                    get_linear_model_params(lr_result_name)
                )
           )


@scope.define
def get_bayes_based_priority_getter(*args, **kwargs):
    return BayesBasedPriorityGetter(*args, **kwargs)


def get_bayes_based_priority_getter_params(name='bayes_based_priority_getter_common'):
    return scope.get_bayes_based_priority_getter(
        max_features=hp.quniform(get_full_name(name, 'max_features'), 0, 10, 1),
    )


@scope.define
def get_min_size_prefilter(*args, **kwargs):
    return MinSizePreFilter(*args, **kwargs)


def get_min_size_prefilter_params(name="min_size_prefilter_common"):
    return scope.get_min_size_prefilter(min_size=np.quniform(0, 10, 1))


@scope.define
def get_and_based_index_getter(*args, **kwargs):
    return AndBasedSimpleFeaturesIndexGetter(*args, **kwargs)


def get_and_based_index_getter_params(name="and_based_index_getter_common"):
    all_features = get_ready_generator()[1].as_matrix()
    return scope.get_and_based_index_getter_params(
        use_raw_candidate=hp.choice(get_full_name(name, 'use_raw_candidate'), (False, True)),
        all_features=all_features,
    )


@scope.define
def get_min_simple_features_index_getter(*args, **kwargs):
    return MinSimpleFeaturesIndexGetter(*args, **kwargs)


def get_min_simple_features_index_getter_params(name="min_simple_features_index_getter_common"):
    generator = get_ready_generator()[0]
    return scope.get_min_simple_features_index_getter(
        generator=generator,
        use_raw_candidate=hp.choice(get_full_name(name, 'use_raw_candidate'), (False, True)),
    )


def get_index_getter_params(name='features_index_getter_common'):
    return hp.choice(name, get_and_based_index_getter_params(), get_min_simple_features_index_getter_params())


@scope.define
def get_extender_strategy(*args, **kwargs):
    return ExtenderStrategy(*args, **kwargs)


def get_priority_getter_params(name='priority_getter_common'):
    return hp.choice(
        name,
        (
            SimplePriorityGetter(),
            get_bayes_based_priority_getter_params(
                get_full_name(name, 'bayes_priority_getter'),
            ),
        ),
    )


@scope.define
def get_complex_features_adder_wrapper(*args, **kwargs):
    return ComplexFeaturesAdderWrapper(*args, **kwargs)


def get_simple_feature_adder_wrapper_params(
        inner_model,
        max_features=None,
        pre_filter=None,
        features_indexes_getter=None,
        priority_getter=None,
        name='feature_adder_common'
    ):
    generator, matrix_before_generating = get_ready_generator()
    priority_getter = priority_getter if priority_getter is not None
        else get_priority_getter_params(get_full_name(name. 'priority_getter'))
    pre_filter = pre_filter if pre_filter is not None
        else get_min_size_prefilter_params(get_full_name(name, 'pre_filter'))
    features_indexes_getter = features_indexes_getter if features_indexes_getter is not None
        else get_index_getter_params(get_full_name(name, 'indexes_getter'))
    max_features = max_features if max_features is not None
        else np.qloguniform(
                get_full_name(name, 'max_features'),
                -1, 10, 1,
            )
    extender_strategy = scope.get_extender_strategy(
        max_features=max_features,
        priority_getter=priority_getter,
        pre_filter=pre_filter,
        generator=get_ready_generator()[0],
        simple_features_indexes_getter=simple_features_index_getter,
    )
    return scope.get_complex_features_adder_wrapper(
        inner_model=inner_model,
        matrix_before_generating=matrix_before_generating.as_matrix(),
        features_names=list(matrix_before_generating.columns.values),
        extender_strategy=extender_strategy
    )


def get_objective_function(X, y, metrics_getter, callback=None):
    def objective(model):
        start_time = time.time()
        try:
            metrics, loss = metrics_getter(model, X, y)
            result = {
                'status': STATUS_OK,
                'loss': loss,
                'full_metrics': metrics,
                'time_calculated': time.time(),
                'time_spent': time.time() - start_time,
            }
        except:
            result = {
                'status': STATUS_FAIL,
                'traceback': traceback.format_exc(),
                'time_calculated': time.time(),
                'time_spent': time.time() - start_time,
            }
        if callback is not None:
            callback(result)
    return objective


class HyperParameterSearcher(BaseEstimator):
    def __init__(self, params, trials_factory, metrics_getter, max_evals=100):
        self.params = params
        self.trials_factory = trials_factory
        self.metrics_getter = metrics_getter

    def fit(self, X, y):
        trials = self.trials_factory.get_new_trials()
        try:
            self._result_model = fmin(
                get_objective_function(
                    X,
                    y,
                    self.metrics_getter,
                    callback=lambda result: self.trials_factory.flush()
                ),
                space=self.params,
                algo=tpe.suggest,
                max_evals=self.max_evals,
                trials=trials,
            )
        finally:
            self.trials_factory.flush()
        self._result_model.fit(X, y)
        return self

    def predict(self, X):
        return self._result_model.predict(X)

    def get_support(self, indices=False):
        return self._result_model.get_support(indices=indices)
