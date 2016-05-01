from collections import defaultdict
from hyperopt import hp
from hyperopt.pyll import scope
from sklearn.linear_model import LogisticRegression
from wrappers import SparseWrapper
from wrappers import XGBoostClassifierFeatureImportances as XGB
from wrappers import ModelFeatureSelectionWrapper
from complex_features_insertion import SimplePriorityGetter
from complex_features_insertion import BayesBasedPriorityGetter
from complex_features_insertion import MinSimpleFeaturesIndexGetter
from complex_features_insertion import AndBasedSimpleFeaturesIndexGetter
from complex_features_insertion import ExtenderStrategy


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
def get_model_based_feature_selection_model(*args, **kwargs):
    return ModelFeatureSelectionWrapper(*args, **kwargs)


def get_model_based_feature_selector(name, inner_model):
    return scope.get_model_based_feature_selection_model(
        inner_model=inner_model,
        feature_selection_threshold_coef=hp.loguniform(get_full_name(name, "threshold"), -5, 5),
    )


def get_model_params(name="model_common", xgb_name=None, lr_name=None):
    xgb_result_name = xgb_name if xgb_name is not None else get_full_name(name, 'xgb')
    lr_result_name = lr_name if lr_name is not None else get_full_name(name, 'lr')
    return hp.choice(name, (get_xgboost_params(xgb_result_name), get_linear_model_params(lr_result_name)))


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


def get_extender_strategy_params(name='extender_strategy',
                                 priority_getter=None,
                                 pre_filter=None,
                                 simple_features_indexes_getter=None):



def get_simple_feature_adder_wrapper_params(
        inner_model,
        pre_filter=None,
        simple_features_indexes_getter=None,
        priority_getter=None,
    ):
    generator, matrix_before_generating = get_ready_generator()
    priority_getter = hp.choice(
        "priority_getter",
        (SimplePriorityGetter(), get_bayes_based_priority_getter_params()),
    )
    pre_filter = get_min_size_prefilter_params() if pre_filter is None else pre_filter
    simple_features_indexes_getter = get_min_simple_features_index_getter_params() if simple_features_indexes_getter is not None else simple_features_indexes_getter
    extender_strategy = ExtenderStrategy(
        max_features=1000,
        priority_getter=priority_getter,
        pre_filter=pre_filter,
        generator=get_ready_generator()[0],
        simple_features_indexes_getter=simple_features_indexes_getter,
    )
    return ComplexFeaturesAdderWrapper(
        inner_model=feature_selector,
        matrix_before_generating=matrix_before_generating.as_matrix(),
        features_names=list(matrix_before_generating.columns.values),
        extender_strategy=extender_strategy
    )
