from run_experiment import run_experiment, ALL_SECOND_LEVEL_DRUGS
from hyperparameter_search import get_simple_feature_adder_wrapper_params,\
                                  get_feature_selector_params, \
                                  get_model_params


def get_all_params():
    inner_model_params = get_model_params()
    feature_selection_params = get_feature_selector_params(
        inner_model_params=inner_model_params,
    )
    result_params = get_simple_feature_adder_wrapper_params(
        inner_model_params=feature_selection_params,
    )
    return result_params


def run_extender_selector_model(drug=ALL_SECOND_LEVEL_DRUGS):
    params = get_all_params()
    return run_experiment(
        params=params,
        experiment_name='extender_selector_model',
        drug=drug,
        as_indexes=True,
        max_evals=250,
    )


if __name__ == '__main__':
    run_extender_selector_model()
