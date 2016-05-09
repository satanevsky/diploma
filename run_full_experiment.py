from data_keeper import get_data_keeper
from run_experiment import run_experiment
from hyperparameter_search import get_simple_feature_adder_wrapper_params,\
                                  get_frn_params, \
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

if __name__ == '__main__':
    params = get_all_params()
    drug = get_data_keeper().get_possible_second_level_drugs()[0]
    run_experiment(
        params=params,
        experiment_name='all_params',
        drug=drug,
    )
