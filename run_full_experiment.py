from multiprocessing import Process
from data_keeper import get_data_keeper
from run_experiment import run_experiment
from hyperparameter_search import get_simple_feature_adder_wrapper_params,\
                                  get_frn_params, \
                                  get_feature_selector_params, \
                                  get_model_params, \
                                  get_as_matrix_wrapper_params, \
                                  get_matrix_cleaning_wrapper_params
from generate_subsets_for_common_x import get_ready_generator


def init_common():
    get_ready_generator()


def get_all_params():
    inner_model_params = get_model_params()
    feature_selection_params = get_feature_selector_params(
        inner_model_params=inner_model_params,
    )
    result_params = get_simple_feature_adder_wrapper_params(
        inner_model_params=feature_selection_params,
    )
    return result_params


def run_full_experiment(drug_index=0):
    params = get_all_params()
    drug = get_data_keeper().get_possible_second_level_drugs()[drug_index]
    return run_experiment(
        params=params,
        experiment_name='all_params',
        drug=drug,
        as_indexes=True,
        max_evals=300,
    )


if __name__ == '__main__':
    init_common()
    processes = list()
    for i in xrange(4):
        process = Process(target=run_full_experiment, args=(i,))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

