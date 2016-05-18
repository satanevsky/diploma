import sys
from concurrent.futures import ProcessPoolExecutor
from common import PROCESSORS_COUNT
from data_keeper import get_data_keeper
from hyperparameter_search import HyperParameterSearcher
from saving_results import ResultsDumper
from testing import MetricsGetter, ALL_METRICS, ACCURACY
from generate_subsets_for_common_x import get_ready_generator


ALL_SECOND_LEVEL_DRUGS = "all_second_level_drugs"


def init_common():
    get_ready_generator()


def format_experiment_name(name):
    return name.replace(':', '').replace('/', '')


class AccuracyLossGetter:
    def __call__(self, metrics):
        return 1.0 - metrics[ACCURACY]


class ExperimentForDrugCaller:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __call__(self, drug):
        kwargs = self._kwargs.copy()
        kwargs['drug'] = drug
        return run_experiment(*self._args, **kwargs)


def run_experiment(
    params,
    experiment_name,
    drug,
    max_evals=100,
    as_indexes=True):

    if drug == ALL_SECOND_LEVEL_DRUGS:
        run_experiment_for_drug = ExperimentForDrugCaller(
            params=params,
            experiment_name=experiment_name,
            max_evals=max_evals,
            as_indexes=as_indexes,
        )
        drugs = get_data_keeper().get_possible_second_level_drugs()
        with ProcessPoolExecutor(max_workers=PROCESSORS_COUNT) as e:
            return list(e.map(run_experiment_for_drug, drugs))

    experiment_name_for_drug = format_experiment_name(
        "{}({})".format(experiment_name, drug),
    )
    results_dumper = ResultsDumper(experiment_name_for_drug)
    sys.stdout = sys.stderr = results_dumper.get_logs_file()
    loss_getter = AccuracyLossGetter()
    inner_metrics_getter = MetricsGetter(
        metrics=ALL_METRICS,
        as_indexes=as_indexes,
        loss_func=loss_getter,
        n_folds=5,
    )
    outer_metrics_getter = MetricsGetter(
        metrics=ALL_METRICS,
        as_indexes=as_indexes,
        loss_func=loss_getter,
        n_folds=10,
    )
    model = HyperParameterSearcher(
        params=params,
        results_dumper=results_dumper,
        metrics_getter=inner_metrics_getter,
        max_evals=max_evals,
    )
    X, y = get_data_keeper().get_train_data(drug, as_indexes=as_indexes)
    result_metrics, result_loss = outer_metrics_getter(model, X, y)
    results_dumper.dump_final_result((result_metrics, result_loss))
    return result_metrics, result_loss
