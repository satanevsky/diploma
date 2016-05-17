import sys
import cPickle as pickle
from data_keeper import get_data_keeper
from hyperparameter_search import HyperParameterSearcher
from saving_results import ResultsDumper
from testing import MetricsGetter, ALL_METRICS, ACCURACY


def format_experiment_name(name):
    return name.replace(':', '').replace('/', '')

class AccuracyLossGetter:
    def __call__(self, metrics):
        return 1.0 - metrics[ACCURACY]


def run_experiment(
    params,
    experiment_name,
    drug,
    max_evals=100,
    as_indexes=True):
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
    print result_metrics, result_loss
    return result_metrics, result_loss
