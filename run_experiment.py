import sys
from multiprocessing import Process
from sklearn.cross_validation import StratifiedKFold
from common import PROCESSORS_COUNT, RANDOM_STATE
from data_keeper import get_data_keeper
from hyperparameter_search import HyperParameterSearcher
from saving_results import ResultsDumper
from testing import MetricsGetter, ALL_METRICS, ACCURACY
from generate_subsets_for_common_x import get_ready_generator


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


def run_experiment_fold(model, X, y, train_index, test_index, fold_index):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    model.results_dumper.set_subdir(str(fold_index))
    model.results_dumper.set_test_true(y_test)
    sys.stdout = sys.stderr = model.results_dumper.get_logs_file()
    model.fit(X_train, y_train, X_test)


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
    loss_getter = AccuracyLossGetter()
    inner_metrics_getter = MetricsGetter(
        metrics=ALL_METRICS,
        as_indexes=as_indexes,
        loss_func=loss_getter,
        n_folds=5,
    )
    model = HyperParameterSearcher(
        params=params,
        results_dumper=results_dumper,
        metrics_getter=inner_metrics_getter,
        max_evals=max_evals,
    )

    X, y = get_data_keeper().get_train_data(drug, as_indexes=as_indexes)

    n_folds = 5

    if len(y) < 50:
        n_folds = 10

    init_common()
    
    processes = list()
    
    for i, (train_index, test_index) in enumerate(StratifiedKFold(y, n_folds=5, shuffle=True, random_state=RANDOM_STATE)):
        process = Process(target=run_experiment_fold, args=(model, X, y, train_index, test_index, i))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
    