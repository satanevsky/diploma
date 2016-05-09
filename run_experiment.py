from data_keeper import get_data_keeper
from hyperparameter_search import HyperParameterSearcher
from trials_keeper import TrialsFactory
from testing import MetricsGetter, ALL_METRICS, F1


def run_experiment(
    params,
    experiment_name,
    drug,
    max_evals=100,
    as_indexes=True):
    experiment_name_for_drug = "{}({})".format(experiment_name, drug)
    trials_factory = TrialsFactory(experiment_name_for_drug)
    inner_metrics_getter = MetricsGetter(
        metrics=ALL_METRICS,
        as_indexes=as_indexes,
        loss_func=lambda metrics: 1.0 - metrics[F1],
        n_folds=5,
    )
    outer_metrics_getter = MetricsGetter(
        metrics=ALL_METRICS,
        as_indexes=as_indexes,
        loss_func=lambda metrics: 1.0 - metrics[F1],
        n_folds=10,
    )
    model = HyperParameterSearcher(
        params=params,
        trials_factory=trials_factory,
        metrics_getter=inner_metrics_getter,
        max_evals=max_evals,
    )
    X, y = get_data_keeper().get_train_data(drug, as_indexes=as_indexes)
    result_metrics, result_loss = outer_metrics_getter(model, X, y)
    with open("{}_final_results.pkl".format(experiment_name_for_drug), "wb") as f:
        pickle.dump((result_metrics, result_loss), f)
    return result_metrics, result_loss
