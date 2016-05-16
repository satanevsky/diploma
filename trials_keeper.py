from os.path import isfile
import cPickle as pickle
from hyperopt import Trials
from sklearn.externals import joblib


ALL_TRIALS = dict()


class TrialsFactory(object):
    def __init__(self, experiment_name):
        self._experiment_name = experiment_name
        self._flushed = 0

    def _get_trials_collection(self):
        if self._experiment_name in ALL_TRIALS:
            return ALL_TRIALS[self._experiment_name]
        else:
            ALL_TRIALS[self._experiment_name] = list()
            return ALL_TRIALS[self._experiment_name]

    def get_new_trials(self):
        trials = Trials()
        trials_collection = self._get_trials_collection()
        trials_collection.append(trials)
        return trials

    def _ready(self):
        for el in ALL_TRIALS[self._experiment_name]:
            if len(el.trials) % 100 == 0:
                return True
        return False

    def flush(self, force=True):
        with open("{}_counter.txt".format(self._experiment_name), 'w') as f:
            f.write("{}\n".format(sum([len(el.trials) for el in ALL_TRIALS[self._experiment_name]])))
        if force or self._ready():
            joblib.dump(ALL_TRIALS[self._experiment_name], "{}.trials.pkl".format(self._experiment_name), compress=3)

    def __del__(self):
        self.flush()
