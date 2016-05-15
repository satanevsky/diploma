from os.path import isfile
import cPickle as pickle
from hyperopt import Trials
import cPickle as pickle


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

    def flush(self, force=True):
        if len(ALL_TRIALS[self._experiment_name]) == self._flushed:
            return
        with open("{}_{}.trials.pkl".format(self._experiment_name, self._flushed), 'wb') as f:
            pickle.dump(ALL_TRIALS[self._experiment_name][self._flushed:], f)
            self._flushed = len(ALL_TRIALS[self._experiment_name])

    def __del__(self):
        self.flush()
