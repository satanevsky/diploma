from os.path import isfile
import cPickle as pickle
from hyperopt import Trials
import cPickle as pickle


ALL_TRIALS = dict()


class TrialsFactory(object):
    def __init__(self, experiment_name):
        self._experiment_name = experiment_name

    def _get_filename(self):
        return "{}.trials.pkl".format(self._experiment_name)

    def _get_trials_collection(self):
        if self._experiment_name in ALL_TRIALS:
            return ALL_TRIALS[self._experiment_name]
        elif isfile(self._get_filename()):
            with open(self._get_filename()) as f:
                ALL_TRIALS[self._experiment_name] = pickle.load(f)
                return ALL_TRIALS[self._experiment_name]
        else:
            ALL_TRIALS[self._experiment_name] = list()
            return ALL_TRIALS[self._experiment_name]

    def get_new_trials(self):
        trials = Trials()
        trials_collection = self._get_trials_collection()
        trials_collection.append(trials)
        return trials

    def flush(self):
        with open("{}.trials.pkl".format(self._experiment_name), 'wb') as f:
            pickle.dump(ALL_TRIALS[self._experiment_name], f)

    def __del__(self):
        self.flush()
