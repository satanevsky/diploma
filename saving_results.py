import sys
from os import mkdir
from os.path import isfile, isdir, join
from collections import defaultdict
from sklearn.externals import joblib
from common import SAVE_RESULTS_AFTER


class ResultsDumper(object):
    def __init__(self, experiment_name):
        self._experiment_name = experiment_name
        self._unflushed = list()
        self._flush_counter = 0
        self._results_counter = 0
        self._folder = "experiments_results"
        self.set_subdir(self._experiment_name)
        
    def _get_folder(self):
        return self._folder

    def set_test_true(self, test_true):
        self._test_true = test_true

    def set_subdir(self, subdir):
        self._folder = join(self._folder, subdir)
        if not isdir(self._get_folder()):
            mkdir(self._get_folder())

    def add_result(self, result):
        result = result.copy()
        result['test_true'] = self._test_true
        self._unflushed.append(result)
        self._results_counter += 1
        with open(join(self._get_folder(), "counter.txt"), 'a') as f:
            f.write("{}\n".format(self._results_counter))
        if len(self._unflushed) >= SAVE_RESULTS_AFTER:
            self.flush()

    def get_logs_file(self):
        filename = join(self._get_folder(), "execution.log")
        return open(filename, 'a')

    def get_errors_log_filename(self):
        filename = join(self._get_folder(), "errors.log")
        return filename

    def dump_final_result(self, result):
        counter = 0
        while True:
            filename = join(self._get_folder(), "final_results_{}.pkl".format(counter))
            if not isfile(filename):
                joblib.dump(result, filename, compress=3)
                return
            counter += 1

    def flush(self):
        while True:
            filename = join(self._get_folder(), "{}.pkl".format(self._flush_counter))
            if not isfile(filename):
                joblib.dump(self._unflushed, filename, compress=3)
                self._unflushed = list()
                return
            self._flush_counter += 1        

    def __del__(self):
        self.flush()
