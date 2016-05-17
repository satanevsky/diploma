import sys
from os import mkdir
from os.path import isfile, isdir, join
from collections import defaultdict
import cPickle as pickle


class ResultsDumper(object):
    def __init__(self, experiment_name):
        self._experiment_name = experiment_name
        self._unflushed = list()
        self._flush_counter = 0
        self._results_counter = 0
        if not isdir(self._get_folder()):
            mkdir(self._get_folder())

    def _get_folder(self):
        return join("experiments_results", self._experiment_name)

    def add_result(self, result):
        self._unflushed.append(result)
        self._results_counter += 1
        with open(join(self._get_folder(), "counter.txt"), 'a') as f:
            f.write("{}\n".format(self._results_counter))
        if len(self._unflushed) >= 50:
            self.flush()

    def get_logs_file(self):
        filename = join(self._get_folder(), "execution.log")
        return open(filename, 'a')

    def dump_final_result(self, result):
        counter = 0
        while True:
            filename = join(self._get_folder(), "final_results_{}.pkl".format(counter))
            if not isfile(filename):
                with open(filename, 'wb') as f:
                    pickle.dump(result, f)
                return
            counter += 1

    def flush(self):
        while True:
            filename = join(self._get_folder(), "{}.pkl".format(self._flush_counter))
            if not isfile(filename):
                with open(filename, 'wb') as f:
                    pickle.dump(self._unflushed, f)
                self._unflushed = list()
                return
            self._flush_counter += 1        

    def __del__(self):
        self.flush()
