import time
import sys
from os.path import isfile, join
import numpy as np
import pandas as pd
from data_keeper import get_data_keeper
from generate_subsets import SubsetGenerator
from wrappers import SubsetGeneratorWrapper

RAW_X_BEFORE_SUBSET_GENERATION_PATH = "raw_X_before_subsets_generation.csv"
POSSIBLE_COMPLEX_FEATURES_PATH = "possible_complex_features.txt"
get_generator_result = None


def make_new_generator():
    start_time = time.time()
    X = get_data_keeper().get_common_x()
    print "matrix shape before:", X.shape
    X[X!=1] = 0
    to_drop = (X.sum(axis=0) >= (X.shape[0] / 2)) | (X.sum(axis=0) < 3)
    to_drop = to_drop[to_drop].index
    X = X.drop(to_drop, axis=1)
    X.to_csv(RAW_X_BEFORE_SUBSET_GENERATION_PATH)
    print "matrix shape after:", X.shape
    sys.stdout.flush()
    generator = SubsetGeneratorWrapper()
    generator.generate_and_set(X.as_matrix().astype(np.uint8))
    print "generating done, time from start spent:", time.time() - start_time
    generator.store(POSSIBLE_COMPLEX_FEATURES_PATH)
    print "storing done, time from start spent:", time.time() - start_time
    return generator, X


def get_ready_generator_inner(compute_if_not_found=True, folder=None):
    global get_generator_result
    if get_generator_result is None:
        if folder is None:
            raw_X_before_subsets_generation_path = RAW_X_BEFORE_SUBSET_GENERATION_PATH
            possible_complex_features_path = POSSIBLE_COMPLEX_FEATURES_PATH
        else:
            raw_X_before_subsets_generation_path = join(folder, RAW_X_BEFORE_SUBSET_GENERATION_PATH)
            possible_complex_features_path = join(folder, POSSIBLE_COMPLEX_FEATURES_PATH)
        if isfile(raw_X_before_subsets_generation_path) and isfile(possible_complex_features_path):
            generator = SubsetGenerator()
            generator.load(possible_complex_features_path)
            X = pd.read_csv(raw_X_before_subsets_generation_path, index_col=0)
            generator.set_raw_matrix(X.as_matrix().astype(np.uint8))
            get_generator_result = generator, X
        else:
            if compute_if_not_found:
                get_generator_result = make_new_generator()
    return get_generator_result


class GeneratorGetter:
    def __call__(self):
        return get_generator_result[0]


def get_ready_generator(*args, **kwargs):
    generator, X = get_ready_generator_inner(*args, **kwargs)
    return SubsetGeneratorWrapper(GeneratorGetter()), X


__all__ = ['get_ready_generator']
