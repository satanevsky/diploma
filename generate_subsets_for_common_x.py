import time
import sys
from os.path import isfile
import numpy as np
import pandas as pd
from data_keeper import get_data_keeper
from generate_subsets import SubsetGenerator


RAW_X_BEFORE_SUBSETE_GENERATION_PATH = "raw_X_before_subsets_generation.csv"
POSSIBLE_COMPLEX_FEATURES_PATH = "possible_complex_features.txt"


def make_new_generator():
    start_time = time.time()
    X = get_data_keeper().get_common_x()
    print "matrix shape before:", X.shape
    X[X!=1] = 0
    to_drop = (X.sum(axis=0) >= (X.shape[0] / 2)) | (X.sum(axis=0) < 3)
    to_drop = to_drop[to_drop].index
    X = X.drop(to_drop, axis=1)
    X.to_csv(RAW_X_BEFORE_SUBSETE_GENERATION_PATH)
    print "matrix shape after:", X.shape
    sys.stdout.flush()
    generator = SubsetGenerator()
    generator.generate_and_set(X.as_matrix().astype(np.uint8))
    print "generating done, time from start spent:", time.time() - start_time
    generator.store(POSSIBLE_COMPLEX_FEATURES_PATH)
    print "storing done, time from start spent:", time.time() - start_time
    return generator, X


def get_ready_generator():
    if isfile(RAW_X_BEFORE_SUBSETE_GENERATION_PATH) and isfile(POSSIBLE_COMPLEX_FEATURES_PATH):
        generator = SubsetGenerator()
        generator.load(RAW_X_BEFORE_SUBSETE_GENERATION_PATH)
        X = pd.read_csv(RAW_X_BEFORE_SUBSETE_GENERATION_PATH)
        return generator, X
    else:
        return make_new_generator()


__all__ = ['get_ready_generator']
