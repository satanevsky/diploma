import sys
from os import devnull
import pandas as pd
import numpy as np


class DataKeeper(object):
    def __init__(self):
        stderr = sys.stderr
        try:
            sys.stderr = open(devnull, "w")
            matrix = pd.read_csv('apr15.snps.matrix', sep='\t').drop([0, 1], axis=0).set_index('#snp_pos')
            self._common_x = matrix.replace('-', -1).astype(np.int32)
            self._drug_results = pd.read_csv('drugs_effect.csv').set_index('Strain')
        finally:
            sys.stderr = stderr

    def _get_y(self, drug_name):
        return self._drug_results[drug_name].dropna() == 2

    def get_common_x(self):
        return self._common_x

    def get_possible_first_level_drugs(self):
        return [u'ETHA: Ethambutol ',
                u'ISON: Isoniazid ',
                u'RIFM: rifampicin ',
                u'RIFP: Rifapentine ',
                u'PYRA: Pyrazinamide ',
                u'STRE: Streptomycin ',
                u'CYCL: Cycloserine ']

    def get_possible_second_level_drugs(self):
        return [u' Kanamycin ',
                u'ETHI: Ethionamide/ Prothionamide ',
                u'PARA: Para-aminosalicyclic acid ',
                u'CAPR: Capreomycin ',
                u'AMIK: Amikacin',
                u'OFLO: Ofloxacin ']

    def get_possible_drugs(self):
        return self.get_possible_first_level_drugs() + self.get_possible_second_level_drugs()

    def get_train_data(self, drug_name, as_indexes=False):
        if not as_indexes:
            result_all = self.get_common_x().join(self._get_y(drug_name), how='inner')
            result_x, result_y = result_all.drop(drug_name, axis=1), result_all[drug_name]
            return result_x, result_y
        else:
            X, y = self.get_train_data(drug_name, as_indexes=False)
            possible_objects = set(y.index)
            indexes = list()
            y_refactored = list()

            for i, el in enumerate(self.get_objects_names()):
                if el in possible_objects:
                    indexes.append(i)
                    y_refactored.append(y[el])
            y_refactored = np.array(y_refactored)
            indexes = np.array(indexes)
            indexes = indexes.reshape((indexes.shape[0], 1))
            return indexes, y_refactored


    def set_common_x(self, common_x):
        self._common_x = common_x

    def get_objects_names(self):
        objects_names = list(self.get_common_x().index)
        return objects_names

    def get_object_name_by_index(self, index):
        return self.get_objects_names()[index]

    def get_feature_names(self):
        return list(self.get_common_x().columns.values)

    def get_feature_name_by_index(self, index):
        return self.get_feature_names()[index]


DATA_KEEPER = DataKeeper()
def get_data_keeper():
    return DATA_KEEPER


__all__ = ['get_data_keeper']
