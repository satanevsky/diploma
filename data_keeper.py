import sys
from os import devnull
import pandas as pd
import numpy as np


class DataKeeper(object):
    def __init__(self):
        stderr = sys.stderr
        try:
            sys.stderr = open(devnull, "w")
            self._matrix = pd.read_csv('apr15.snps.matrix', sep='\t').drop([0, 1], axis=0).set_index('#snp_pos')
            self._drug_results = pd.read_csv('drugs_effect.csv').set_index('Strain')
        finally:
            sys.stderr = stderr

    def _get_y(self, drug_name):
        return self._drug_results[drug_name].dropna() == 2

    def get_common_x(self):
        return self._matrix.replace('-', -1).astype(np.int32)

    def get_possible_first_level_drugs(self):
        return [u'ETHA: Ethambutol ',
                u'ISON: Isoniazid ',
                u'RIFM: rifampicin ',
                u'RIFP: Rifapentine ',
                u'PYRA: Pyrazinamide ',
                u'STRE: Streptomycin ',
                u'CYCL: Cycloserine ']
    def get_possible_second_level_drugs(self):
        return [u'ETHI: Ethionamide/ Prothionamide ',
                u'PARA: Para-aminosalicyclic acid ',
                u'AMIK: Amikacin',
                u' Kanamycin ',
                u'CAPR: Capreomycin ',
                u'OFLO: Ofloxacin ']

    def get_possible_drugs(self):
        return self.get_possible_first_level_drugs() + self.get_possible_second_level_drugs()

    def get_train_data(self, drug_name):
        result_all = self.get_common_x().join(self._get_y(drug_name), how='inner')
        result_x, result_y = result_all.drop(drug_name, axis=1), result_all[drug_name]
        return result_x, result_y


DATA_KEEPER = DataKeeper()
def get_data_keeper():
    return DATA_KEEPER


__all__ = ['get_data_keeper']
