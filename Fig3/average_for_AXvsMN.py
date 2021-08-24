# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 21:47:36 2021

@author: Maxsc
"""

import pandas as pd
import os

input_path = "C:\\Users\\Maxsc\\Nextcloud\\01ANALYSIS\\TUBB\\MTRF-longterm\\"
input_path = os.path.abspath("C:\\Users\\Maxsc\\Nextcloud\\01ANALYSIS\\"
                             "TUBB\\MTRF-fluctuations-longterm_AXvsMN\\")
file_name = "MT-RF_growth_longterm_allData_fluctuations.csv"
file_name = "MT-RF_growth_longterm_allData_neuron-norm_fluctuations_MTRF.csv"

data = pd.read_csv(os.path.join(input_path, file_name))

data_averaged = data.groupby(["date", "neuron", "stage", "axon", "cycle_threshold"]).mean()

data_averaged.to_csv(os.path.join(input_path, file_name.replace(".csv", "_averaged.csv")))
