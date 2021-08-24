# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:40:57 2021

@author: Maxsc
"""

import pandas as pd

input_path = "C:\\Users\\Maxsc\\Documents\\01DATA\\TUBB\\MT-fluctuations"
input_path = "E:\\TUBB\\MT-fluctuations\\MT-fluctuations"
input_path = "C:\\Users\\Maxsc\\Documents\\data_tmp\\MT-fluctuations"
file_name = "MT-fluctuations_small_singlebranch.csv"
file_name = file_name.replace(".csv","_same-time.csv")
file_name = file_name.replace(".csv","_analysis.csv")
file_name = "MT-fluctuations_small_singlebranch_neurite-norm_fluctuations.csv"
file_name = "MT-fluctuations_small_singlebranch_same-time_analysis.csv"
file_name = "MT-fluctuations_small_singlebranch_neuron-norm_fluctuations.csv"


data = pd.read_csv(input_path+"\\"+file_name)

data_averaged = data.groupby(["date", "neuron", "cycle_threshold", "treatment"]).mean()

neuron_averages_file_name = file_name.replace(".csv", "_average.csv")

data_averaged.to_csv(input_path + "\\" + neuron_averages_file_name)
print(input_path + "\\" + neuron_averages_file_name)