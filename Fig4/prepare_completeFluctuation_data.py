# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 21:37:11 2021

@author: Maxsc
"""

import pandas as pd
import os

input_path = os.path.abspath("C:\\Users\\Maxsc\\Nextcloud\\06PAPERS\\MTRF\\Figures\\Figure4\\")
input_path = "C:\\Users\\Maxsc\\Documents\\data_tmp\\MT-fluctuations"
file_name = "MT-fluctuations.csv"
final_file_name = file_name.replace(".csv", "_small_withInts.csv")

columns_to_delete = ["local_mass_array", 
                     "diameter_array", 
                     "preAW", 
                     "maxInt",
                     "gain_y",
                     "gain_x",
                     "loss_x",
                     "loss_y",
                     "x",
                     "y"]

data = pd.read_csv(os.path.join(input_path, file_name))

data.drop(labels=columns_to_delete, axis=1, inplace=True)

data.to_csv(os.path.join(input_path, final_file_name))