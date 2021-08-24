# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 08:03:05 2021

@author: Maxsc
"""


import numpy as np
import pandas as pd
import os
import copy
import seaborn as sb
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from scipy import stats

def convert_data(data, channels_to_type):
    for channel, data_type  in channels_to_type.items():
        data[channel] = data[channel].astype(data_type)
    return data

def convert_channel_to_int(data):
    #convert channel to int
    data["channel"] = data["channel"].astype(str)
    data["channel"] = data["channel"].str.replace("c0001","1")
    data["channel"] = data["channel"].str.replace("c0000","0")
    data["channel"] = data["channel"].astype(int)
    return data

def exclude_data_after_frame(data, max_frame):
    
    data = data.loc[data["time"] < max_frame]

    return data

def select_branches_with_most_frames(all_data, min_nb_frames,
                                     neurite_columns, branch_column):
    
    #for assertion check to make sure that no origins are lost
    nb_origins_before = len(all_data[neurite_columns].drop_duplicates())
    
    branch_columns = copy.copy(neurite_columns)
    branch_columns.append("branch")
    
    #count the number of frames that each branch occured
    nb_frames_per_branch = all_data.groupby(branch_columns)["avInt"].count().reset_index()
    
    #get the ids of rows within each origin present at the highest number of frames
    idx_branch_present_longest = nb_frames_per_branch.groupby(neurite_columns, 
                                                              sort=False)["avInt"].idxmax()
    
    #get the branches that were present for the most timeframes
    branches_longest = nb_frames_per_branch.loc[idx_branch_present_longest]
    
    #exclude all branches and thereby origins that were not present for min_nb_frames
    brief_branches_idxs = branches_longest["avInt"] >= min_nb_frames
    branches_longest_min_frames = branches_longest.loc[brief_branches_idxs]
    
    #get the values of columns needed to identify data from one branch
    #will be one index for each origin (for one corresponding branch)
    branches_longest_idx = branches_longest_min_frames [branch_columns].values
    

    #set index, then sort to speed up upcoming look ups
    all_data.set_index(branch_columns, inplace=True)
    all_data.sort_index(inplace=True)
    
    #go through each index of a origin-branch combination
    all_data_one_branch  = pd.DataFrame()
    for branch_longest_idx in branches_longest_idx:
        new_data = all_data.loc[tuple(branch_longest_idx)]
        all_data_one_branch = pd.concat([all_data_one_branch, new_data])


    nb_origins_after = len(all_data_one_branch.reset_index()[neurite_columns].drop_duplicates())
    print(nb_origins_before, nb_origins_after)

    return all_data_one_branch


#also use channel as neurite column to determine the corect
#number of frames that each channel is present
neurite_columns = ["date","neuron","origin", "channel"]
branch_column = "branch"


input_path = "C:\\Users\\Maxsc\\Documents\\01DATA\\TUBB\\MT-fluctuations"
input_path = "E:\\TUBB\\MT-fluctuations\\MT-fluctuations"
input_path = "C:\\Users\\Maxsc\\Documents\\data_tmp\\MT-fluctuations"
file_name = "MT-fluctuations_small.csv"
file_name = "MT-fluctuations_small_withInts.csv"
final_file_name = file_name.replace(".csv", "_singlebranch.csv")

minimum_nb_frames = 40
max_timeframe = 81

all_data = pd.read_csv(input_path+"\\"+file_name)

channels_to_type = {}
channels_to_type["date"] = int
channels_to_type["time"] = int
channels_to_type["avInt"] = float

all_data = convert_data(all_data, channels_to_type)

all_data = convert_channel_to_int(all_data)

all_data = exclude_data_after_frame(all_data, max_timeframe)

all_data_one_branch = select_branches_with_most_frames(all_data, 
                                                       minimum_nb_frames,
                                                       neurite_columns, 
                                                       branch_column)

all_data_one_branch.to_csv(os.path.join(input_path, final_file_name))