# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 22:13:22 2019

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


def smoothen(row,new_growth_data,radius,column):
    #calculate smooth growth by averaging the last three growth rates
    start = int(row.name)-radius
    end = int(row.name)+radius+1
    smooth_value = new_growth_data.iloc[start:end][column].mean()
#    print(smooth_value)
    return smooth_value

def classify_stages_from_axon_formation(all_data, outgrowth_time_data):
    
    #for each neuron, add information 
    #about at which time its in stage 2 and in which time its in stage 3
    outgrowth_idx = [tuple(idx) for idx in outgrowth_time_data.index.values]
    outgrowth_idx = set(outgrowth_idx)
    
    all_neurons = all_data[neuron_cols].drop_duplicates()
    all_data.set_index(neuron_cols, inplace=True)
    
    all_data["stage"] = np.nan
    
    #classify stages based on when the axon grew out
    for neuron in all_neurons.values:
        if tuple(neuron) in outgrowth_idx:
            growth_time = int(outgrowth_time_data.loc[tuple(neuron)])
            growth_time_st3 = growth_time
            growth_time_st2 = growth_time - 60
            neuron_data = all_data.loc[tuple(neuron)]
            neuron_data.loc[neuron_data["time"] > growth_time_st3, "stage"] = 3
            neuron_data.loc[neuron_data["time"] < growth_time_st2, "stage"] = 2
    
    
    all_data = all_data.dropna(subset=["stage"])

    all_data.reset_index(inplace=True)
    
    return all_data

def analyze_fluctuations(all_data, int_column, unit_columns, all_thresholds,
                                  timeframes_smoothened,
                                  background_val, min_nb_vals=40):
    
    data = pd.DataFrame(columns=("date", "neuron","treatment",
                                 "origin", "branch",
                                 "frames", "cycles",
                                 "cycles/h", "cycle_amplitude",
                                 "pos_amplitude", "neg_amplitude",
                                 "cycle_threshold", 
                                 ))

    # all_data['int_norm'] = np.nan
    all_data['avInt_smooth'] = np.nan
    all_units = all_data[unit_cols].drop_duplicates()
    all_data.set_index(unit_columns, inplace=True)
    nb = 0
    for one_index in all_units.values:
        one_index = tuple(one_index)
        one_branch = all_data.loc[one_index].reset_index()
        
        treatment = one_branch['treatment'].iloc[0]
        origin = one_branch['origin'].iloc[0]
        branch = one_branch['branch'].iloc[0]
        #subtract background
        
        #apply rolling average over 3 timeframes to MT intensity course
        int_smooth = one_branch.apply(smoothen,
                                      axis=1,
                                      args=(one_branch,timeframes_smoothened,
                                            int_column))
        
        all_data.loc[one_index,"avInt_smooth"] = np.array(int_smooth)
        
        one_branch = all_data.loc[one_index ]
    
        max_time = max(one_branch['time'])
        if len(one_branch) >= min_nb_vals:
            print(one_index)
            # ints_norm = one_branch['avInt_smooth'] / one_branch_mean_int
            # all_data.loc[one_index,'int_norm'] = ints_norm
            one_branch = all_data.loc[one_index]
            
            current_pos = 0
            mean_ints = np.array(one_branch["int_norm"])
            if len(mean_ints) > 0:
                #get all local maxima & minima
                all_maxima = argrelextrema(mean_ints,np.greater)[0]
                
                all_minima = argrelextrema(mean_ints,np.less)[0]
                if (len(all_minima) > 0) & (len(all_maxima) > 0):
                    for point_nb, val in enumerate(mean_ints):
                        if not np.isnan(val):
                            first_point = point_nb
                            break
                    #add first and last point to maxima or minima list 
                    #(depending on whether it is bigger or smaller 
                    #than later of prev value)
                    if ((first_point not in all_minima) & 
                        (first_point not in all_maxima)):
                        if mean_ints[first_point] > mean_ints[first_point+1]:
                            all_maxima = np.insert(all_maxima,0,first_point)
                        else:
                            all_minima = np.insert(all_minima,0,first_point)
                    last_index = len(mean_ints) -1
                    if ((last_index not in all_minima) & 
                        (last_index not in all_maxima)):
                        if mean_ints[last_index] > mean_ints[last_index-1]:
                            all_maxima = np.append(all_maxima,last_index)
                        else:
                            all_minima = np.append(all_minima,last_index)
                    
                    start_point = min(min(all_maxima),min(all_minima))
                    if start_point in all_maxima:
                        rel_ex_type = "max"
                    else:
                        rel_ex_type = "min"
                        
                    for min_diff_for_cycle in all_thresholds:
                        #add start_point to all fluct points
                        #since all fluct points always contain the 
                        #start point and the end point of each fluctuation
                        all_fluct_points = [start_point]
                        all_flucts = []
                        #only continue, if minimum before was found 
                        #(otherwise maximum could not be chained 
                        # to last minimum)
                        results = move_further(start_point,
                                               all_flucts, 
                                               all_minima, all_maxima, 
                                               all_fluct_points, mean_ints, 
                                               rel_ex_type, min_diff_for_cycle)
                        current_point, all_flucts, all_fluct_points = results
                        nb_of_cycles = max(len(all_flucts)/2,0)
                        cycle_amplitude = np.mean(all_flucts)
                        
                        #check that values from all fluct_points are 
                        #similar to values of all fluct
                        fluct_points_int = mean_ints[all_fluct_points]
                        fluct_points_diff = np.abs(fluct_points_int[1:] - fluct_points_int[:-1])
                        assert np.all(np.isclose(fluct_points_diff, all_flucts))

                        fluct_points_diff = fluct_points_int[1:]-fluct_points_int[:-1]
                        pos_amplitudes_idxs = np.where(fluct_points_diff > 0)
                        pos_amplitudes = fluct_points_diff[pos_amplitudes_idxs]
                        neg_amplitudes_idxs = np.where(fluct_points_diff < 0)
                        neg_amplitudes = fluct_points_diff[neg_amplitudes_idxs]
                    
                        new_row = []
                        new_row.append(one_index[0])
                        new_row.append(one_index[1])
                        new_row.append(treatment)
                        new_row.append(origin)
                        new_row.append(branch)
                        new_row.append(max_time)
                        new_row.append(nb_of_cycles)
                        new_row.append(nb_of_cycles/(max_time/60))
                        new_row.append(cycle_amplitude)
                        new_row.append(np.mean(pos_amplitudes))
                        new_row.append(np.mean(neg_amplitudes))
                        new_row.append(min_diff_for_cycle)
            
                        data.loc[len(data)] = new_row
                        
                        nb += 1
    return data


def move_further(current_point, all_flucts,
                 all_minima, all_maxima, 
                 all_fluct_points, mean_ints,
                 rel_ex_type, min_diff_for_cycle):
    
    #if the last point was a maximum, get all minima after that
    if rel_ex_type == "max":
        points_after = all_minima[np.where(all_minima > current_point)]
    else:
        #if the last point was a minimum, get all maxima after that
        points_after = all_maxima[np.where(all_maxima > current_point)]

    point_val = mean_ints[current_point]
    for next_point in points_after:
        next_point_val = mean_ints[next_point]
        #for the first position (no fluctuation yet)
        #choose a new starting position if a larger value is found (max)
        #or a smaller value is found  (min)
        #before the first half cycle for a fluctuation is found
        if len(all_flucts) == 0:
            if rel_ex_type == "max":
                if next_point_val > point_val:
                    rel_ex_type = "min"
                    curr_point_idx = np.where(all_minima > current_point)[0][0]
                    current_point = all_minima[curr_point_idx]
                    all_fluct_points = [current_point]
                    results = move_further(current_point,
                                            all_flucts, all_minima, all_maxima,
                                            all_fluct_points,
                                            mean_ints,
                                            rel_ex_type, min_diff_for_cycle)
                    current_point, all_flucts, all_fluct_points = results
                    break
            elif next_point_val < point_val:
                rel_ex_type = "max"
                curr_point_idx = np.where(all_maxima > current_point)[0][0]
                current_point = all_maxima[curr_point_idx]
                all_fluct_points = [current_point]
                results = move_further(current_point,
                                        all_flucts, all_minima, all_maxima,
                                        all_fluct_points,
                                        mean_ints, rel_ex_type,
                                        min_diff_for_cycle)
                current_point, all_flucts, all_fluct_points = results
                break
        #check points between for lower or higher values than current extrema
        #should be also done if the first point is defined already?
        if rel_ex_type == "max":
            points_between_idxs = np.where((all_maxima > current_point) & 
                                            (all_maxima < next_point))
            points_between = all_maxima[points_between_idxs]
            if len(points_between) > 0:
                points_between_vals = mean_ints[points_between]
                alt_point_idx = np.where(points_between_vals == 
                                          max(points_between_vals))[0][0]
                alternative_point = points_between[alt_point_idx]
                if max(points_between_vals) > point_val:
                    if len(all_flucts) > 0:
                        #take the startpoint from the last fluctuation
                        #since the last fluctuation value will be updated
                        int_last_fluct_point = mean_ints[all_fluct_points[-2]]
                        new_int_diff = abs(mean_ints[alternative_point] - 
                                        int_last_fluct_point)
                        #relative int diff NOT desireable
                        #since then the same change up and then down again
                        #wont lead to the value it started with!
                        # new_rel_int_diff = new_int_diff / int_last_fluct_point
                        all_flucts[-1] = new_int_diff
                    #get a new point_val for the current point
                    #if the new current point does not lead to a new fluctuation
                    #the point_val is then not updated and a wrong point_val
                    #will be used to check for higher point in between
                    #in next iteration
                    current_point = alternative_point
                    point_val = mean_ints[current_point]
                    #also update the corresponding point in all_fluct_points
                    all_fluct_points[-1] = current_point
                    
        if rel_ex_type == "min":
            points_between_idxs = np.where((all_minima > current_point) & 
                                    (all_minima < next_point))
            points_between = all_minima[points_between_idxs]
            if len(points_between) > 0:
                points_between_vals = mean_ints[points_between]
                alt_point_idx = np.where(points_between_vals == 
                                          min(points_between_vals))[0][0]
                alternative_point = points_between[alt_point_idx]
                if min(points_between_vals) < point_val:
                    if len(all_flucts) > 0:
                        #take the startpoint from the last fluctuation
                        #since the last fluctuation value will be updated
                        int_last_fluct_point = mean_ints[all_fluct_points[-2]]
                        new_int_diff = abs(mean_ints[alternative_point] - 
                                        int_last_fluct_point)
                        # new_rel_int_diff = new_int_diff / int_last_fluct_point
                        all_flucts[-1] = new_int_diff
                        
                    current_point = alternative_point
                    point_val = mean_ints[current_point]
                    #also update the corresponding point in all_fluct_points
                    all_fluct_points[-1] = current_point
                
        #get point val of current point again, in case it was changed
        point_val = mean_ints[current_point]
        next_point_val = mean_ints[next_point]
        
        #calculate absolute change
        diff = abs(point_val - next_point_val)
        
        if (diff > min_diff_for_cycle) :
            
            current_point = next_point
            
            all_flucts.append(diff)
            
            all_fluct_points.append(next_point)
            
            #switch from max to min or vice versa
            if rel_ex_type == "max":
                rel_ex_type = "min"
            else:
                rel_ex_type = "max"
                
            results = move_further(current_point, 
                                   all_flucts, all_minima, all_maxima,
                                   all_fluct_points, mean_ints,
                                   rel_ex_type, min_diff_for_cycle)
            current_point, all_flucts, all_fluct_points = results
            break
    return current_point, all_flucts, all_fluct_points


input_path = os.path.abspath("C:\\Users\\Maxsc\\Documents\\"
                             "data_tmp\\MT-fluctuations")
file_name = "MT-fluctuations_small_singlebranch.csv"


normalization = "neuron"
int_column = "avInt"
min_per_frame = 3

all_data = pd.read_csv(os.path.join(input_path, file_name))

#VERY important to exclude timepoints after 100 for experiment 190426 - those were images with hours of imaging gap!
all_data = all_data.loc[(all_data['date'] != 190426) | (all_data['time'] < 101)]
all_data = all_data.loc[all_data['time'] < 82]

all_data["channel"] = all_data["channel"].astype(int)

all_data = all_data.loc[all_data['channel'] == 0]

all_data['time'] = all_data['time'] * min_per_frame

neuron_cols = ["date", "neuron"]
neurite_cols = [*neuron_cols, "origin"]

unit_cols = neurite_cols

timeframes_smoothened = 2
all_fluct_thresholds = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
background_val = 115


#----------------------- change file name according to type of normalization--------------
if normalization == "neuron":
    file_name_addition = "_neuron-norm"
    norm_cols = neuron_cols
elif normalization == "neurite":
    file_name_addition = "_neurite-norm"
    norm_cols = neurite_cols
file_name = file_name.replace(".csv",file_name_addition+".csv")

all_data.loc[:,int_column] = all_data[int_column] - background_val
        
#normalize intensities based on neuron or neurite
ints_norm = all_data.groupby(norm_cols)[int_column].transform(lambda x: x / x.mean())
all_data.loc[:,"int_norm"] = ints_norm

fluct_data = analyze_fluctuations(all_data, "int_norm", unit_cols, 
                                  all_fluct_thresholds,
                                  timeframes_smoothened,
                                  background_val)

file_name = file_name.replace(".csv","_fluctuations.csv")
data_path = os.path.join(input_path, file_name)
fluct_data.to_csv(data_path)