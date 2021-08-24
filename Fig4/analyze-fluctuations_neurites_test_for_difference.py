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
    smooth_value = new_growth_data.iloc[int(row.name)-radius:int(row.name)+radius+1][column].mean()
#    print(smooth_value)
    return smooth_value


input_path = "C:\\Users\\Maxsc\\Documents\\01DATA\\TUBB\\MT-fluctuations"
input_path = "E:\\TUBB\\MT-fluctuations\\MT-fluctuations"
input_path = "C:\\Users\\Maxsc\\Documents\\data_tmp\\MT-fluctuations"
file_name = "MT-fluctuations_small_singlebranch.csv"



lineplot_path = input_path+"\\lineplots\\"

if not os.path.exists(lineplot_path):
    os.mkdir(lineplot_path)

min_per_frame = 3
#sb.set(font_scale=3)
sb.set(font_scale=3,rc={'axes.facecolor':'black', 'grid.color': '1', 'figure.facecolor':'black', 'text.color': '1', 'xtick.color': '1','ytick.color': '1','axes.labelcolor': '1'})


normalization = "neurites"

all_data = pd.read_csv(input_path+"\\"+file_name)


file_name = file_name.replace(".csv","_same-time.csv")

UMperPX = 0.22
minutes_per_frame = 3

#VERY important to exclude timepoints after 100 for experiment 190426 - those were images with hours of imaging gap!
all_data = all_data.loc[(all_data['date'] != 190426) | (all_data['time'] < 101)]
all_data = all_data.loc[all_data['time'] < 82]

all_data["channel"] = all_data["channel"].astype(int)

all_data = all_data.loc[all_data['channel'] == 0]

# test_data_blebb = all_data.loc[all_data["treatment"] == "pablebb"]
# all_cells_blebb = test_data_blebb[["date", "neuron"]].drop_duplicates()
# print(all_cells_blebb)

all_data['int_norm'] = np.nan
all_data['avInt_smooth'] = np.nan


#test specific cell
# all_data.set_index(["date", "neuron"], inplace=True)
# index = (190426, "cell0055_registered")
# all_data = all_data.loc[index].reset_index()


all_indices = all_data[["date","neuron","origin","branch"]].drop_duplicates()

#exclude timepoints with big fluctuations in length?

timeframes_smoothened = 2

def move_further(current_point, all_flucts, all_minima, all_maxima,
                 all_fluct_points,mean_ints,
                 rel_ex_type,min_diff_for_cycle):
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



def get_smooth_growth(row,new_growth_data,radius,column):
    #calculate smooth growth by averaging the last three growth rates
#    print(new_growth_data.iloc[int(row.name)-radius:int(row.name)+1][['time',column]])
    return new_growth_data.iloc[int(row.name)-radius:int(row.name)+1][column].mean()



def remove_outlier_raw_growth(row,new_growth_data,outlier_size):
    outlier = 0
    #check which position is the outlier
    for pos in range(-(outlier_size-1),outlier_size):
        if (pos != 0) & ((int(row.name)+pos) >= 0) & ((int(row.name)+pos) < len(new_growth_data)):
            av_growth = (new_growth_data.iloc[int(row.name)+pos]['growth_raw'] + new_growth_data.iloc[int(row.name)]['growth_raw'])/2
            growth_difference = abs((av_growth - row['growth_raw']) / av_growth)
#            if abs(new_growth_data.iloc[int(row.name)]['growth_raw']) > 15:
#                print("orig: {}, compare: {}, diff: {}".format(new_growth_data.iloc[int(row.name)]['growth_raw'],new_growth_data.iloc[int(row.name)+pos]['growth_raw'],growth_difference))
            if growth_difference > 6:
                outlier = pos
                break
    if outlier == 0:
        return row['growth_raw']
    else:
        #if an outlier was found (opposite growth nearby), return average growth over whole time as growth
        return (new_growth_data.iloc[int(row.name)+pos]['growth_raw'] + new_growth_data.iloc[int(row.name)]['growth_raw'])/(abs(pos)+1)
        

def calculate_growth(data):
    #calculate neurite growth at each time
    data = data.sort_values(axis=0,by=["date","neuron","origin","branch","time"])
    print("calculating growth...")
    data['length_um'] = data['length'] * UMperPX
#    data['growh_rate_raw'] = np.nan
    all_neurites = data[['date','neuron','origin','branch']].drop_duplicates()
    for neurite_nb in all_neurites.values:
        print("neurite: {}".format(neurite_nb))
        oneNeurite = data.loc[(data['date'] == neurite_nb[0]) & (data['neuron'] == neurite_nb[1]) & (data['origin'] == neurite_nb[2]) & (data['branch'] == neurite_nb[3])]
        baseDetails = []
        baseDetails.append(oneNeurite['date'])
        all_lengths = np.array(oneNeurite['length_um'])
        all_growth = all_lengths[1:len(all_lengths)] - all_lengths[0:len(all_lengths)-1]
        all_growth = np.insert(all_growth,0,np.nan)
        data.loc[(data['date'] == neurite_nb[0]) & (data['neuron'] == neurite_nb[1]) & (data['origin'] == neurite_nb[2]) & (data['branch'] == neurite_nb[3]),'growth_raw'] = all_growth
        
        #check if smoothened growth deviates a lot from raw growth, if so remove this outlier (set raw growth as smooth growth)
        oneNeurite = data.loc[(data['date'] == neurite_nb[0]) & (data['neuron'] == neurite_nb[1]) & (data['origin'] == neurite_nb[2]) & (data['branch'] == neurite_nb[3])].reset_index()
        outlier_size = 2
        
        data.loc[(data['date'] == neurite_nb[0]) & (data['neuron'] == neurite_nb[1]) & (data['origin'] == neurite_nb[2]) & (data['branch'] == neurite_nb[3]),'growth_raw_corr'] = np.array(oneNeurite.apply(remove_outlier_raw_growth,axis=1,args=(oneNeurite,outlier_size)))
        oneNeurite = data.loc[(data['date'] == neurite_nb[0]) & (data['neuron'] == neurite_nb[1]) & (data['origin'] == neurite_nb[2]) & (data['branch'] == neurite_nb[3])].reset_index()
        
        all_times = np.array(oneNeurite['time'])
        all_dT = all_times[1:len(all_times)] - all_times[0:len(all_times)-1]
        all_dT = all_dT.astype(float)
        all_dT = np.insert(all_dT,0,np.nan)
        all_dT = all_dT * minutes_per_frame
        data.loc[(data['date'] == neurite_nb[0]) & (data['neuron'] == neurite_nb[1]) & (data['origin'] == neurite_nb[2]) & (data['branch'] == neurite_nb[3]),'dT'] = all_dT
        oneNeurite = data.loc[(data['date'] == neurite_nb[0]) & (data['neuron'] == neurite_nb[1]) & (data['origin'] == neurite_nb[2]) & (data['branch'] == neurite_nb[3])]
        
        data.loc[(data['date'] == neurite_nb[0]) & (data['neuron'] == neurite_nb[1]) & (data['origin'] == neurite_nb[2]) & (data['branch'] == neurite_nb[3]),'growth_rate_raw_corr'] = (oneNeurite['growth_raw_corr'] / oneNeurite['dT'])
        oneNeurite = data.loc[(data['date'] == neurite_nb[0]) & (data['neuron'] == neurite_nb[1]) & (data['origin'] == neurite_nb[2]) & (data['branch'] == neurite_nb[3])].reset_index()
        radius = 1
        data.loc[(data['date'] == neurite_nb[0]) & (data['neuron'] == neurite_nb[1]) & (data['origin'] == neurite_nb[2]) & (data['branch'] == neurite_nb[3]),'growth_rate'] = np.array(oneNeurite.apply(get_smooth_growth,axis=1,args=(oneNeurite,radius,'growth_rate_raw_corr')))
    return data

data = pd.DataFrame(columns=("date","neuron","treatment","origin","branch","frames","cycles","cycles/h","cycle_amplitude","cycle_threshold"))
all_thresholds = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

background_val = 115


# all_data = calculate_growth(all_data)

# all_data.to_csv(input_path+"\\"+file_name.replace(".csv","_growth.csv"))


# all_data = pd.read_csv(input_path+"\\"+file_name.replace(".csv","_growth.csv"))
all_data['time'] = all_data['time'] * min_per_frame
# all_data.to_csv(input_path+"\\"+file_name.replace(".csv","_growth.csv"))

# all_data = pd.read_csv(input_path+"\\"+file_name.replace(".csv","_growth.csv"))

#----------------------- change file name according to type of normalization--------------
if normalization == "neuron":
    file_name_addition = "_neuron-norm"
    file_name = file_name.replace(".csv",file_name_addition+".csv")
elif normalization == "neurite":
    file_name_addition = "_neurite-norm"
    file_name = file_name.replace(".csv",file_name_addition+".csv")

neuron_cols = ["date", "neuron"]
neurite_cols = [*neuron_cols, "origin"]

#normalize intensities based on neuron or neurite
norm_cols = neuron_cols
ints_norm = all_data.groupby(norm_cols)["avInt"].transform(lambda x: x / x.mean())
all_data.loc[:,"int_norm"] = ints_norm


nb = 0
for one_index in all_indices.values:
#    one_branch = all_data.loc[(all_data['date'] == one_index[0]) & (all_data['neuron'] == one_index[1]) & (all_data['origin'] == one_index[2]) & (all_data['branch'] == one_index[3])].reset_index()
    one_branch = all_data.loc[(all_data['date'] == one_index[0]) & (all_data['neuron'] == one_index[1]) & (all_data['origin'] == one_index[2]) & (all_data['branch'] == one_index[3]) ].reset_index()
    
    #subtract background
    # one_branch['avInt'] = one_branch['avInt'] - background_val
    
    #apply rolling average over 3 timeframes to MT intensity course
    all_data.loc[(all_data['date'] == one_index[0]) & (all_data['neuron'] == one_index[1]) & (all_data['origin'] == one_index[2]) & (all_data['branch'] == one_index[3]),"avInt_smooth"] = np.array(one_branch.apply(smoothen,axis=1,args=(one_branch,timeframes_smoothened,'int_norm')))
    
    one_branch = all_data.loc[(all_data['date'] == one_index[0]) & (all_data['neuron'] == one_index[1]) & (all_data['origin'] == one_index[2]) & (all_data['branch'] == one_index[3]) ]
    
#    one_branch = one_branch.dropna()
    # one_branch_average_int = one_branch['avInt_smooth'].mean()
    max_time = max(one_branch['time'])
    if len(one_branch) > 39:
        print(one_index)
        # all_data.loc[(all_data['date'] == one_index[0]) & (all_data['neuron'] == one_index[1]) & (all_data['origin'] == one_index[2]) & (all_data['branch'] == one_index[3]),'int_norm'] = one_branch['avInt_smooth'] / one_branch_average_int
        # one_branch = all_data.loc[(all_data['date'] == one_index[0]) & (all_data['neuron'] == one_index[1]) & (all_data['origin'] == one_index[2]) & (all_data['branch'] == one_index[3]) ]
#        print(one_branch[['origin','branch','time','int_norm']])
        treatment = one_branch['treatment'].iloc[0]
        
        current_pos = 0
        mean_ints = np.array(one_branch["int_norm"])
        if len(mean_ints) > 0:
            #get all local maxima & minima
            all_maxima = argrelextrema(mean_ints,np.greater)[0]
            
    #        all_maxima = np.insert(all_maxima,0,0)
            all_minima = argrelextrema(mean_ints,np.less)[0]
            if (len(all_minima) > 0) & (len(all_maxima) > 0):
                for point_nb, val in enumerate(mean_ints):
                    if not np.isnan(val):
                        first_point = point_nb
                        break
                #add first and last point to maxima or minima list (depending on whether it is bigger or smaller than later of prev value)
                if (first_point not in all_minima) & (first_point not in all_maxima):
                    if mean_ints[first_point] > mean_ints[first_point+1]:
                        all_maxima = np.insert(all_maxima,0,first_point)
                    else:
                        all_minima = np.insert(all_minima,0,first_point)
                last_index = len(mean_ints) -1
                if (last_index not in all_minima) & (last_index not in all_maxima):
                    if mean_ints[last_index] > mean_ints[last_index-1]:
                        all_maxima = np.append(all_maxima,last_index)
                    else:
                        all_minima = np.append(all_minima,last_index)
                
        #        all_maxima = all_maxima[:15]
        #        print(all_minima)
        #        print(all_maxima)
                start_point = min(min(all_maxima),min(all_minima))
                if start_point in all_maxima:
                    rel_ex_type = "max"
                else:
                    rel_ex_type = "min"
                    
                for min_diff_for_cycle in all_thresholds:
                    all_fluct_points = [start_point]
                    all_flucts = []
                    #only continue, if minimum before was found (otherwise maximum could not be chained to last minimum)
                    current_point, all_flucts, all_fluct_points = move_further(start_point,all_flucts,
                                                                               all_minima, all_maxima,
                                                                               all_fluct_points,mean_ints,
                                                                               rel_ex_type,min_diff_for_cycle)
                    nb_of_cycles = max(len(all_flucts)/2,0)
                    cycle_amplitude = np.mean(all_flucts)
                    
                    # print(all_flucts)
                    # print(all_fluct_points)
                    # print(min_diff_for_cycle)
                
                    new_row = []
                    new_row.append(one_branch['date'].iloc[0])
                    new_row.append(one_branch['neuron'].iloc[0])
                    new_row.append(treatment)
                    new_row.append(one_branch['origin'].iloc[0])
                    new_row.append(one_branch['branch'].iloc[0])
                    new_row.append(max_time)
                    new_row.append(nb_of_cycles)
                    new_row.append(nb_of_cycles/(max_time/60))
                    new_row.append(cycle_amplitude)
                    new_row.append(min_diff_for_cycle)
                    # if (len(all_flucts) > -1) & (min_diff_for_cycle == 0.3):
                    #     fig_title = str(one_branch['date'].iloc[0])+"-"+one_branch['neuron'].iloc[0]+"_"+str(one_branch['origin'].iloc[0])+"-"+str(one_branch['branch'].iloc[0])+"_"+str(nb)
                        
                    #     plt.figure(figsize=(7,5))
                    #     plot = sb.lineplot(x="time",y="int_norm",data = one_branch.reset_index().drop(axis=1,labels="level_0").reset_index(),linewidth=5,color="white")
                    #     plot.set_title(fig_title)
                    #     axes = plt.gca()
                    #     plt.yticks(np.arange(0.5,1.6,0.2))
                    #     figure = plot.get_figure()
                    #     figure.savefig(lineplot_path+treatment+fig_title+".png")
                    #     print(treatment)
                    #     print(one_index)
                    #     print(all_fluct_points)
                    #     print(all_flucts)
        
                    data.loc[len(data)] = new_row
            #        print(treatment+"-"+str(len(all_flucts)/max_time))
            #        print(mean_ints[all_fluct_points])         
                    nb += 1
#        break
####
##
test_data_blebb = data.loc[data["treatment"] == "pablebb"]
all_cells_blebb = test_data_blebb[["date", "neuron"]].drop_duplicates()
print(all_cells_blebb)


data.to_csv(input_path+"\\"+file_name.replace(".csv","_analysis.csv"))
# all_data.to_csv(input_path+"\\"+file_name.replace(".csv","_growth_ints.csv"))



# all_data = pd.read_csv(input_path+"\\"+file_name.replace(".csv","_growth_ints.csv"))
# data = pd.read_csv(input_path+"\\"+file_name.replace(".csv","_analysis.csv"))


# data["growth_diff_int"] = np.nan
# data["growth_high-int"] = np.nan
# data["growth_low-int"] = np.nan
# all_cells = data[['date','neuron']].drop_duplicates()
# new_all_data = pd.DataFrame()
# for neurite in all_cells.values:
#     print(neurite)
#     #& (all_data['origin'] == neurite[2])& (all_data['branch'] == neurite[3])
#     neurite_data = all_data.loc[(all_data['date'] == neurite[0]) & (all_data['neuron'] == neurite[1])]
#     neurite_analysis = data.loc[(data['date'] == neurite[0]) & (data['neuron'] == neurite[1]) & (data['cycle_threshold'] == 0.3)]
#     neurite_data_low_int = neurite_data.loc[neurite_data['int_norm'] < 0.7].dropna()
#     neurite_data_high_int = neurite_data.loc[neurite_data['int_norm'] > 1.3].dropna()
#     if len(neurite_data_low_int) == 0:
#         print(neurite_data['int_norm'])
# #    print(neurite_data_low_int)
# #    print(len(neurite_data_low_int))
#     if (len(neurite_data_low_int) > 0) & (len(neurite_data_high_int) > 0):
#         growth_high_int = neurite_data_high_int['growth_rate'].mean()
#         growth_low_int = neurite_data_low_int['growth_rate'].mean()
#         growth_diff = (growth_high_int - growth_low_int)
#         # & (data['origin'] == neurite[2]) & (data['branch'] == neurite[3]) 
#         data.loc[(data['date'] == neurite[0]) & (data['neuron'] == neurite[1])& (data['cycle_threshold'] == 0.3),"growth_diff_int"] = growth_diff
#         data.loc[(data['date'] == neurite[0]) & (data['neuron'] == neurite[1]) & (data['cycle_threshold'] == 0.3),"growth_high-int"] = growth_high_int
#         data.loc[(data['date'] == neurite[0]) & (data['neuron'] == neurite[1]) & (data['cycle_threshold'] == 0.3),"growth_low-int"] = growth_low_int
# #    print(cell_analysis)
#     if (neurite_analysis['cycles'].iloc[0] > 2):
#         new_all_data = pd.concat((new_all_data,neurite_data))
# #
# data.to_csv(input_path+"\\"+file_name.replace(".csv","_analysis_growth-diff.csv"))    
       

# data = pd.read_csv(input_path+"\\"+file_name.replace(".csv","_analysis_growth-diff.csv")) 


# #data = data.loc[(data['growth_diff_int'] < -0.2) | (data['growth_diff_int'] > 0.2)]


# #print(data.loc[data.treatment == "dmso",'growth_diff_int'].mean())

# plt.figure(figsize=(3,10))
# bp = sb.boxplot(y=data.loc[data.treatment == "dmso",'growth_diff_int'])
# bp.set_ylim(-0.5,0.75)
# plt.figure(figsize=(3,10))
# bp = sb.boxplot(y=data.loc[data.treatment == "dmso",'growth_high-int'])
# bp.set_ylim(-0.5,0.75)
# plt.figure(figsize=(3,10))
# bp = sb.boxplot(y=data.loc[data.treatment == "dmso",'growth_low-int'])
# bp.set_ylim(-0.5,0.75)

# new_all_data.to_csv(input_path+"\\"+file_name.replace(".csv","_growth_int_filtered.csv"))
# all_data = pd.read_csv(input_path+"\\"+file_name.replace(".csv","_growth_int_filtered.csv"))

# all_data_dmso = all_data.loc[all_data.treatment == "dmso"]

# #all_data_dmso = all_data_dmso.loc[(all_data_dmso['growth_rate'] > 0.1) | (all_data_dmso['growth_rate'] < -0.1)]

# def categorize_int(intensity):
#     if intensity > 1.3:
#         return "high"
#     elif intensity < 0.7:
#         return "low"

# data_dmso = data.loc[data['treatment'] == "dmso"]
# data_dmso = data_dmso.dropna()
# print(len(data_dmso))

# #thresholds = [0,0.05,0.1,0.2,0.3,0.4,0.5]
# #for threshold in thresholds:
# #    print(threshold)
# #    print(len(data_dmso.loc[data_dmso['growth_diff_int'] > threshold])/len(data_dmso))
# #    print(len(data_dmso.loc[data_dmso['growth_diff_int'] < - threshold])/len(data_dmso))

# print(data_dmso['growth_diff_int'].mean())
# print(data_dmso['growth_high-int'].mean())
# print(data_dmso['growth_low-int'].mean())

# thresholds = [0.1,0.2,0.3,0.4,0.5]
# for threshold in thresholds:
#     print(threshold)
#     chance_of_growth_low = len(data_dmso.loc[data_dmso['growth_low-int'] > threshold])/len(data_dmso)
#     chance_of_retraction_low = len(data_dmso.loc[data_dmso['growth_low-int'] < - threshold])/len(data_dmso) 
    
#     chance_of_growth_high = len(data_dmso.loc[data_dmso['growth_high-int'] > threshold])/len(data_dmso)
#     chance_of_retraction_high = len(data_dmso.loc[data_dmso['growth_high-int'] < -threshold])/len(data_dmso)
    
#     print(chance_of_growth_low)
#     print(chance_of_retraction_low)
#     print(chance_of_growth_high)
#     print(chance_of_retraction_high)


# all_data_dmso['group'] = all_data_dmso['int_norm'].apply(categorize_int)

# low_int_data = all_data_dmso.loc[all_data_dmso['int_norm'] < 0.7]

# high_int_data = all_data_dmso.loc[all_data_dmso['int_norm'] > 1.3]

# plt.figure(figsize=(3,10))
# boxplot = sb.boxplot(y="growth_rate",x="group",data=all_data_dmso,showmeans=True)
# boxplot.set_ylim(-1,1)

# thresholds = [0.1,0.2,0.3,0.4,0.5]
# for threshold in thresholds:
#     print(threshold)
#     chance_of_growth_low = len(low_int_data.loc[low_int_data['growth_rate'] > threshold])/len(low_int_data)
#     chance_of_retraction_low = len(low_int_data.loc[low_int_data['growth_rate'] < - threshold])/len(low_int_data) 
    
#     chance_of_growth_high = len(high_int_data.loc[high_int_data['growth_rate'] > threshold])/len(high_int_data)
#     chance_of_retraction_high = len(high_int_data.loc[high_int_data['growth_rate'] < -threshold])/len(high_int_data)
    
#     print(chance_of_growth_low)
#     print(chance_of_retraction_low)
#     print(chance_of_growth_high)
#     print(chance_of_retraction_high)

# #print(len(high_ing))
# print(high_int_data['growth_rate_raw_corr'].mean())
# print(low_int_data['growth_rate_raw_corr'].mean())





# col1 = "int_norm"
# col2 = "growth_rate_raw_corr"
# #
# all_data_dmso = all_data_dmso.dropna(subset=[col1,col2])
# print(stats.pearsonr(all_data_dmso[col1],all_data_dmso[col2]))

# plt.figure(figsize=(8,10))
# sb.scatterplot(x=all_data_dmso[col1],y=all_data_dmso[col2])


#all_data = all_data.loc[all_data['growth_rate'] < 0]


#
#for threshold in all_thresholds:
#    print("--------------THRESHOLD: "+str(threshold)+"------------------")
#    data_crop = data.loc[data['cycle_threshold'] == threshold]
#    print(data_crop.groupby("treatment").mean())
#    
#    data_crop2 = data_crop.loc[data['cycles'] > 0]
#    print(data_crop2.groupby("treatment").mean())
#    
#    data_crop2 = data_crop.loc[data['cycles'] < 1]
#    print(data_crop.groupby("treatment").count())
#    print(data_crop2.groupby("treatment").count())
#
#
