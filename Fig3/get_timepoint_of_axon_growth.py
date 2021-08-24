import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import signal
from scipy import stats
import os

def get_length_difference_axon(data):
    axon_length = data.loc[data["neurite"] == 0, "length_um"]
    max_length_MN = data.loc[data["neurite"] != 0, "length_um"].max()
    return axon_length - max_length_MN

def get_last_time_below_growth(data):
    times_below_growth = data.loc[data["length_um"] == False, "time"]
    if len(times_below_growth) == 0:
        return 0
    else:
        return data.loc[data["length_um"] == False, "time"].iloc[-1]

def get_first_frame_with_zero_growth(data):
    data = data.reset_index()
    out_of_frame = data.loc[(data["growth"] == 0) & (data["neurite"] == 0), "time"]
    #if axon never goes out of frame, return maximum time
    if len(out_of_frame) == 0:
        return data["time"].max()
    else:
        last_time = 0
        ticker = 0
        #check if four consecutive frames seem out of frame
        for time in out_of_frame:
            if time == last_time +1:
                ticker += 1
                if ticker > 2:
                    return time
                else:
                    last_time = time
            else:
                last_time = time
                ticker = 0
        return data["time"].max()
    
    

min_length_diff_axon = 10
min_length_axon = 50


data_path = "C:\\Users\\Maxsc\\ownCloud\\01ANALYSIS\\TUBB\\MTRF-longterm\\"
data_path = "C:\\Users\\Maxsc\\Nextcloud\\01ANALYSIS\\TUBB\\MTRF-longterm\\"
data_path = os.path.abspath("C:\\Users\\Maxsc\\Nextcloud\\01ANALYSIS\\"
                             "TUBB\\MTRF-fluctuations-longterm_AXvsMN\\")
file_name = "MT-RF_growth_longterm_allData.csv"

neuron_columns = ["date", "neuron"]
time_columns = [*neuron_columns, "time"]

data = pd.read_csv(os.path.join(data_path ,file_name))

#for each neuron get first timepoint where growth of axon is 0
times_out_of_frame = data.groupby(neuron_columns).apply(get_first_frame_with_zero_growth)

#exclude all data beyond that to prevent other neurites from seemingly catching up
#while in fact the axon grew out of frame
#didnt work to set index on data first for some reason...
for neuron_idx in times_out_of_frame.index.drop_duplicates().values:
    neuron_idx = tuple(neuron_idx)
    time_out_of_frame = times_out_of_frame.loc[neuron_idx]
    data.loc[(data["date"] == neuron_idx[0]) & (data["neuron"] == neuron_idx[1]) & (data["time"] > time_out_of_frame), "length_um"] = np.nan

data = data.dropna(subset=["length_um", "growth"])

#for each time point calculate the different in length 
#from the axon to the longest minor neurite

length_diff_AX = data.groupby(time_columns).apply(get_length_difference_axon)

#to prevent the axon growth timepoint to be too early, exclude data
#since data with these row values would otherwise evaluate as FALSE in next eval
#thereby it would look like at these timepoints the axon was not long enough
length_diff_AX = length_diff_AX.dropna()
length_diff_AX = length_diff_AX.loc[length_diff_AX > 0]


# def print_res(tmp_data):
#     growth = data.set_index(neuron_columns).loc[tmp_data.name]
#     growth = growth.loc[growth["neurite"] == 0]
#     print(growth.loc[growth["time"] > 1300, "growth"].iloc[:30])
#     print(tmp_data.loc[tmp_data["time"] > 1300].iloc[:30])
# length_diff_AX.reset_index().groupby(neuron_columns).apply(print_res)


#now for each neuron find out at which timepoint
#the axon started to be consistently longer than all other neurites
#first set the length_difference to 1 when its larger than the min_length_difference
length_diff_AX_higher = length_diff_AX  > min_length_diff_axon

#get the last time when the axon was not min_length_diff longer
time_axon_growth = length_diff_AX_higher.reset_index().groupby(neuron_columns).apply(get_last_time_below_growth)

time_axon_growth.to_csv(os.path.join(data_path,  "axon_outgrowth_time.csv"))
print(time_axon_growth)