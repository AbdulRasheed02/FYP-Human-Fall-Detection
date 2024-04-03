import pandas as pd
import numpy as np
from functools import reduce
import sys
import os
import re
from glob import glob

# Importing constants from parameters.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from Code.parameters import project_directory, dataset_directory

sys.path.remove(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

"""
load in all datasets manually
load in label csvs for all datasets
then compare the labels for the fall_number as this is the unique ID and should also match TOD 
add option to choose only day or only night examples 
save list of common falls in a csv that you can read in the pipeline in order to ensure that it is a common fall
"""

# For Fall Videos

modalities = ["Thermal", "ONI_IR", "IP"]
dataframes = []

for modality in modalities:
    # Labels.csv file
    labels = pd.read_csv("{}/Dataset/Fall-Data/Base/{}/Labels.csv".format(dataset_directory, modality))

    """
    # Corner Case - There are video id's in labels.csv file for which corresponding folder does not exist in the dataset. 
    # So filter out those video id's
    """
    # Extract video IDs from labels.csv
    labels_video_ids = labels["Video"].unique()
    # Find all the video id's from the dataset
    dataset_video_ids = []
    for root, dirs, files in os.walk("{}/Dataset/Fall-Data/Base/{}/Fall".format(dataset_directory, modality)):
        for dir in dirs:
            # Ignore folders which have less than 10 videos (As this causes inconsistency for multi modality in dataset_loader.py file)
            if len(glob(f"{root}/{dir}/*")) >= 10:
                dataset_video_id = re.findall("[0-9]+", dir)[0]
                dataset_video_ids.append(int(dataset_video_id))

    # Only select video id's which are common to both
    filtered_labels_video_ids = set(labels_video_ids) & set(dataset_video_ids)

    # Filter the labels dataframe according to the new video ids.
    filtered_labels = labels[labels["Video"].isin(filtered_labels_video_ids)]
    dataframes.append(filtered_labels)

df_merged = reduce(lambda left, right: pd.merge(left, right, on=["Video"], how="inner"), dataframes)
# print(df_merged.keys())

shared_fall_vids = df_merged.Video.unique()
np.savetxt("{}/Dataset/Metadata/shared_fall_vids_1.txt".format(project_directory), shared_fall_vids)
print("Shared Fall Vids - {}".format(len(shared_fall_vids)))

night_vids = df_merged[df_merged["ToD_x"] == 0].Video.unique()
np.savetxt("{}/Dataset/Metadata/night_fall_vids_1.txt".format(project_directory), night_vids)
print("Shared Night Fall Vids - {}".format(len(night_vids)))

day_vids = df_merged[df_merged["ToD_x"] == 1].Video.unique()
np.savetxt("{}/Dataset/Metadata/day_fall_vids_1.txt".format(project_directory), day_vids)
print("Shared Day Fall Vids - {}".format(len(day_vids)))

# For ADL Videos
# Ignore folders which have less than 10 videos (As this causes inconsistency for multi modality in dataset_loader.py file)

Thermal_falls = glob("{}/Dataset/Fall-Data/Base/{}/NonFall/*".format(dataset_directory, modalities[0]))
# [63:] Extract folder number from path
Thermal_falls = [Thermal_fall[63:] for Thermal_fall in Thermal_falls if len(glob(f"{Thermal_fall}/*")) >= 10]

ONI_IR_falls = glob("{}/Dataset/Fall-Data/Base/{}/NonFall/*".format(dataset_directory, modalities[1]))
# [62:] Extract folder number from path
ONI_IR_falls = [ONI_IR_fall[62:] for ONI_IR_fall in ONI_IR_falls if len(glob(f"{ONI_IR_fall}/*")) >= 10]

IP_falls = glob("{}/Dataset/Fall-Data/Base/{}/NonFall/*".format(dataset_directory, modalities[2]))
# [58:] Extract folder number from path
IP_falls = [IP_fall[58:] for IP_fall in IP_falls if len(glob(f"{IP_fall}/*")) >= 10]

shared_adl_vids = np.asarray(list(set(IP_falls) & set(ONI_IR_falls) & set(Thermal_falls)))
shared_adl_vids = np.sort(shared_adl_vids)
np.savetxt("{}/Dataset/Metadata/shared_adl_vids_1.txt".format(project_directory), shared_adl_vids, fmt="%s")
print("Shared ADL Vids - {}".format(len(shared_adl_vids)))
