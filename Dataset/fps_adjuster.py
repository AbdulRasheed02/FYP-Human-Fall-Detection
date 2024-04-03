import pandas as pd
import os
import glob
import shutil
import math
import re
import sys
import numpy as np
from glob import glob


# Importing constants from parameters.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Code.parameters import dataset_directory

sys.path.remove(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def adjust_fps(dirs, destination_folder, adjustment_factor, dirs_type):

    print("\nDirectory Type - {}, {} Folders\n".format(dirs_type, len(dirs)))

    # Create 'Fall' / 'NonFall' folder if it does not exist.
    # Ex - F:\FYP Datasets\MUVIM\Dataset\Fall-Data\FPS-Adjusted\ONI_IR\Fall
    dirs_folder = "{}\{}".format(destination_folder, dirs_type)
    if not os.path.exists(dirs_folder):
        os.makedirs(dirs_folder)

    # Loop through each Fall / NonFall folder
    for i, dir in enumerate(dirs):
        old_frames = glob(dir + "/*.jpg") + glob(dir + "/*.png")  # All the old frames
        # Sort the frames
        old_frames.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

        dir_num = re.findall("[0-9]+", dir)[0]  # Extract directory number
        dir_folder = "{}\{}{}".format(dirs_folder, dirs_type, dir_num)
        # Create folder for current video trial if it does not exist. (Fall0, Fall1, NonFall1, NonFall2 etc)
        # Ex - F:\FYP Datasets\MUVIM\Dataset\Fall-Data\FPS-Adjusted\ONI_IR\Fall\Fall0
        if not os.path.exists(dir_folder):
            os.makedirs(dir_folder)

        frame = 0  # For numbering the new frames
        """
        If adjustment factor is 2.5, count will be 0, 2.5, 5, 7.5 etc..
        If adjustment factor is 0.5, count will be 0, 0.5, 1, 1.5 etc..
        """
        for count in np.arange(0, len(old_frames), adjustment_factor):
            """
            If count is 0, 2.5, 5, 7.5 etc, adjusted_index will be 0, 2, 5, 7. So frames will get skipped
            If count is 0, 0.5, 1, 1.5 etc, adjusted_index will be 0, 0, 1, 1. So frames will get repeated
            """
            adjusted_index = math.floor(count)
            # print(count, adjusted_index)
            shutil.copyfile(
                old_frames[adjusted_index], dir_folder + "/frame" + str(frame) + old_frames[adjusted_index][-4:]
            )
            frame += 1
            count += adjustment_factor

        print(
            "{}/{} Completed : {}{} - {} to {} frames ".format(
                i + 1,
                len(dirs),
                dirs_type,
                dir_num,
                len(old_frames),
                frame,
            )
        )


def adjust_fall_labels(source_folder, destination_folder, adjustment_factor):
    labels = pd.read_csv("{}/Labels.csv".format(source_folder))
    """
        If adjustment factor is 0.5 -
            Start - 350, 23, 43 will be 700, 46, 86 (Multiplied by 2)
            Stop - 380, 33, 51 will be 761, 47, 102 (Multiplied by 2 + 1)
        If adjustment factor is 2.5, 
            Start - 2449, 2533, 2776 will be 980, 1014, 1111 (Multiplied by 2/5 and ceil)
            Stop - 2466, 2570, 2797 will be 986, 1028, 1118 (Multiplied by 2/5 and floor)
    """
    if adjustment_factor == 0.5:
        labels["Start"] = labels["Start"].apply(lambda x: x * (1 / adjustment_factor))
        labels["Stop"] = labels["Stop"].apply(lambda x: (x * (1 / adjustment_factor)) + 1)
    else:
        labels["Start"] = labels["Start"].apply(lambda x: math.ceil(x * (1 / adjustment_factor)))
        labels["Stop"] = labels["Stop"].apply(lambda x: math.floor(x * (1 / adjustment_factor)))

    labels.to_csv("{}/Labels.csv".format(destination_folder))

    print("\nCompleted : FPS Adjusted Labels.csv file")


# Directory names of the raw dataset from the Dataset/Fall-Data/Base folder
# list_of_files = ['Thermal','ONI_IR','IP']
list_of_files = ["IP"]

for i in range(len(list_of_files)):
    dset = list_of_files[i]

    source_folder = "{}\Dataset\Fall-Data\Base\{}".format(dataset_directory, dset)
    destination_folder = "{}\Dataset\Fall-Data\FPS-Adjusted\{}".format(dataset_directory, dset)

    # All fall and adl folders
    falls = glob(source_folder + "\\Fall\\Fall*")
    adl = glob(source_folder + "\\NonFall\\NonFall*")

    # adjustment_factor = current_frame_rate/desired_frame_rate
    if dset == "Thermal":
        adjustment_factor = 0.5  # (adjustment_factor = 4/8)
    else:
        adjustment_factor = 2.5  # (adjustment_factor = 20/8)

    print("\nFPS Adjustment for - {}, Adjustment Factor - {}".format(dset, adjustment_factor))

    adjust_fps(falls, destination_folder, adjustment_factor, dirs_type="Fall")
    adjust_fps(adl, destination_folder, adjustment_factor, dirs_type="NonFall")

    adjust_fall_labels(source_folder, destination_folder, adjustment_factor)
